#!/usr/bin/env python3

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration


MODEL_ID = "RedHatAI/whisper-large-v3-turbo-quantized.w8a8"


def to_np(tensor, dtype=None):
    if tensor is None:
        return None
    value = tensor.detach().cpu().numpy()
    if dtype is not None:
        value = value.astype(dtype)
    return value


def load_encoder(model_path: str):
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
    )
    return model.model.encoder


def build_program(encoder):
    conv1_w = to_np(encoder.conv1.weight, np.float16)
    conv1_b = to_np(encoder.conv1.bias, np.float16)
    conv2_w = to_np(encoder.conv2.weight, np.float16)
    conv2_b = to_np(encoder.conv2.bias, np.float16)
    pos = to_np(encoder.embed_positions.weight, np.float16)[None, :, :]
    final_ln_gamma = to_np(encoder.layer_norm.weight, np.float16)
    final_ln_beta = to_np(encoder.layer_norm.bias, np.float16)
    eps = np.float16(np.finfo(np.float16).eps)
    scale_denom = np.float16(127.0)

    def dyn_token_fake_quant(x, name):
        max_abs = mb.reduce_max(
            x=mb.abs(x=x, name=f"{name}_abs"),
            axes=[2],
            keep_dims=True,
            name=f"{name}_amax",
        )
        scale = mb.real_div(x=max_abs, y=scale_denom, name=f"{name}_scale_raw")
        scale = mb.maximum(x=scale, y=eps, name=f"{name}_scale")
        x = mb.real_div(x=x, y=scale, name=f"{name}_div")
        x = mb.round(x=x, name=f"{name}_round")
        x = mb.clip(x=x, alpha=np.float16(-128.0), beta=np.float16(127.0), name=f"{name}_clip")
        x = mb.cast(x=x, dtype="int8", name=f"{name}_q")
        x = mb.cast(x=x, dtype="fp16", name=f"{name}_dq_cast")
        return mb.mul(x=x, y=scale, name=f"{name}_dq")

    def linear_weight(module, name):
        return mb.constexpr_affine_dequantize(
            quantized_data=to_np(module.weight, np.int8),
            zero_point=np.array(0, dtype=np.int8),
            scale=to_np(module.weight_scale, np.float16).reshape(-1),
            axis=np.int32(0),
            name=f"{name}_weight",
        )

    def quantized_linear(x, module, name):
        x = dyn_token_fake_quant(x, f"{name}_act")
        weight = linear_weight(module, name)
        bias = to_np(module.bias, np.float16)
        if bias is None:
            return mb.linear(x=x, weight=weight, name=name)
        return mb.linear(x=x, weight=weight, bias=bias, name=name)

    def layer_norm(x, gamma, beta, epsilon, name):
        return mb.layer_norm(
            x=x,
            axes=[-1],
            gamma=gamma,
            beta=beta,
            epsilon=np.float16(epsilon),
            name=name,
        )

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 128, 3000), dtype=ct.converters.mil.mil.types.fp32)],
        opset_version=ct.target.iOS17,
    )
    def prog(logmel_data):
        x = mb.cast(x=logmel_data, dtype="fp16", name="input_fp16")
        x = mb.conv(
            x=x,
            weight=conv1_w,
            bias=conv1_b,
            strides=[1],
            pad_type="custom",
            pad=[1, 1],
            name="conv1",
        )
        x = mb.gelu(x=x, mode="EXACT", name="conv1_gelu")
        x = mb.conv(
            x=x,
            weight=conv2_w,
            bias=conv2_b,
            strides=[2],
            pad_type="custom",
            pad=[1, 1],
            name="conv2",
        )
        x = mb.gelu(x=x, mode="EXACT", name="conv2_gelu")
        x = mb.transpose(x=x, perm=[0, 2, 1], name="post_conv_transpose")
        x = mb.add(x=x, y=pos, name="add_positional_embedding")

        for idx, layer in enumerate(encoder.layers):
            prefix = f"layers_{idx}"
            residual = x
            x_ln = layer_norm(
                x,
                to_np(layer.self_attn_layer_norm.weight, np.float16),
                to_np(layer.self_attn_layer_norm.bias, np.float16),
                layer.self_attn_layer_norm.eps,
                f"{prefix}_self_attn_ln",
            )

            q = quantized_linear(x_ln, layer.self_attn.q_proj, f"{prefix}_q_proj")
            k = quantized_linear(x_ln, layer.self_attn.k_proj, f"{prefix}_k_proj")
            v = quantized_linear(x_ln, layer.self_attn.v_proj, f"{prefix}_v_proj")

            q = mb.mul(x=q, y=np.float16(layer.self_attn.scaling), name=f"{prefix}_q_scale")

            q = mb.reshape(x=q, shape=[1, 1500, layer.self_attn.num_heads, layer.self_attn.head_dim], name=f"{prefix}_q_reshape")
            k = mb.reshape(x=k, shape=[1, 1500, layer.self_attn.num_heads, layer.self_attn.head_dim], name=f"{prefix}_k_reshape")
            v = mb.reshape(x=v, shape=[1, 1500, layer.self_attn.num_heads, layer.self_attn.head_dim], name=f"{prefix}_v_reshape")

            q = mb.transpose(x=q, perm=[0, 2, 1, 3], name=f"{prefix}_q_transpose")
            k = mb.transpose(x=k, perm=[0, 2, 1, 3], name=f"{prefix}_k_transpose")
            v = mb.transpose(x=v, perm=[0, 2, 1, 3], name=f"{prefix}_v_transpose")

            attn = mb.matmul(x=q, y=k, transpose_y=True, name=f"{prefix}_attn_scores")
            attn = mb.softmax(x=attn, axis=-1, name=f"{prefix}_attn_softmax")
            attn = mb.matmul(x=attn, y=v, name=f"{prefix}_attn_values")
            attn = mb.transpose(x=attn, perm=[0, 2, 1, 3], name=f"{prefix}_attn_transpose")
            attn = mb.reshape(x=attn, shape=[1, 1500, 1280], name=f"{prefix}_attn_merge")

            attn = quantized_linear(attn, layer.self_attn.out_proj, f"{prefix}_out_proj")
            x = mb.add(x=residual, y=attn, name=f"{prefix}_residual_1")

            residual = x
            x_ln = layer_norm(
                x,
                to_np(layer.final_layer_norm.weight, np.float16),
                to_np(layer.final_layer_norm.bias, np.float16),
                layer.final_layer_norm.eps,
                f"{prefix}_ffn_ln",
            )
            x_ff = quantized_linear(x_ln, layer.fc1, f"{prefix}_fc1")
            x_ff = mb.gelu(x=x_ff, mode="EXACT", name=f"{prefix}_fc1_gelu")
            x_ff = quantized_linear(x_ff, layer.fc2, f"{prefix}_fc2")
            x = mb.add(x=residual, y=x_ff, name=f"{prefix}_residual_2")

        x = layer_norm(x, final_ln_gamma, final_ln_beta, encoder.layer_norm.eps, "encoder_final_ln")
        return mb.cast(x=x, dtype="fp32", name="output")

    return prog


def export_model(model_path: str, output_path: Path):
    encoder = load_encoder(model_path)
    program = build_program(encoder)
    mlmodel = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.save(str(output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/coreml-encoder-large-v3-turbo-w8a8.mlpackage"),
    )
    args = parser.parse_args()

    model_path = args.model_path or snapshot_download(MODEL_ID)
    export_model(model_path, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
