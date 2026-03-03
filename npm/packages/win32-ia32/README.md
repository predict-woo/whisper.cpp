# @whisper-cpp-node/win32-ia32

Native binary for Windows x86 (32-bit), CPU-only.

This package is automatically installed by `whisper-cpp-node` on compatible platforms.

**Do not install this package directly.** Install `whisper-cpp-node` instead.

## Requirements

- Windows 10/11 (32-bit)
- Node.js 18+ (ia32)

## Acceleration Features

This binary is CPU-only. GPU acceleration (Vulkan, OpenVINO) is not available on 32-bit Windows.

| Feature | Description |
|---------|-------------|
| **SSE2** | SIMD instructions for x86 |
| **Flash Attention** | Faster attention computation |

### Usage

```javascript
const ctx = createWhisperContext({
  model: "./models/ggml-base.en.bin",
  use_gpu: false,
});
```

## License

MIT
