import { promisify } from "util";
import { loadNativeAddon } from "./loader";
import type {
  WhisperAddon,
  WhisperContext,
  WhisperContextOptions,
  VadContext,
  VadContextOptions,
  TranscribeOptions,
  TranscribeResult,
  GpuDevice,
} from "./types";

// Re-export types
export type {
  WhisperContextOptions,
  VadContextOptions,
  TranscribeOptions,
  TranscribeOptionsBase,
  TranscribeOptionsFile,
  TranscribeOptionsBuffer,
  TranscribeResult,
  TranscriptSegment,
  StreamingSegment,
  StreamingToken,
  WhisperContext,
  VadContext,
  WhisperContextConstructor,
  VadContextConstructor,
  GpuDevice,
} from "./types";

// Load native addon
const addon: WhisperAddon = loadNativeAddon();

// Export native constructors with different names to avoid conflict
export const WhisperContextClass = addon.WhisperContext;
export const VadContextClass = addon.VadContext;

// Original callback-based transcribe
export const transcribe = addon.transcribe;

// Promisified version for async/await
export const transcribeAsync = promisify(addon.transcribe) as (
  context: WhisperContext,
  options: TranscribeOptions
) => Promise<TranscribeResult>;

/**
 * Create a new WhisperContext
 *
 * @example
 * ```typescript
 * const ctx = createWhisperContext({
 *   model: './models/ggml-base.en.bin',
 *   use_gpu: true,
 *   use_coreml: true,
 * });
 *
 * const result = await transcribeAsync(ctx, {
 *   fname_inp: './audio.wav',
 *   language: 'en',
 * });
 *
 * console.log(result.segments);
 * ctx.free();
 * ```
 */
export function createWhisperContext(
  options: WhisperContextOptions
): WhisperContext {
  return new addon.WhisperContext(options);
}

/**
 * Create a new VadContext for voice activity detection
 *
 * @example
 * ```typescript
 * const vad = createVadContext({
 *   model: './models/ggml-silero-v6.2.0.bin',
 *   threshold: 0.5,
 * });
 *
 * const samples = new Float32Array(512);
 * const probability = vad.process(samples);
 *
 * vad.free();
 * ```
 */
export function createVadContext(options: VadContextOptions): VadContext {
  return new addon.VadContext(options);
}

/**
 * Enumerate available GPU backend devices.
 * Returns an array of GPU/IGPU devices with their properties.
 * The `index` field matches what `gpu_device` option expects in WhisperContextOptions.
 * Never throws — returns an empty array if no GPUs are available or on any error.
 *
 * @example
 * ```typescript
 * const gpus = getGpuDevices();
 * for (const gpu of gpus) {
 *   console.log(`[${gpu.index}] ${gpu.description} (${gpu.type}, ${(gpu.memory_total / 1e9).toFixed(1)} GB)`);
 * }
 * // Use a specific GPU:
 * const ctx = createWhisperContext({ model: '...', gpu_device: gpus[1].index });
 * ```
 */
export function getGpuDevices(): GpuDevice[] {
  return addon.getGpuDevices();
}

// Default export with all functionality
export default {
  WhisperContext: addon.WhisperContext,
  VadContext: addon.VadContext,
  transcribe: addon.transcribe,
  transcribeAsync,
  createWhisperContext,
  createVadContext,
  getGpuDevices,
};
