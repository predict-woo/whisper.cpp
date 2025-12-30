# @whisper-cpp-node/core

Node.js bindings for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - fast speech-to-text on Apple Silicon with Core ML and Metal support.

## Features

- **Fast**: Native whisper.cpp performance with Metal GPU acceleration
- **Core ML**: Optional Apple Neural Engine support for 3x+ speedup
- **Streaming VAD**: Built-in Silero voice activity detection
- **TypeScript**: Full type definitions included
- **Self-contained**: No external dependencies, just install and use

## Requirements

- macOS 13.3+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Node.js 18+

## Installation

```bash
npm install @whisper-cpp-node/core
# or
pnpm add @whisper-cpp-node/core
```

The platform-specific binary (`@whisper-cpp-node/darwin-arm64`) is automatically installed.

## Quick Start

### File-based transcription

```typescript
import {
  createWhisperContext,
  transcribeAsync,
} from "@whisper-cpp-node/core";

// Create a context with your model
const ctx = createWhisperContext({
  model: "./models/ggml-base.en.bin",
  use_gpu: true,
});

// Transcribe audio file
const result = await transcribeAsync(ctx, {
  fname_inp: "./audio.wav",
  language: "en",
});

// Result: { segments: [["00:00:00,000", "00:00:02,500", " Hello world"], ...] }
for (const [start, end, text] of result.segments) {
  console.log(`[${start} --> ${end}]${text}`);
}

// Clean up
ctx.free();
```

### Buffer-based transcription

```typescript
import {
  createWhisperContext,
  transcribeAsync,
} from "@whisper-cpp-node/core";

const ctx = createWhisperContext({
  model: "./models/ggml-base.en.bin",
  use_gpu: true,
});

// Pass raw PCM audio (16kHz, mono, float32)
const pcmData = new Float32Array(/* your audio samples */);
const result = await transcribeAsync(ctx, {
  pcmf32: pcmData,
  language: "en",
});

for (const [start, end, text] of result.segments) {
  console.log(`[${start} --> ${end}]${text}`);
}

ctx.free();
```

## API

### `createWhisperContext(options)`

Create a persistent context for transcription.

```typescript
interface WhisperContextOptions {
  model: string;           // Path to GGML model file (required)
  use_gpu?: boolean;       // Enable GPU acceleration (default: true)
  use_coreml?: boolean;    // Enable Core ML on macOS (default: false)
  flash_attn?: boolean;    // Enable Flash Attention (default: false)
  gpu_device?: number;     // GPU device index (default: 0)
  dtw?: string;            // DTW preset for word timestamps
  no_prints?: boolean;     // Suppress log output (default: false)
}
```

### `transcribeAsync(context, options)`

Transcribe audio (Promise-based). Accepts either a file path or PCM buffer.

```typescript
// File input
interface TranscribeOptionsFile {
  fname_inp: string;       // Path to audio file
  // ... common options
}

// Buffer input
interface TranscribeOptionsBuffer {
  pcmf32: Float32Array;    // Raw PCM (16kHz, mono, float32, -1.0 to 1.0)
  // ... common options
}

// Common options (partial list - see types.ts for full options)
interface TranscribeOptionsBase {
  // Language
  language?: string;       // Language code ('en', 'zh', 'auto')
  translate?: boolean;     // Translate to English
  detect_language?: boolean; // Auto-detect language

  // Threading
  n_threads?: number;      // CPU threads (default: 4)
  n_processors?: number;   // Parallel processors

  // Audio processing
  offset_ms?: number;      // Start offset in ms
  duration_ms?: number;    // Duration to process (0 = all)

  // Output control
  no_timestamps?: boolean; // Disable timestamps
  max_len?: number;        // Max segment length (chars)
  max_tokens?: number;     // Max tokens per segment
  split_on_word?: boolean; // Split on word boundaries
  token_timestamps?: boolean; // Include token-level timestamps

  // Sampling
  temperature?: number;    // Sampling temperature (0.0 = greedy)
  beam_size?: number;      // Beam search size (-1 = greedy)
  best_of?: number;        // Best-of-N sampling

  // Thresholds
  entropy_thold?: number;  // Entropy threshold
  logprob_thold?: number;  // Log probability threshold
  no_speech_thold?: number; // No-speech probability threshold

  // Context
  prompt?: string;         // Initial prompt text
  no_context?: boolean;    // Don't use previous context

  // VAD preprocessing
  vad?: boolean;           // Enable VAD preprocessing
  vad_model?: string;      // Path to VAD model
  vad_threshold?: number;  // VAD threshold (0.0-1.0)
  vad_min_speech_duration_ms?: number;
  vad_min_silence_duration_ms?: number;
  vad_speech_pad_ms?: number;

  // Callbacks
  progress_callback?: (progress: number) => void;
}

// Result
interface TranscribeResult {
  segments: TranscriptSegment[];
}

// Segment is a tuple: [start, end, text]
type TranscriptSegment = [string, string, string];
// Example: ["00:00:00,000", "00:00:02,500", " Hello world"]
```

### `createVadContext(options)`

Create a voice activity detection context for streaming audio.

```typescript
interface VadContextOptions {
  model: string;           // Path to Silero VAD model
  threshold?: number;      // Speech threshold (default: 0.5)
  n_threads?: number;      // Number of threads (default: 1)
  no_prints?: boolean;     // Suppress log output
}

interface VadContext {
  getWindowSamples(): number;  // Returns 512 (32ms at 16kHz)
  getSampleRate(): number;     // Returns 16000
  process(samples: Float32Array): number;  // Returns probability 0.0-1.0
  reset(): void;               // Reset LSTM state
  free(): void;                // Release resources
}
```

#### VAD Example

```typescript
import { createVadContext } from "@whisper-cpp-node/core";

const vad = createVadContext({
  model: "./models/ggml-silero-v6.2.0.bin",
  threshold: 0.5,
});

const windowSize = vad.getWindowSamples(); // 512 samples

// Process audio in 32ms chunks
function processAudioChunk(samples: Float32Array) {
  const probability = vad.process(samples);
  if (probability >= 0.5) {
    console.log("Speech detected!", probability);
  }
}

// Reset when starting new audio stream
vad.reset();

// Clean up when done
vad.free();
```

## Core ML Acceleration

For 3x+ faster encoding on Apple Silicon:

1. Generate a Core ML model:
   ```bash
   pip install ane_transformers openai-whisper coremltools
   ./models/generate-coreml-model.sh base.en
   ```

2. Place it next to your GGML model:
   ```
   models/ggml-base.en.bin
   models/ggml-base.en-encoder.mlmodelc/
   ```

3. Enable Core ML:
   ```typescript
   const ctx = createWhisperContext({
     model: "./models/ggml-base.en.bin",
     use_coreml: true,
   });
   ```

## Models

Download models from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp):

```bash
# Base English model (~150MB)
curl -L -o models/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# Large v3 Turbo quantized (~500MB)
curl -L -o models/ggml-large-v3-turbo-q4_0.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q4_0.bin

# Silero VAD model (for streaming VAD)
curl -L -o models/ggml-silero-v6.2.0.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-silero-v6.2.0.bin
```

## License

MIT
