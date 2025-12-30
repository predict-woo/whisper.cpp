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

// Transcribe audio
const result = await transcribeAsync(ctx, {
  fname_inp: "./audio.wav",
  language: "en",
});

console.log(result.segments);

// Clean up
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

Transcribe audio file (Promise-based).

```typescript
interface TranscribeOptions {
  fname_inp: string;       // Path to audio file (required)
  language?: string;       // Language code (e.g., 'en', 'zh', 'auto')
  translate?: boolean;     // Translate to English
  n_threads?: number;      // Number of threads
  no_timestamps?: boolean; // Disable timestamps
  // ... see types.ts for full options
}

interface TranscribeResult {
  segments: Array<{
    start: string;         // "HH:MM:SS,mmm"
    end: string;
    text: string;
  }>;
}
```

### `createVadContext(options)`

Create a voice activity detection context.

```typescript
interface VadContextOptions {
  model: string;           // Path to Silero VAD model
  threshold?: number;      // Speech threshold (default: 0.5)
  n_threads?: number;      // Number of threads (default: 1)
  no_prints?: boolean;     // Suppress log output
}

// Usage
const vad = createVadContext({
  model: "./models/ggml-silero-v6.2.0.bin",
});

const samples = new Float32Array(vad.getWindowSamples());
// ... fill samples with 16kHz audio
const probability = vad.process(samples);

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
```

## License

MIT
