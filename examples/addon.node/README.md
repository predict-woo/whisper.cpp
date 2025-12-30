# whisper.cpp Node.js addon

This is an addon demo that can **perform whisper model reasoning in `node` and `electron` environments**, based on [cmake-js](https://github.com/cmake-js/cmake-js).
It can be used as a reference for using the whisper.cpp project in other node projects.

This addon supports:
- **Persistent Context**: Load a model once and transcribe multiple files efficiently
- **Voice Activity Detection (VAD)**: Process only speech segments for improved performance
- **Full Server Options**: All parameters supported by whisper-server are available

## Install

```shell
npm install
```

## Compile

Make sure it is in the project root directory and compiled with cmake-js.

```shell
npx cmake-js compile -T addon.node -B Release
```

For Electron addon and cmake-js options, you can see [cmake-js](https://github.com/cmake-js/cmake-js) and make very few configuration changes.

> Such as appointing special cmake path:
> ```shell
> npx cmake-js compile -c 'xxx/cmake' -T addon.node -B Release
> ```

## API Overview

The addon exports three main components:

| Export | Description |
|--------|-------------|
| `WhisperContext` | Class for persistent model context (recommended for batch processing) |
| `transcribe()` | Async function that uses a WhisperContext for transcription |
| `whisper()` | Legacy function that loads/frees model per call (backwards compatible) |

## Persistent Context API (Recommended)

For batch processing or applications that need to transcribe multiple files, use the persistent context API. This loads the model once and reuses it for all transcriptions.

### Basic Usage

```javascript
const path = require("path");
const { WhisperContext, transcribe } = require(path.join(
  __dirname,
  "../../build/Release/addon.node"
));
const { promisify } = require("util");

const transcribeAsync = promisify(transcribe);

async function main() {
  // Step 1: Create a persistent context (model loaded once)
  const ctx = new WhisperContext({
    model: path.join(__dirname, "../../models/ggml-base.en.bin"),
    use_gpu: true,
    flash_attn: false,
  });

  console.log(`System info: ${ctx.getSystemInfo()}`);
  console.log(`Is multilingual: ${ctx.isMultilingual()}`);

  // Step 2: Transcribe multiple files using the same context
  const files = ["audio1.wav", "audio2.wav", "audio3.wav"];

  for (const file of files) {
    const result = await transcribeAsync(ctx, {
      fname_inp: file,
      language: "en",
      progress_callback: (p) => console.log(`Progress: ${p}%`)
    });

    for (const [start, end, text] of result.segments) {
      console.log(`[${start} --> ${end}] ${text}`);
    }
  }

  // Step 3: Free the context when done
  ctx.free();
}

main();
```

### WhisperContext Constructor Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | *required* | Path to whisper model file |
| `use_gpu` | boolean | true | Enable GPU acceleration |
| `flash_attn` | boolean | false | Enable flash attention |
| `gpu_device` | number | 0 | GPU device index |
| `dtw` | string | - | DTW alignment preset ("tiny", "base", "small", "medium", "large.v1/v2/v3") |
| `no_prints` | boolean | false | Suppress log output |

### WhisperContext Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `free()` | void | Release the context and print timings |
| `isMultilingual()` | boolean | Check if model supports multiple languages |
| `getSystemInfo()` | string | Get system/build information |

### transcribe() Options

All options from the whisper-server are supported:

#### Audio Input (one required)

| Parameter | Type | Description |
|-----------|------|-------------|
| `fname_inp` | string | Path to input audio file |
| `pcmf32` | Float32Array | Raw PCM audio buffer (16kHz mono) |

#### Threading

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_threads` | number | 4 | Number of threads for inference |
| `n_processors` | number | 1 | Number of parallel processors |

#### Transcription Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | string | "en" | Language code or "auto" |
| `translate` | boolean | false | Translate to English |
| `detect_language` | boolean | false | Auto-detect language |
| `no_timestamps` | boolean | false | Disable timestamps |
| `single_segment` | boolean | false | Force single segment output |
| `no_context` | boolean | true | Disable context from previous segment |

#### Audio Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offset_ms` | number | 0 | Audio offset in milliseconds |
| `duration_ms` | number | 0 | Audio duration to process (0 = all) |
| `audio_ctx` | number | 0 | Audio context size |

#### Segment Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_len` | number | 0 | Max segment length in characters |
| `max_tokens` | number | 0 | Max tokens per segment |
| `max_context` | number | -1 | Max context tokens (-1 = default) |
| `split_on_word` | boolean | false | Split on word boundaries |

#### Sampling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | number | 0.0 | Sampling temperature |
| `temperature_inc` | number | 0.2 | Temperature increment on fallback |
| `best_of` | number | 2 | Best of N samples (greedy) |
| `beam_size` | number | -1 | Beam size (-1 = greedy) |
| `no_fallback` | boolean | false | Disable temperature fallback |

#### Quality Thresholds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entropy_thold` | number | 2.4 | Entropy threshold for fallback |
| `logprob_thold` | number | -1.0 | Log probability threshold |
| `no_speech_thold` | number | 0.6 | No speech probability threshold |
| `word_thold` | number | 0.01 | Word timestamp threshold |

#### Output Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comma_in_time` | boolean | true | Use comma in timestamps |
| `token_timestamps` | boolean | false | Output token-level timestamps |
| `print_special` | boolean | false | Print special tokens |
| `print_progress` | boolean | false | Print progress to console |
| `print_realtime` | boolean | false | Print results in realtime |
| `print_timestamps` | boolean | true | Print timestamps |

#### Suppression

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `suppress_blank` | boolean | true | Suppress blank outputs |
| `suppress_nst` | boolean | false | Suppress non-speech tokens |

#### Other

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "" | Initial prompt for decoder |
| `tinydiarize` | boolean | false | Enable tinydiarize |
| `diarize` | boolean | false | Enable speaker diarization |
| `progress_callback` | function | - | Progress callback `(progress: number) => void` |

#### VAD Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vad` | boolean | false | Enable VAD |
| `vad_model` | string | - | Path to VAD model file |
| `vad_threshold` | number | 0.5 | Speech detection threshold (0.0-1.0) |
| `vad_min_speech_duration_ms` | number | 250 | Min speech duration in ms |
| `vad_min_silence_duration_ms` | number | 100 | Min silence duration in ms |
| `vad_max_speech_duration_s` | number | FLT_MAX | Max speech duration in seconds |
| `vad_speech_pad_ms` | number | 30 | Speech padding in ms |
| `vad_samples_overlap` | number | 0.1 | Sample overlap (0.0-1.0) |

### Result Format

```javascript
{
  segments: [
    ["00:00:00,000", "00:00:02,500", " Hello world."],
    ["00:00:02,500", "00:00:05,000", " This is a test."]
  ],
  language: "en"  // Only present if detect_language is true
}
```

## Legacy API (Backwards Compatible)

The original `whisper()` function is still available for backwards compatibility. It loads and frees the model on each call.

```shell
cd examples/addon.node
node index.js --language='en' --model='../../models/ggml-base.en.bin' --fname_inp='../../samples/jfk.wav'
```

```javascript
const path = require("path");
const { whisper } = require(path.join(__dirname, "../../build/Release/addon.node"));
const { promisify } = require("util");

const whisperAsync = promisify(whisper);

const result = await whisperAsync({
  language: "en",
  model: path.join(__dirname, "../../models/ggml-base.en.bin"),
  fname_inp: path.join(__dirname, "../../samples/jfk.wav"),
  use_gpu: true,
  progress_callback: (progress) => console.log(`Progress: ${progress}%`)
});

console.log(result);
```

## Examples

### Run Persistent Context Example

```shell
node persistent-context-example.js
```

### Run VAD Example

```shell
node vad-example.js
```

## Voice Activity Detection (VAD)

VAD can significantly improve transcription performance by only processing speech segments. This is especially beneficial for audio files with long periods of silence.

### VAD Model Setup

Before using VAD, download a VAD model:

```shell
# From the whisper.cpp root directory
./models/download-vad-model.sh silero-v6.2.0
```

### VAD Example

```javascript
const result = await transcribeAsync(ctx, {
  fname_inp: "audio.wav",
  language: "en",

  // Enable VAD
  vad: true,
  vad_model: path.join(__dirname, "../../models/ggml-silero-v6.2.0.bin"),
  vad_threshold: 0.5,
  vad_min_speech_duration_ms: 250,
  vad_min_silence_duration_ms: 100,

  progress_callback: (p) => console.log(`Progress: ${p}%`)
});
```

## CoreML Support

If whisper.cpp was compiled with CoreML support (`-DWHISPER_COREML=1`), the addon will automatically use CoreML acceleration. Ensure you have the `.mlmodelc` file alongside your model.

## Thread Safety

The `WhisperContext` class is thread-safe. Multiple transcription calls can be made concurrently, but they will be serialized internally using a mutex.
