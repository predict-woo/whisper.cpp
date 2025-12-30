/**
 * Options for creating a WhisperContext
 */
export interface WhisperContextOptions {
  /** Path to the GGML model file */
  model: string;
  /** Enable GPU acceleration (default: true) */
  use_gpu?: boolean;
  /** Enable Flash Attention (default: false) */
  flash_attn?: boolean;
  /** GPU device index (default: 0) */
  gpu_device?: number;
  /** Enable Core ML acceleration on macOS (default: false) */
  use_coreml?: boolean;
  /** DTW alignment preset for word-level timestamps (e.g., 'base.en', 'small', 'large.v3') */
  dtw?: string;
  /** Suppress whisper.cpp log output (default: false) */
  no_prints?: boolean;
}

/**
 * Options for transcription
 */
export interface TranscribeOptions {
  /** Path to the audio file */
  fname_inp: string;
  /** Language code (e.g., 'en', 'zh', 'auto') */
  language?: string;
  /** Translate to English */
  translate?: boolean;
  /** Number of threads to use */
  n_threads?: number;
  /** Number of processors */
  n_processors?: number;
  /** Disable timestamps in output */
  no_timestamps?: boolean;
  /** Detect language automatically */
  detect_language?: boolean;
  /** Single segment mode */
  single_segment?: boolean;
  /** Maximum segment length (0 = no limit) */
  max_len?: number;
  /** Maximum tokens per segment (0 = no limit) */
  max_tokens?: number;
  /** Maximum context size (-1 = default) */
  max_context?: number;
  /** Temperature for sampling */
  temperature?: number;
  /** Temperature increment for fallback */
  temperature_inc?: number;
  /** Best of N sampling */
  best_of?: number;
  /** Beam size (-1 = greedy) */
  beam_size?: number;
  /** Entropy threshold */
  entropy_thold?: number;
  /** Log probability threshold */
  logprob_thold?: number;
  /** No speech threshold */
  no_speech_thold?: number;
  /** Initial prompt text */
  prompt?: string;
}

/**
 * Transcription result segment (tuple format)
 * [0]: Start time in format "HH:MM:SS,mmm"
 * [1]: End time in format "HH:MM:SS,mmm"
 * [2]: Transcribed text
 */
export type TranscriptSegment = [start: string, end: string, text: string];

/**
 * Transcription result
 */
export interface TranscribeResult {
  /** Array of transcript segments as [start, end, text] tuples */
  segments: TranscriptSegment[];
}

/**
 * Options for creating a VadContext
 */
export interface VadContextOptions {
  /** Path to the Silero VAD model file */
  model: string;
  /** Speech detection threshold (default: 0.5) */
  threshold?: number;
  /** Number of threads (default: 1) */
  n_threads?: number;
  /** Suppress model loading prints */
  no_prints?: boolean;
}

/**
 * WhisperContext class for persistent model context
 */
export interface WhisperContext {
  /** Get whisper.cpp system info string */
  getSystemInfo(): string;
  /** Check if model is multilingual */
  isMultilingual(): boolean;
  /** Free the context and release resources */
  free(): void;
}

/**
 * WhisperContext constructor type
 */
export interface WhisperContextConstructor {
  new (options: WhisperContextOptions): WhisperContext;
}

/**
 * VadContext class for voice activity detection
 */
export interface VadContext {
  /** Get the required window size in samples */
  getWindowSamples(): number;
  /** Get the expected sample rate (16000 Hz) */
  getSampleRate(): number;
  /** Process audio samples and return speech probability [0, 1] */
  process(samples: Float32Array): number;
  /** Reset the internal LSTM state */
  reset(): void;
  /** Free the context and release resources */
  free(): void;
}

/**
 * VadContext constructor type
 */
export interface VadContextConstructor {
  new (options: VadContextOptions): VadContext;
}

/**
 * Transcribe callback function signature
 */
export type TranscribeCallback = (
  error: Error | null,
  result?: TranscribeResult
) => void;

/**
 * Native addon interface
 */
export interface WhisperAddon {
  WhisperContext: WhisperContextConstructor;
  VadContext: VadContextConstructor;
  transcribe: (
    context: WhisperContext,
    options: TranscribeOptions,
    callback: TranscribeCallback
  ) => void;
  whisper: Record<string, unknown>;
}
