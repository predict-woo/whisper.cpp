const path = require("path");
const addon = require(path.join(__dirname, "packages/win32-ia32/whisper.node"));

console.log("=== 32-bit whisper.node test ===\n");
console.log("Exports:", Object.keys(addon));

// Create context with tiny.en model
const modelPath = path.join(__dirname, "../models/ggml-tiny.en.bin");
const audioPath = path.join(__dirname, "../samples/jfk.wav");

console.log("\nLoading model:", modelPath);
const start = performance.now();

const ctx = new addon.WhisperContext({
  model: modelPath,
  use_gpu: false,
  no_prints: false,
});

const loadTime = ((performance.now() - start) / 1000).toFixed(2);
console.log(`Context created in ${loadTime}s`);
console.log("System info:", ctx.getSystemInfo());

// Transcribe
console.log("\nTranscribing:", audioPath);
const transcribeStart = performance.now();

addon.transcribe(
  ctx,
  {
    fname_inp: audioPath,
    language: "en",
    n_threads: 4,
  },
  (err, result) => {
    if (err) {
      console.error("Transcription error:", err);
      process.exit(1);
    }

    const transcribeTime = ((performance.now() - transcribeStart) / 1000).toFixed(2);
    console.log(`\nTranscription completed in ${transcribeTime}s`);
    console.log(`Segments: ${result.segments.length}\n`);

    for (const seg of result.segments) {
      console.log("Raw segment:", JSON.stringify(seg));
      // Try both tuple and object formats
      if (Array.isArray(seg)) {
        console.log(`[${seg[0]} --> ${seg[1]}]${seg[2]}`);
      } else {
        console.log(`[${seg.start} --> ${seg.end}]${seg.text}`);
      }
    }

    ctx.free();
    console.log("\n=== Test passed! ===");
  }
);
