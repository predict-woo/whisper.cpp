/**
 * Test OpenVINO encoder acceleration
 * 
 * Prerequisites:
 * 1. Build with OpenVINO support: npx cmake-js compile --CDADDON_OPENVINO=ON
 * 2. Install dependencies: pip install openai-whisper openvino openvino-dev torch
 * 3. Convert whisper encoder to OpenVINO format:
 *    python models/convert-whisper-to-openvino.py --model large-v3-turbo
 * 4. Download whisper model: ./models/download-ggml-model.cmd large-v3-turbo-q5_0
 * 
 * Usage:
 *   node test-openvino.js [model_path] [audio_path] [openvino_model_path] [device]
 * 
 * Examples:
 *   node test-openvino.js                                    # Default (CPU)
 *   node test-openvino.js - - - GPU                          # Use GPU (Intel iGPU)
 *   node test-openvino.js - - - NPU                          # Use NPU (Intel AI Boost)
 *   node test-openvino.js ../../models/ggml-large-v3-turbo-q5_0.bin ../../samples/jfk.wav - GPU
 * 
 * Note: The OpenVINO encoder model is converted from the original (non-quantized) Whisper model.
 *       It works with any quantized variant of the same model family.
 */

const path = require('path');
const fs = require('fs');

// Load the addon
const addon = require('./build/Release/addon.node.node');

// Default paths
const defaultModelPath = path.join(__dirname, '../../models/ggml-large-v3-turbo-q5_0.bin');
const defaultAudioPath = path.join(__dirname, '../../samples/jfk.wav');
// OpenVINO encoder model (converted from original large-v3-turbo, works with any quantized variant)
const defaultOpenVinoModelPath = path.join(__dirname, '../../models/ggml-large-v3-turbo-encoder-openvino.xml');

// Parse arguments (use '-' or empty to use defaults)
const arg = (idx, defaultVal) => {
    const val = process.argv[idx];
    return (val && val !== '-') ? val : defaultVal;
};
const modelPath = arg(2, defaultModelPath);
const audioPath = arg(3, defaultAudioPath);
const openvinoModelPath = arg(4, defaultOpenVinoModelPath);
const openvinoDevice = arg(5, 'CPU');  // 'CPU', 'GPU', or 'NPU'

// Check files exist
if (!fs.existsSync(modelPath)) {
    console.error(`Error: Model not found at ${modelPath}`);
    console.error('Download with: ./models/download-ggml-model.sh base.en');
    process.exit(1);
}

if (!fs.existsSync(audioPath)) {
    console.error(`Error: Audio file not found at ${audioPath}`);
    process.exit(1);
}

// Check for OpenVINO model files
const openvinoXml = openvinoModelPath;
const openvinoBin = openvinoModelPath.replace('.xml', '.bin');

const hasOpenVinoModel = fs.existsSync(openvinoXml) && fs.existsSync(openvinoBin);

console.log('='.repeat(60));
console.log('OpenVINO Transcription Test');
console.log('='.repeat(60));
console.log(`Model: ${modelPath}`);
console.log(`Audio: ${audioPath}`);
console.log(`OpenVINO model found: ${hasOpenVinoModel ? 'Yes' : 'No'}`);
if (hasOpenVinoModel) {
    console.log(`  - ${openvinoXml}`);
    console.log(`  - ${openvinoBin}`);
}
console.log('');

/**
 * Run transcription and measure time
 */
async function transcribe(ctx, options, label) {
    return new Promise((resolve, reject) => {
        const startTime = performance.now();
        
        addon.transcribe(ctx, options, (err, result) => {
            const endTime = performance.now();
            const elapsed = ((endTime - startTime) / 1000).toFixed(2);
            
            if (err) {
                reject(err);
                return;
            }
            
            console.log(`\n[${label}] Completed in ${elapsed}s`);
            console.log('-'.repeat(40));
            
            for (const [start, end, text] of result.segments) {
                console.log(`[${start} --> ${end}]${text}`);
            }
            
            resolve({ result, elapsed: parseFloat(elapsed) });
        });
    });
}

async function main() {
    // Test 1: Standard GPU/CPU transcription (baseline)
    console.log('\n>>> Test 1: Standard transcription (baseline)');
    console.log('Loading model...');
    
    const ctxStandard = new addon.WhisperContext({
        model: modelPath,
        use_gpu: true,
        no_prints: true
    });
    
    console.log(`System info: ${ctxStandard.getSystemInfo()}`);
    console.log(`Multilingual: ${ctxStandard.isMultilingual()}`);
    
    const baseline = await transcribe(ctxStandard, {
        fname_inp: audioPath,
        language: 'en',
        n_threads: 4
    }, 'Standard');
    
    ctxStandard.free();
    
    // Test 2: OpenVINO transcription (if available)
    if (hasOpenVinoModel) {
        console.log('\n>>> Test 2: OpenVINO transcription');
        console.log('Loading model with OpenVINO encoder...');
        
        try {
            console.log(`Using OpenVINO device: ${openvinoDevice}`);
            const ctxOpenvino = new addon.WhisperContext({
                model: modelPath,
                use_gpu: false,  // OpenVINO handles acceleration
                use_openvino: true,
                openvino_model_path: openvinoXml,  // Explicit path to OpenVINO encoder
                openvino_device: openvinoDevice,  // 'CPU', 'GPU', or 'NPU'
                openvino_cache_dir: path.join(__dirname, 'openvino_cache'),  // Cache compiled model
                no_prints: false  // Show OpenVINO init logs
            });
            
            const openvino = await transcribe(ctxOpenvino, {
                fname_inp: audioPath,
                language: 'en',
                n_threads: 4
            }, 'OpenVINO');
            
            ctxOpenvino.free();
            
            // Compare results
            console.log('\n' + '='.repeat(60));
            console.log('Performance Comparison');
            console.log('='.repeat(60));
            console.log(`Standard:  ${baseline.elapsed}s`);
            console.log(`OpenVINO:  ${openvino.elapsed}s`);
            
            const speedup = (baseline.elapsed / openvino.elapsed).toFixed(2);
            if (openvino.elapsed < baseline.elapsed) {
                console.log(`Speedup:   ${speedup}x faster with OpenVINO`);
            } else {
                console.log(`Note:      OpenVINO was ${(1/speedup).toFixed(2)}x slower (first run may be slow due to model compilation)`);
            }
            
        } catch (err) {
            console.error('\nOpenVINO test failed:', err.message);
            console.log('\nPossible causes:');
            console.log('  1. OpenVINO support not compiled in (rebuild with -DADDON_OPENVINO=ON)');
            console.log('  2. OpenVINO runtime not installed');
            console.log('  3. OpenVINO model files missing or corrupted');
        }
    } else {
        console.log('\n>>> Skipping OpenVINO test (model not found)');
        console.log('\nTo enable OpenVINO:');
        console.log('  1. Install OpenVINO: pip install openvino openvino-dev');
        console.log('  2. Convert model: python models/convert-whisper-to-openvino.py --model base.en');
        console.log('  3. Rebuild addon: npx cmake-js compile --CDADDON_OPENVINO=ON');
    }
    
    console.log('\nDone!');
}

main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
