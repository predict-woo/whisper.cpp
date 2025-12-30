#include "napi.h"
#include "common.h"
#include "common-whisper.h"

#include "whisper.h"

#include <string>
#include <thread>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cfloat>
#include <mutex>

// ============================================================================
// Utility functions
// ============================================================================

void cb_log_disable(enum ggml_log_level, const char *, void *) {}

// Helper to safely get optional boolean from JS object
static bool get_bool(const Napi::Object& obj, const char* key, bool default_val) {
    if (obj.Has(key) && obj.Get(key).IsBoolean()) {
        return obj.Get(key).As<Napi::Boolean>();
    }
    return default_val;
}

// Helper to safely get optional number from JS object
static int32_t get_int32(const Napi::Object& obj, const char* key, int32_t default_val) {
    if (obj.Has(key) && obj.Get(key).IsNumber()) {
        return obj.Get(key).As<Napi::Number>().Int32Value();
    }
    return default_val;
}

static float get_float(const Napi::Object& obj, const char* key, float default_val) {
    if (obj.Has(key) && obj.Get(key).IsNumber()) {
        return obj.Get(key).As<Napi::Number>().FloatValue();
    }
    return default_val;
}

static std::string get_string(const Napi::Object& obj, const char* key, const std::string& default_val) {
    if (obj.Has(key) && obj.Get(key).IsString()) {
        return obj.Get(key).As<Napi::String>();
    }
    return default_val;
}

// ============================================================================
// WhisperContext - Persistent context wrapper using ObjectWrap
// ============================================================================

class WhisperContext : public Napi::ObjectWrap<WhisperContext> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    WhisperContext(const Napi::CallbackInfo& info);
    ~WhisperContext();

    // Check if context is valid
    bool IsValid() const { return ctx_ != nullptr; }
    whisper_context* GetContext() { return ctx_; }
    std::mutex& GetMutex() { return mutex_; }

private:
    static Napi::FunctionReference constructor;

    Napi::Value Free(const Napi::CallbackInfo& info);
    Napi::Value IsMultilingual(const Napi::CallbackInfo& info);
    Napi::Value GetSystemInfo(const Napi::CallbackInfo& info);

    whisper_context* ctx_ = nullptr;
    std::mutex mutex_;
};

Napi::FunctionReference WhisperContext::constructor;

Napi::Object WhisperContext::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "WhisperContext", {
        InstanceMethod("free", &WhisperContext::Free),
        InstanceMethod("isMultilingual", &WhisperContext::IsMultilingual),
        InstanceMethod("getSystemInfo", &WhisperContext::GetSystemInfo),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("WhisperContext", func);
    return exports;
}

WhisperContext::WhisperContext(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<WhisperContext>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "Expected options object").ThrowAsJavaScriptException();
        return;
    }

    Napi::Object options = info[0].As<Napi::Object>();

    // Required: model path
    if (!options.Has("model") || !options.Get("model").IsString()) {
        Napi::TypeError::New(env, "model path is required").ThrowAsJavaScriptException();
        return;
    }
    std::string model_path = options.Get("model").As<Napi::String>();

    // Context parameters (set once at load time)
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = get_bool(options, "use_gpu", true);
    cparams.flash_attn = get_bool(options, "flash_attn", false);
    cparams.gpu_device = get_int32(options, "gpu_device", 0);

    // DTW parameters
    if (options.Has("dtw") && options.Get("dtw").IsString()) {
        std::string dtw = options.Get("dtw").As<Napi::String>();
        cparams.dtw_token_timestamps = true;
        cparams.dtw_aheads_preset = WHISPER_AHEADS_NONE;

        if (dtw == "tiny")              cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY;
        else if (dtw == "tiny.en")      cparams.dtw_aheads_preset = WHISPER_AHEADS_TINY_EN;
        else if (dtw == "base")         cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE;
        else if (dtw == "base.en")      cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
        else if (dtw == "small")        cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL;
        else if (dtw == "small.en")     cparams.dtw_aheads_preset = WHISPER_AHEADS_SMALL_EN;
        else if (dtw == "medium")       cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;
        else if (dtw == "medium.en")    cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM_EN;
        else if (dtw == "large.v1")     cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V1;
        else if (dtw == "large.v2")     cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V2;
        else if (dtw == "large.v3")     cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3;
        else if (dtw == "large.v3.turbo") cparams.dtw_aheads_preset = WHISPER_AHEADS_LARGE_V3_TURBO;
    }

    // Suppress logging if requested
    if (get_bool(options, "no_prints", false)) {
        whisper_log_set(cb_log_disable, NULL);
    }

    // Load the model
    ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);

    if (ctx_ == nullptr) {
        Napi::Error::New(env, "Failed to initialize whisper context from model: " + model_path)
            .ThrowAsJavaScriptException();
        return;
    }
}

WhisperContext::~WhisperContext() {
    if (ctx_ != nullptr) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
}

Napi::Value WhisperContext::Free(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ctx_ != nullptr) {
        whisper_print_timings(ctx_);
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
    return info.Env().Undefined();
}

Napi::Value WhisperContext::IsMultilingual(const Napi::CallbackInfo& info) {
    if (ctx_ == nullptr) {
        return Napi::Boolean::New(info.Env(), false);
    }
    return Napi::Boolean::New(info.Env(), whisper_is_multilingual(ctx_));
}

Napi::Value WhisperContext::GetSystemInfo(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), whisper_print_system_info());
}

// ============================================================================
// TranscribeWorker - Async worker for transcription
// ============================================================================

struct transcribe_params {
    // Threading
    int32_t n_threads    = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors = 1;

    // Audio processing
    int32_t offset_ms   = 0;
    int32_t duration_ms = 0;

    // Context and length
    int32_t max_context = -1;
    int32_t max_len     = 0;
    int32_t max_tokens  = 0;
    int32_t audio_ctx   = 0;

    // Sampling
    int32_t best_of   = 2;
    int32_t beam_size = -1;

    // Thresholds
    float word_thold      = 0.01f;
    float entropy_thold   = 2.40f;
    float logprob_thold   = -1.00f;
    float temperature     = 0.00f;
    float temperature_inc = 0.20f;
    float no_speech_thold = 0.6f;

    // Flags
    bool translate       = false;
    bool no_context      = true;
    bool no_timestamps   = false;
    bool single_segment  = false;
    bool print_special   = false;
    bool print_progress  = false;
    bool print_realtime  = false;
    bool print_timestamps = true;
    bool token_timestamps = false;
    bool split_on_word   = false;
    bool detect_language = false;
    bool suppress_blank  = true;
    bool suppress_nst    = false;
    bool tinydiarize     = false;
    bool no_fallback     = false;
    bool comma_in_time   = true;
    bool diarize         = false;

    // Language and prompt
    std::string language = "en";
    std::string prompt   = "";

    // Audio input
    std::string fname_inp = "";
    std::vector<float> pcmf32;

    // VAD parameters
    bool        vad                         = false;
    std::string vad_model                   = "";
    float       vad_threshold               = 0.5f;
    int         vad_min_speech_duration_ms  = 250;
    int         vad_min_silence_duration_ms = 100;
    float       vad_max_speech_duration_s   = FLT_MAX;
    int         vad_speech_pad_ms           = 30;
    float       vad_samples_overlap         = 0.1f;
};

struct transcribe_result {
    std::vector<std::vector<std::string>> segments;  // [start, end, text]
    std::string language;
    bool success = false;
    std::string error;
};

class TranscribeWorker : public Napi::AsyncWorker {
public:
    TranscribeWorker(
        Napi::Function& callback,
        WhisperContext* wrapper,
        transcribe_params params,
        Napi::Function progress_callback,
        Napi::Env env
    ) : Napi::AsyncWorker(callback),
        wrapper_(wrapper),
        params_(std::move(params)),
        env_(env) {

        // Create thread-safe function for progress callback
        if (!progress_callback.IsEmpty() && progress_callback.IsFunction()) {
            tsfn_ = Napi::ThreadSafeFunction::New(
                env,
                progress_callback,
                "Progress Callback",
                0,
                1
            );
        }
    }

    ~TranscribeWorker() {
        if (tsfn_) {
            tsfn_.Release();
        }
    }

    void Execute() override {
        whisper_context* ctx = wrapper_->GetContext();
        if (ctx == nullptr) {
            result_.error = "Context is not initialized or has been freed";
            return;
        }

        // Lock the context for thread safety
        std::lock_guard<std::mutex> lock(wrapper_->GetMutex());

        // Get audio data
        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;

        if (!params_.pcmf32.empty()) {
            pcmf32 = params_.pcmf32;
        } else if (!params_.fname_inp.empty()) {
            if (!::read_audio_data(params_.fname_inp, pcmf32, pcmf32s, params_.diarize)) {
                result_.error = "Failed to read audio file: " + params_.fname_inp;
                return;
            }
        } else {
            result_.error = "No audio input provided (use pcmf32 or fname_inp)";
            return;
        }

        // Validate language
        if (params_.language != "auto" && whisper_lang_id(params_.language.c_str()) == -1) {
            result_.error = "Unknown language: " + params_.language;
            return;
        }

        // Handle non-multilingual models
        if (!whisper_is_multilingual(ctx)) {
            if (params_.language != "en" || params_.translate) {
                params_.language = "en";
                params_.translate = false;
            }
        }

        // Build whisper_full_params with all options
        whisper_full_params wparams = whisper_full_default_params(
            params_.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY
        );

        // Core parameters
        wparams.n_threads        = params_.n_threads;
        wparams.offset_ms        = params_.offset_ms;
        wparams.duration_ms      = params_.duration_ms;

        // Task parameters
        wparams.translate        = params_.translate;
        wparams.language         = params_.detect_language ? "auto" : params_.language.c_str();
        wparams.detect_language  = params_.detect_language;

        // Context parameters
        wparams.n_max_text_ctx   = params_.max_context >= 0 ? params_.max_context : wparams.n_max_text_ctx;
        wparams.no_context       = params_.no_context;
        wparams.single_segment   = params_.single_segment;

        // Output parameters
        wparams.print_special    = params_.print_special;
        wparams.print_progress   = params_.print_progress;
        wparams.print_realtime   = params_.print_realtime;
        wparams.print_timestamps = params_.print_timestamps;
        wparams.no_timestamps    = params_.no_timestamps;

        // Token-level timestamps
        wparams.token_timestamps = params_.token_timestamps || params_.max_len > 0;
        wparams.thold_pt         = params_.word_thold;
        wparams.max_len          = params_.max_len;
        wparams.max_tokens       = params_.max_tokens;
        wparams.split_on_word    = params_.split_on_word;

        // Audio context
        wparams.audio_ctx        = params_.audio_ctx;

        // Sampling parameters
        wparams.temperature      = params_.temperature;
        wparams.temperature_inc  = params_.no_fallback ? 0.0f : params_.temperature_inc;
        wparams.entropy_thold    = params_.entropy_thold;
        wparams.logprob_thold    = params_.logprob_thold;
        wparams.no_speech_thold  = params_.no_speech_thold;

        wparams.greedy.best_of        = params_.best_of;
        wparams.beam_search.beam_size = params_.beam_size;

        // Prompt
        wparams.initial_prompt   = params_.prompt.empty() ? nullptr : params_.prompt.c_str();

        // Suppression
        wparams.suppress_blank   = params_.suppress_blank;
        wparams.suppress_nst     = params_.suppress_nst;

        // Tinydiarize
        wparams.tdrz_enable      = params_.tinydiarize;

        // VAD parameters
        wparams.vad              = params_.vad;
        wparams.vad_model_path   = params_.vad_model.empty() ? nullptr : params_.vad_model.c_str();

        wparams.vad_params.threshold               = params_.vad_threshold;
        wparams.vad_params.min_speech_duration_ms  = params_.vad_min_speech_duration_ms;
        wparams.vad_params.min_silence_duration_ms = params_.vad_min_silence_duration_ms;
        wparams.vad_params.max_speech_duration_s   = params_.vad_max_speech_duration_s;
        wparams.vad_params.speech_pad_ms           = params_.vad_speech_pad_ms;
        wparams.vad_params.samples_overlap         = params_.vad_samples_overlap;

        // Progress callback
        wparams.progress_callback = [](struct whisper_context*, struct whisper_state*, int progress, void* user_data) {
            TranscribeWorker* worker = static_cast<TranscribeWorker*>(user_data);
            worker->OnProgress(progress);
        };
        wparams.progress_callback_user_data = this;

        // Run inference
        int ret = whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params_.n_processors);

        if (ret != 0) {
            result_.error = "Failed to process audio (whisper_full_parallel returned " + std::to_string(ret) + ")";
            return;
        }

        // Extract results
        if (params_.detect_language || params_.language == "auto") {
            result_.language = whisper_lang_str(whisper_full_lang_id(ctx));
        }

        const int n_segments = whisper_full_n_segments(ctx);
        result_.segments.resize(n_segments);

        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text(ctx, i);
            const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
            const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

            result_.segments[i].push_back(to_timestamp(t0, params_.comma_in_time));
            result_.segments[i].push_back(to_timestamp(t1, params_.comma_in_time));
            result_.segments[i].push_back(text);
        }

        result_.success = true;
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());

        if (!result_.success) {
            Callback().Call({Napi::Error::New(Env(), result_.error).Value()});
            return;
        }

        Napi::Object resultObj = Napi::Object::New(Env());

        // Add language if detected
        if (!result_.language.empty()) {
            resultObj.Set("language", Napi::String::New(Env(), result_.language));
        }

        // Add segments
        Napi::Array segments = Napi::Array::New(Env(), result_.segments.size());
        for (size_t i = 0; i < result_.segments.size(); ++i) {
            Napi::Array segment = Napi::Array::New(Env(), 3);
            segment.Set((uint32_t)0, Napi::String::New(Env(), result_.segments[i][0]));  // start
            segment.Set((uint32_t)1, Napi::String::New(Env(), result_.segments[i][1]));  // end
            segment.Set((uint32_t)2, Napi::String::New(Env(), result_.segments[i][2]));  // text
            segments.Set((uint32_t)i, segment);
        }
        resultObj.Set("segments", segments);

        Callback().Call({Env().Null(), resultObj});
    }

    void OnProgress(int progress) {
        if (tsfn_) {
            auto callback = [progress](Napi::Env env, Napi::Function jsCallback) {
                jsCallback.Call({Napi::Number::New(env, progress)});
            };
            tsfn_.BlockingCall(callback);
        }
    }

private:
    WhisperContext* wrapper_;
    transcribe_params params_;
    transcribe_result result_;
    Napi::Env env_;
    Napi::ThreadSafeFunction tsfn_;
};

// ============================================================================
// transcribe() - Async transcription function
// ============================================================================

Napi::Value Transcribe(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Validate arguments: (context, options, callback)
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected 3 arguments: context, options, callback")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (!info[0].IsObject()) {
        Napi::TypeError::New(env, "First argument must be a WhisperContext")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (!info[1].IsObject()) {
        Napi::TypeError::New(env, "Second argument must be an options object")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (!info[2].IsFunction()) {
        Napi::TypeError::New(env, "Third argument must be a callback function")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Get context wrapper
    WhisperContext* wrapper = Napi::ObjectWrap<WhisperContext>::Unwrap(info[0].As<Napi::Object>());
    if (!wrapper->IsValid()) {
        Napi::Error::New(env, "WhisperContext has been freed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object options = info[1].As<Napi::Object>();
    Napi::Function callback = info[2].As<Napi::Function>();

    // Parse all options into transcribe_params
    transcribe_params params;

    // Threading
    params.n_threads    = get_int32(options, "n_threads", params.n_threads);
    params.n_processors = get_int32(options, "n_processors", params.n_processors);

    // Audio processing
    params.offset_ms   = get_int32(options, "offset_ms", params.offset_ms);
    params.duration_ms = get_int32(options, "duration_ms", params.duration_ms);

    // Context and length
    params.max_context = get_int32(options, "max_context", params.max_context);
    params.max_len     = get_int32(options, "max_len", params.max_len);
    params.max_tokens  = get_int32(options, "max_tokens", params.max_tokens);
    params.audio_ctx   = get_int32(options, "audio_ctx", params.audio_ctx);

    // Sampling
    params.best_of   = get_int32(options, "best_of", params.best_of);
    params.beam_size = get_int32(options, "beam_size", params.beam_size);

    // Thresholds
    params.word_thold      = get_float(options, "word_thold", params.word_thold);
    params.entropy_thold   = get_float(options, "entropy_thold", params.entropy_thold);
    params.logprob_thold   = get_float(options, "logprob_thold", params.logprob_thold);
    params.temperature     = get_float(options, "temperature", params.temperature);
    params.temperature_inc = get_float(options, "temperature_inc", params.temperature_inc);
    params.no_speech_thold = get_float(options, "no_speech_thold", params.no_speech_thold);

    // Flags
    params.translate        = get_bool(options, "translate", params.translate);
    params.no_context       = get_bool(options, "no_context", params.no_context);
    params.no_timestamps    = get_bool(options, "no_timestamps", params.no_timestamps);
    params.single_segment   = get_bool(options, "single_segment", params.single_segment);
    params.print_special    = get_bool(options, "print_special", params.print_special);
    params.print_progress   = get_bool(options, "print_progress", params.print_progress);
    params.print_realtime   = get_bool(options, "print_realtime", params.print_realtime);
    params.print_timestamps = get_bool(options, "print_timestamps", params.print_timestamps);
    params.token_timestamps = get_bool(options, "token_timestamps", params.token_timestamps);
    params.split_on_word    = get_bool(options, "split_on_word", params.split_on_word);
    params.detect_language  = get_bool(options, "detect_language", params.detect_language);
    params.suppress_blank   = get_bool(options, "suppress_blank", params.suppress_blank);
    params.suppress_nst     = get_bool(options, "suppress_nst", params.suppress_nst);
    params.tinydiarize      = get_bool(options, "tinydiarize", params.tinydiarize);
    params.no_fallback      = get_bool(options, "no_fallback", params.no_fallback);
    params.comma_in_time    = get_bool(options, "comma_in_time", params.comma_in_time);
    params.diarize          = get_bool(options, "diarize", params.diarize);

    // Language and prompt
    params.language = get_string(options, "language", params.language);
    params.prompt   = get_string(options, "prompt", params.prompt);

    // Audio input - file path
    params.fname_inp = get_string(options, "fname_inp", "");

    // Audio input - PCM buffer
    if (options.Has("pcmf32") && options.Get("pcmf32").IsTypedArray()) {
        Napi::Float32Array pcmf32 = options.Get("pcmf32").As<Napi::Float32Array>();
        size_t length = pcmf32.ElementLength();
        params.pcmf32.reserve(length);
        for (size_t i = 0; i < length; i++) {
            params.pcmf32.push_back(pcmf32[i]);
        }
    }

    // VAD parameters
    params.vad                         = get_bool(options, "vad", params.vad);
    params.vad_model                   = get_string(options, "vad_model", params.vad_model);
    params.vad_threshold               = get_float(options, "vad_threshold", params.vad_threshold);
    params.vad_min_speech_duration_ms  = get_int32(options, "vad_min_speech_duration_ms", params.vad_min_speech_duration_ms);
    params.vad_min_silence_duration_ms = get_int32(options, "vad_min_silence_duration_ms", params.vad_min_silence_duration_ms);
    params.vad_max_speech_duration_s   = get_float(options, "vad_max_speech_duration_s", params.vad_max_speech_duration_s);
    params.vad_speech_pad_ms           = get_int32(options, "vad_speech_pad_ms", params.vad_speech_pad_ms);
    params.vad_samples_overlap         = get_float(options, "vad_samples_overlap", params.vad_samples_overlap);

    // Progress callback
    Napi::Function progress_callback;
    if (options.Has("progress_callback") && options.Get("progress_callback").IsFunction()) {
        progress_callback = options.Get("progress_callback").As<Napi::Function>();
    }

    // Create and queue the async worker
    TranscribeWorker* worker = new TranscribeWorker(
        callback, wrapper, std::move(params), progress_callback, env
    );
    worker->Queue();

    return env.Undefined();
}

// ============================================================================
// Legacy whisper() function - maintains backwards compatibility
// ============================================================================

struct legacy_whisper_params {
    int32_t n_threads    = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms  = 0;
    int32_t offset_n     = 0;
    int32_t duration_ms  = 0;
    int32_t max_context  = -1;
    int32_t max_len      = 0;
    int32_t best_of      = 5;
    int32_t beam_size    = -1;
    int32_t audio_ctx    = 0;

    float word_thold    = 0.01f;
    float entropy_thold = 2.4f;
    float logprob_thold = -1.0f;

    bool translate      = false;
    bool diarize        = false;
    bool print_special  = false;
    bool print_progress = false;
    bool no_timestamps  = false;
    bool no_prints      = false;
    bool detect_language= false;
    bool use_gpu        = true;
    bool flash_attn     = false;
    bool comma_in_time  = true;

    std::string language = "en";
    std::string prompt;
    std::string model    = "";

    std::vector<std::string> fname_inp = {};
    std::vector<float> pcmf32 = {};

    bool        vad           = false;
    std::string vad_model     = "";
    float       vad_threshold = 0.5f;
    int         vad_min_speech_duration_ms = 250;
    int         vad_min_silence_duration_ms = 100;
    float       vad_max_speech_duration_s = FLT_MAX;
    int         vad_speech_pad_ms = 30;
    float       vad_samples_overlap = 0.1f;
};

struct legacy_whisper_result {
    std::vector<std::vector<std::string>> segments;
    std::string language;
};

class LegacyProgressWorker : public Napi::AsyncWorker {
public:
    LegacyProgressWorker(Napi::Function& callback, legacy_whisper_params params, Napi::Function progress_callback, Napi::Env env)
        : Napi::AsyncWorker(callback), params_(params), env_(env) {
        if (!progress_callback.IsEmpty()) {
            tsfn_ = Napi::ThreadSafeFunction::New(env, progress_callback, "Progress Callback", 0, 1);
        }
    }

    ~LegacyProgressWorker() {
        if (tsfn_) {
            tsfn_.Release();
        }
    }

    void Execute() override {
        if (params_.no_prints) {
            whisper_log_set(cb_log_disable, NULL);
        }

        if (params_.fname_inp.empty() && params_.pcmf32.empty()) {
            SetError("no input files or audio buffer specified");
            return;
        }

        if (params_.language != "auto" && whisper_lang_id(params_.language.c_str()) == -1) {
            SetError("unknown language: " + params_.language);
            return;
        }

        struct whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = params_.use_gpu;
        cparams.flash_attn = params_.flash_attn;

        struct whisper_context* ctx = whisper_init_from_file_with_params(params_.model.c_str(), cparams);
        if (ctx == nullptr) {
            SetError("failed to initialize whisper context");
            return;
        }

        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;

        if (!params_.pcmf32.empty()) {
            pcmf32 = params_.pcmf32;
        } else if (!params_.fname_inp.empty()) {
            if (!::read_audio_data(params_.fname_inp[0], pcmf32, pcmf32s, params_.diarize)) {
                whisper_free(ctx);
                SetError("failed to read audio file");
                return;
            }
        }

        if (!whisper_is_multilingual(ctx)) {
            if (params_.language != "en" || params_.translate) {
                params_.language = "en";
                params_.translate = false;
            }
        }

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.strategy = params_.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

        wparams.print_realtime   = false;
        wparams.print_progress   = params_.print_progress;
        wparams.print_timestamps = !params_.no_timestamps;
        wparams.print_special    = params_.print_special;
        wparams.translate        = params_.translate;
        wparams.language         = params_.detect_language ? "auto" : params_.language.c_str();
        wparams.detect_language  = params_.detect_language;
        wparams.n_threads        = params_.n_threads;
        wparams.n_max_text_ctx   = params_.max_context >= 0 ? params_.max_context : wparams.n_max_text_ctx;
        wparams.offset_ms        = params_.offset_t_ms;
        wparams.duration_ms      = params_.duration_ms;
        wparams.thold_pt         = params_.word_thold;
        wparams.entropy_thold    = params_.entropy_thold;
        wparams.logprob_thold    = params_.logprob_thold;
        wparams.max_len          = params_.max_len;
        wparams.audio_ctx        = params_.audio_ctx;
        wparams.greedy.best_of        = params_.best_of;
        wparams.beam_search.beam_size = params_.beam_size;
        wparams.initial_prompt   = params_.prompt.c_str();
        wparams.no_timestamps    = params_.no_timestamps;

        // VAD parameters
        wparams.vad            = params_.vad;
        wparams.vad_model_path = params_.vad_model.c_str();
        wparams.vad_params.threshold               = params_.vad_threshold;
        wparams.vad_params.min_speech_duration_ms  = params_.vad_min_speech_duration_ms;
        wparams.vad_params.min_silence_duration_ms = params_.vad_min_silence_duration_ms;
        wparams.vad_params.max_speech_duration_s   = params_.vad_max_speech_duration_s;
        wparams.vad_params.speech_pad_ms           = params_.vad_speech_pad_ms;
        wparams.vad_params.samples_overlap         = params_.vad_samples_overlap;

        // Progress callback
        wparams.progress_callback = [](struct whisper_context*, struct whisper_state*, int progress, void* user_data) {
            LegacyProgressWorker* worker = static_cast<LegacyProgressWorker*>(user_data);
            worker->OnProgress(progress);
        };
        wparams.progress_callback_user_data = this;

        if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params_.n_processors) != 0) {
            whisper_free(ctx);
            SetError("failed to process audio");
            return;
        }

        if (params_.detect_language || params_.language == "auto") {
            result_.language = whisper_lang_str(whisper_full_lang_id(ctx));
        }

        const int n_segments = whisper_full_n_segments(ctx);
        result_.segments.resize(n_segments);

        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text(ctx, i);
            const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
            const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

            result_.segments[i].push_back(to_timestamp(t0, params_.comma_in_time));
            result_.segments[i].push_back(to_timestamp(t1, params_.comma_in_time));
            result_.segments[i].push_back(text);
        }

        whisper_print_timings(ctx);
        whisper_free(ctx);
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());

        Napi::Object resultObj = Napi::Object::New(Env());
        if (!result_.language.empty()) {
            resultObj.Set("language", Napi::String::New(Env(), result_.language));
        }

        Napi::Array transcriptionArray = Napi::Array::New(Env(), result_.segments.size());
        for (size_t i = 0; i < result_.segments.size(); ++i) {
            Napi::Array tmp = Napi::Array::New(Env(), 3);
            for (size_t j = 0; j < 3; ++j) {
                tmp.Set((uint32_t)j, Napi::String::New(Env(), result_.segments[i][j]));
            }
            transcriptionArray.Set((uint32_t)i, tmp);
        }
        resultObj.Set("transcription", transcriptionArray);
        Callback().Call({Env().Null(), resultObj});
    }

    void OnProgress(int progress) {
        if (tsfn_) {
            auto callback = [progress](Napi::Env env, Napi::Function jsCallback) {
                jsCallback.Call({Napi::Number::New(env, progress)});
            };
            tsfn_.BlockingCall(callback);
        }
    }

private:
    legacy_whisper_params params_;
    legacy_whisper_result result_;
    Napi::Env env_;
    Napi::ThreadSafeFunction tsfn_;
};

Napi::Value LegacyWhisper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() <= 0 || !info[0].IsObject()) {
        Napi::TypeError::New(env, "object expected").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    legacy_whisper_params params;
    Napi::Object options = info[0].As<Napi::Object>();

    params.language   = get_string(options, "language", params.language);
    params.model      = get_string(options, "model", params.model);

    if (options.Has("fname_inp") && options.Get("fname_inp").IsString()) {
        params.fname_inp.push_back(options.Get("fname_inp").As<Napi::String>());
    }

    params.use_gpu        = get_bool(options, "use_gpu", params.use_gpu);
    params.flash_attn     = get_bool(options, "flash_attn", params.flash_attn);
    params.no_prints      = get_bool(options, "no_prints", params.no_prints);
    params.no_timestamps  = get_bool(options, "no_timestamps", params.no_timestamps);
    params.detect_language = get_bool(options, "detect_language", params.detect_language);
    params.audio_ctx      = get_int32(options, "audio_ctx", params.audio_ctx);
    params.comma_in_time  = get_bool(options, "comma_in_time", params.comma_in_time);
    params.max_len        = get_int32(options, "max_len", params.max_len);
    params.max_context    = get_int32(options, "max_context", params.max_context);
    params.prompt         = get_string(options, "prompt", params.prompt);
    params.print_progress = get_bool(options, "print_progress", params.print_progress);
    params.translate      = get_bool(options, "translate", params.translate);
    params.diarize        = get_bool(options, "diarize", params.diarize);

    // VAD parameters
    params.vad                         = get_bool(options, "vad", params.vad);
    params.vad_model                   = get_string(options, "vad_model", params.vad_model);
    params.vad_threshold               = get_float(options, "vad_threshold", params.vad_threshold);
    params.vad_min_speech_duration_ms  = get_int32(options, "vad_min_speech_duration_ms", params.vad_min_speech_duration_ms);
    params.vad_min_silence_duration_ms = get_int32(options, "vad_min_silence_duration_ms", params.vad_min_silence_duration_ms);
    params.vad_max_speech_duration_s   = get_float(options, "vad_max_speech_duration_s", params.vad_max_speech_duration_s);
    params.vad_speech_pad_ms           = get_int32(options, "vad_speech_pad_ms", params.vad_speech_pad_ms);
    params.vad_samples_overlap         = get_float(options, "vad_samples_overlap", params.vad_samples_overlap);

    // PCM buffer
    if (options.Has("pcmf32") && options.Get("pcmf32").IsTypedArray()) {
        Napi::Float32Array pcmf32 = options.Get("pcmf32").As<Napi::Float32Array>();
        size_t length = pcmf32.ElementLength();
        params.pcmf32.reserve(length);
        for (size_t i = 0; i < length; i++) {
            params.pcmf32.push_back(pcmf32[i]);
        }
    }

    Napi::Function callback = info[1].As<Napi::Function>();

    Napi::Function progress_callback;
    if (options.Has("progress_callback") && options.Get("progress_callback").IsFunction()) {
        progress_callback = options.Get("progress_callback").As<Napi::Function>();
    }

    LegacyProgressWorker* worker = new LegacyProgressWorker(callback, params, progress_callback, env);
    worker->Queue();

    return env.Undefined();
}

// ============================================================================
// Module initialization
// ============================================================================

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Initialize WhisperContext class
    WhisperContext::Init(env, exports);

    // Export transcribe function
    exports.Set("transcribe", Napi::Function::New(env, Transcribe));

    // Export legacy whisper function for backwards compatibility
    exports.Set("whisper", Napi::Function::New(env, LegacyWhisper));

    return exports;
}

NODE_API_MODULE(whisper, Init)
