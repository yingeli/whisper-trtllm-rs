#include "whisper-trtllm-rs/src/sys/whisper.rs.h"
#include "whisper-trtllm-rs/src/sys/vocab.h"
#include "whisper-trtllm-rs/src/sys/logits.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <span>
#include <mutex>

#include "rust/cxx.h"

namespace tlr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;

tle::ExecutorConfig executor_config(
    const Config config,
    TranscribeLogitsProcessor& transcribe_logits_processor
) {
    tle::ExecutorConfig executor_config = tle::ExecutorConfig(config.max_beam_width);
    executor_config.setBatchingType(tle::BatchingType::kINFLIGHT);

    tle::KvCacheConfig kv_cache_config;
    kv_cache_config.setFreeGpuMemoryFraction(0.9);
    kv_cache_config.setCrossKvCacheFraction(0.5);
    executor_config.setKvCacheConfig(kv_cache_config);

    auto process_transcribe_logits = [&transcribe_logits_processor](
        tle::IdType req_id, 
        tle::Tensor& logits, 
        tle::BeamTokens const& tokens,
        tle::StreamPtr const& stream_ptr, 
        std::optional<tle::IdType> client_id)
    {
        transcribe_logits_processor.process(req_id, logits, tokens, stream_ptr, false);
    };

    auto process_transcribe_segment_logits = [&transcribe_logits_processor](
        tle::IdType req_id, 
        tle::Tensor& logits, 
        tle::BeamTokens const& tokens,
        tle::StreamPtr const& stream_ptr, 
        std::optional<tle::IdType> client_id)
    {
        transcribe_logits_processor.process(req_id, logits, tokens, stream_ptr, true);
    };

    auto process_detect_logits = [](
        tle::IdType req_id, 
        tle::Tensor& logits,
        tle::BeamTokens const& tokens,
        tle::StreamPtr const& stream_ptr, 
        std::optional<tle::IdType> client_id)
    {
        at::cuda::CUDAStreamGuard guard(tlr::TorchUtils::stream(*stream_ptr));
        Logits lgts(logits);
        lgts.suppress_non_languages();
    };

    tle::LogitsPostProcessorConfig logits_proc_config;
    auto logits_proc_map = std::unordered_map<std::string, tensorrt_llm::executor::LogitsPostProcessor>{
        {"transcribe", process_transcribe_logits},
        {"transcribe_segment", process_transcribe_segment_logits},
        {"detect", process_detect_logits}
    };
    logits_proc_config.setProcessorMap(logits_proc_map);
    executor_config.setLogitsPostProcessorConfig(logits_proc_config);

    //auto decodingMode = DecodingMode::Auto();
    //decodingMode.useTemperature(true);
    //auto decodingConfig = DecodingConfig(decodingMode);
    //decodingConfig.setDecodingMode(decodingMode);
    //executorConfig.setDecodingConfig(decodingConfig);

    //executorConfig.setEnableChunkedContext(false);
    
    return executor_config;
}

Whisper::Whisper(
    const std::filesystem::path& model_path, 
    const Config& config
) : // mTranscribeLogitsProcessor(),
    executor_(
        model_path / "encoder",
        model_path / "decoder",
        tle::ModelType::kENCODER_DECODER,
        executor_config(config, transcribe_logits_processor_)
) {
}

tle::IdType Whisper::enqueue_detect_language_request(
    const torch::Tensor& features
) {
    auto mel = features.contiguous();
    auto encoder_output_length = mel.size(0) / 2;

    // Create the request
    auto request = tle::Request({token::START_OF_TRANSCRIPT}, 1);
    request.setEncoderInputFeatures(tle::detail::ofITensor(tlr::TorchView::of(mel)));
    request.setEncoderOutputLength(encoder_output_length);
    request.setEndId(token::END_OF_TEXT);
    request.setPadId(token::END_OF_TEXT);
    request.setLogitsPostProcessorName("detect");

    return executor_.enqueueRequest(request);
}

uint32_t Whisper::await_detect_language_response(
    tle::IdType const &request_id
) {
    auto response = executor_.awaitResponses(request_id)[0];
    auto result = response.getResult();
    return result.outputTokenIds[0].back();
}

tle::IdType Whisper::enqueue_transcribe_request(
    const torch::Tensor& features,
    const tle::VecTokens prompt,
    const TranscribeOptions &options,
    const bool stop_on_timestamps
) {
    auto mel = features.contiguous();

    int encoder_output_length = mel.size(0) / 2;

    // Create the request
    auto request = tle::Request(prompt, MAX_NEW_TOKENS);
    request.setEncoderInputFeatures(tle::detail::ofITensor(tlr::TorchView::of(mel)));
    request.setEncoderOutputLength(encoder_output_length);
    request.setEndId(token::END_OF_TEXT);
    request.setPadId(token::END_OF_TEXT);

    if (stop_on_timestamps) {
        request.setLogitsPostProcessorName("transcribe_segment");
    } else {
        request.setLogitsPostProcessorName("transcribe");
    }

    //auto sampling_config = tle::SamplingConfig(options.beam_width, options.top_k);
    //if (options.top_p > 0) {
    //    sampling_config.setTopP(options.top_p);
    //}
    //sampling_config.setTemperature(options.temperature);
    //samplingConfig.setEarlyStopping(std::nullopt);
    //sampling_config.setEarlyStopping(0);
    //sampling_config.setLengthPenalty(1);
    //samplingConfig.setNoRepeatNgramSize(0);
    //samplingConfig.setBeamSearchDiversityRate(0.2);
    //sampling_config.setNumReturnSequences(1);
    //request.setSamplingConfig(sampling_config);

    tle::OutputConfig output_config;
    output_config.returnLogProbs = true;
    request.setOutputConfig(output_config);

    auto request_id = executor_.enqueueRequest(request);
    // transcribe_logits_processor_.register_request(request_id, prompt.size());

    return request_id;
}

TranscribeResult Whisper::await_transcribe_response(
    tle::IdType const &request_id        
) {
    auto response = executor_.awaitResponses(request_id)[0];
    auto result = response.getResult();

    if (result.isFinal) {
        transcribe_logits_processor_.unregister_request(request_id);
    }

    rust::Vec<uint32_t> tokens;
    tokens.reserve(result.outputTokenIds[0].size());
    for (const auto& token : result.outputTokenIds[0]) {
        tokens.push_back(static_cast<uint32_t>(token));
    }

    auto avg_logprob = result.cumLogProbs.value()[0] / static_cast<float>(result.logProbs.value()[0].size() + 1);

    return TranscribeResult {
        .is_final = result.isFinal,
        .is_sequence_final = result.isSequenceFinal,
        .tokens = tokens,
        .avg_logprob = avg_logprob
    };
}

bool Whisper::is_response_ready(
    tle::IdType const &request_id
) const {
    return executor_.getNumResponsesReady(request_id) > 0;
}

void TranscribeLogitsProcessor::register_request(
    const tle::IdType req_id, 
    const std::size_t sample_begin
) {
    std::lock_guard<std::mutex> lock(mutex_); 
    context_map_.emplace(req_id, TranscribeContext{sample_begin});
}

void TranscribeLogitsProcessor::unregister_request(
    const tle::IdType req_id
) {
    std::lock_guard<std::mutex> lock(mutex_); 
    context_map_.erase(req_id);
}

void TranscribeLogitsProcessor::process(
    tle::IdType req_id,
    tle::Tensor& tle_logits, 
    tle::BeamTokens const& tokens,
    tle::StreamPtr const& stream_ptr,
    const bool stop_on_timestamps
) {
    at::cuda::CUDAStreamGuard guard(tlr::TorchUtils::stream(*stream_ptr));

    Logits logits(tle_logits);

    // mutex_.lock();
    // auto sample_begin = context_map_[req_id].sample_begin;
    // mutex_.unlock();

    // suppress notimestamps
    // logits.suppress_notimestamps();

    // suppress blank at the beginning
    // if (tokens[0].size() == sample_begin) {
    //    logits.suppress_blank();
    //    logits.suppress_non_timestamps();
    //    return;
    // }

    if (tokens[0].back() == token::START_OF_TRANSCRIPT) {
        logits.suppress_non_languages();
        return;
    }

    if (tokens[0].size() > 1 && tokens[0][tokens[0].size() - 2] == token::START_OF_TRANSCRIPT) {
        logits.set_transcribe();
        return;
    }

    auto is_first_text = tokens[0][tokens[0].size() - 2] == token::TRANSCRIBE;

    // suppress notimestamps
    logits.suppress_notimestamps();
    
    bool check_timestamps_prob = false;

    for (auto b = 0; b < tokens.size(); b++) {
        auto beam_logits = logits.beam(b);
        auto beam_tokens = tokens[b];

        auto n_tokens = beam_tokens.size();
        bool last_was_timestamp = token::is_timestamp(beam_tokens[n_tokens - 1]);
        bool penultimate_was_timestamp = is_first_text || token::is_timestamp(beam_tokens[n_tokens - 2]);        
        //bool last_was_timestamp = n_tokens > sample_begin &&
        //    token::is_timestamp(beam_tokens[n_tokens - 1]);
        //bool penultimate_was_timestamp = n_tokens < sample_begin + 2 ||
        //    n_tokens > sample_begin + 1 && token::is_timestamp(beam_tokens[n_tokens - 2]);

        if (last_was_timestamp) {
            if (penultimate_was_timestamp) {
                beam_logits.suppress_timestamps();
            } else {
                if (stop_on_timestamps) {
                    beam_logits.set_eot();
                    return;
                }
                beam_logits.suppress_text();
                beam_logits.suppress_timestamps(beam_tokens[n_tokens - 1]);
                check_timestamps_prob = true;
                // beam_logits.suppress_non_eot();
            }
        } else {
            // for (auto i = n_tokens - 1; i >= sample_begin; i--) {
            for (auto i = n_tokens - 1; beam_tokens[i] != token::TRANSCRIBE; i--) {
                auto token = beam_tokens[i];
                if (token::is_timestamp(token)) {
                    beam_logits.suppress_timestamps(token + 1);
                    break;
                }
            }
            check_timestamps_prob = true;
        }
    }

    if (check_timestamps_prob) {
        auto logprobs = logits.logprobs();
        for (auto b = 0; b < tokens.size(); b++) {
            auto beam_logprobs = logprobs.beam(b);
            auto timestamps_logprob = beam_logprobs.timestamps().logsumexp();

            auto max_text_logprob = beam_logprobs.non_timestamps().max();

            if (timestamps_logprob > max_text_logprob) {
                logits.beam(b).suppress_non_timestamps();
                //std::cout << tokens[b] << std::endl;
            }
        }
    }
}

std::unique_ptr<Whisper> whisper(const rust::Str model_path, const Config& config) {
    auto path = std::filesystem::path(static_cast<std::string>(model_path));
    return std::make_unique<Whisper>(
        path,
        config
    );
}