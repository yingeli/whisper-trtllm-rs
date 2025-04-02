#pragma once

#include "whisper-trtllm-rs/src/sys/features.h"

#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/runtime/torchView.h"
#include "tensorrt_llm/runtime/torch.h"

#include <memory>
#include <filesystem>

#include "rust/cxx.h"

namespace tle = tensorrt_llm::executor;
namespace tlr = tensorrt_llm::runtime;

const int MAX_NEW_TOKENS = 224;

struct Config;

struct TranscribeOptions;

struct TranscribeResult;

struct TranscribeContext {
    std::size_t sample_begin;
    //torch::Half prevTimestampLogprob;
};

class TranscribeLogitsProcessor {
    public:
        void register_request(
            const tle::IdType req_id, 
            const std::size_t sample_begin
        );

        void unregister_request(
            const tle::IdType req_id
        );

        void process(
            tle::IdType req_id,
            tle::Tensor& logits, 
            tle::BeamTokens const& tokens,
            tle::StreamPtr const& stream_ptr,
            const bool stop_on_timestamps
        );

    private:
        std::mutex mutex_;
        std::unordered_map<tle::IdType, TranscribeContext> context_map_;
};

class Whisper {
    public:
        Whisper(
            std::filesystem::path const& model_path,
            Config const& config
        );

        tle::IdType enqueue_detect_language_request(
            const torch::Tensor& features
        );

        inline tle::IdType enqueue_detect_language_request(
            const Features& features
        ) {
            return enqueue_detect_language_request(features.tensor());
        };

        uint32_t await_detect_language_response(
            tle::IdType const &request_id
        );

        tle::IdType enqueue_transcribe_request(
            const torch::Tensor& features,
            const tle::VecTokens prompt,
            const TranscribeOptions &options,
            const bool stop_on_timestamps = false
        );

        tle::IdType enqueue_transcribe_request(
            const Features& features,
            const rust::Slice<const std::uint32_t> prompt,
            const TranscribeOptions &options,
            const bool stop_on_timestamps = false
        ) {

            return enqueue_transcribe_request(
                features.tensor(),
                tle::VecTokens(prompt.begin(), prompt.end()),
                options,
                stop_on_timestamps
            );
        }

        TranscribeResult await_transcribe_response(
            tle::IdType const &request_id
        );

        bool is_response_ready(
            tle::IdType const &request_id
        ) const;

    private:
        tle::Executor executor_;
        TranscribeLogitsProcessor transcribe_logits_processor_;
};

inline bool init() {
    return initTrtLlmPlugins();
}

std::unique_ptr<Whisper> whisper(const rust::Str model_path, const Config& config);