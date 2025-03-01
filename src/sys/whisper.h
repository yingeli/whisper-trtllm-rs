#pragma once

#include "whisper-trtllm-rs/cpp/whisper.h"

#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include <memory>
#include <filesystem>

#include "rust/cxx.h"

namespace tle = tensorrt_llm::executor;
namespace tlw = tensorrt_llm::whisper;

using tle::BatchingType;
using tle::Tensor;
using tle::VecTokens;
using tle::IdType;
using tle::TokenIdType;
using tlw::Config;
using tlw::TranscribeOptions;

struct TranscribeResult;

class Whisper {
    public:
        Whisper(
            std::filesystem::path const& model_path,
            Config config
        );

        IdType enqueue_detect_language_request(
            const rust::Slice<const float> first,
            const rust::Slice<const float> second
        );

        uint32_t await_detect_language_response(
            IdType const &request_id
        );

        IdType enqueue_transcribe_request(
            const rust::Slice<const float> first,
            const rust::Slice<const float> second,
            const rust::Slice<const std::uint32_t> prompt,
            const TranscribeOptions &options
        );

        TranscribeResult await_transcribe_response(
            IdType const &request_id
        );

        bool is_response_ready(
            IdType const &request_id
        ) const;

    private:
        tlw::Whisper _inner;
};

inline std::unique_ptr<Whisper> whisper(const rust::Str model_path_str, const Config config) {
    auto model_path = std::filesystem::path(static_cast<std::string>(model_path_str));
    return std::make_unique<Whisper>(
        model_path,
        config
    );
}

inline bool init() {
    return initTrtLlmPlugins();
}