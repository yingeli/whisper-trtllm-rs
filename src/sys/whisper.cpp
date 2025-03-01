//#include "whisper.h"
#include "whisper-trtllm-rs/src/sys/whisper.rs.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"

#include <span>

#include "rust/cxx.h"

namespace tlr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;

Whisper::Whisper(
    std::filesystem::path const& model_path,
    const Config config
): _inner(model_path, config) {}

IdType Whisper::enqueue_detect_language_request(
    const rust::Slice<const float> first,
    const rust::Slice<const float> second
) {
    auto first_span = std::span(first.data(), first.size());
    auto second_span = std::span(second.data(), second.size());

    return _inner.enqueueDetectLanguageRequest(first_span, second_span);
}

uint32_t Whisper::await_detect_language_response(
    IdType const &request_id
) {
    return _inner.awaitDetectLanguageResponse(request_id);
}

IdType Whisper::enqueue_transcribe_request(
    const rust::Slice<const float> first,
    const rust::Slice<const float> second,
    const rust::Slice<const std::uint32_t> prompt,
    const TranscribeOptions &options
) {
    auto first_span = std::span(first.data(), first.size());
    auto second_span = std::span(second.data(), second.size());

    auto prompt_vec = tle::VecTokens(prompt.begin(), prompt.end());

    return _inner.enqueueTranscribeRequest(first_span, second_span, prompt_vec, options);
}

TranscribeResult Whisper::await_transcribe_response(
    IdType const &request_id
) {
    auto result = _inner.awaitTranscribeResponse(request_id);
    
    rust::Vec<uint32_t> tokens;
    tokens.reserve(result.tokens.size());
    for (const auto& token : result.tokens) {
        tokens.push_back(static_cast<uint32_t>(token));
    }
    
    return TranscribeResult {
        .tokens = tokens,
        .avgLogProb = result.avgLogProb
    };
}

bool Whisper::is_response_ready(
    IdType const &request_id
) const {
    return _inner.isResponseReady(request_id);
}