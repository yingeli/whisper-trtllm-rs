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

IdType Whisper::enqueue_transcribe_request(
    const rust::Slice<const float> audio,
    const rust::Slice<const std::uint32_t> prompt
) {
    auto audio_span = std::span(audio.data(), audio.size());

    auto prompt_vec = tle::VecTokens(prompt.begin(), prompt.end());

    return _inner.enqueueTranscribeRequest(audio_span, prompt_vec);
}


TranscribeResult Whisper::await_transcribe_response(
    IdType const &request_id
) {
    auto result = _inner.awaitTranscribeResponse(request_id);

    rust::Vec<std::uint32_t> tokens;
    for (const auto& item : result.tokens) {
        tokens.push_back(item);
    }
    
    return TranscribeResult {
        tokens,
    };
}

bool Whisper::is_response_ready(
    IdType const &request_id
) const {
    return _inner.isResponseReady(request_id);
}