//#include "whisper.h"
#include "whisper-trtllm-rs/src/sys/whisper.rs.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
//#include "tensorrt_llm/runtime/torchView.h"

namespace tlr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;

Whisper::Whisper(
    std::filesystem::path const& model_path,
    const Config config
): _inner(model_path, config) {}

/*
TranscribeResult Whisper::transcribe(
    //std::span<float> audio,
    //std::vector<float> audio,
    Tensor features,
    VecTokens prompts
) {
    // Create the request
    auto request = tle::Request(prompt, MAX_NEW_TOKENS);
    request.setEncoderInputFeatures(features);

    auto requestId = mExecutor.enqueueRequest(request);
    auto response = mExecutor.awaitResponses(requestId)[0];
    auto result = response.getResult();
    auto output = result.outputTokenIds[0];
    auto transcribeResult = TranscribeResult{output};
    
    return transcribeResult;
}
    */