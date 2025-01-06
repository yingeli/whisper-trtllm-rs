#include "whisper.h"
#include "tensor.h"
#include "tensorrt_llm/executor/executor.h"
#include <span>

//namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

Whisper::Whisper(
    std::filesystem::path const& modelPath, 
    tle::ExecutorConfig executorConfig
) : mMel(modelPath / "mel_filters.npz"),
    mExecutor(
        modelPath / "encoder",
        modelPath / "decoder",
        tle::ModelType::kENCODER_DECODER,
        executorConfig) {
}

TranscribeResult Whisper::transcribe(
    std::span<float> audio,
    tle::VecTokens prompt
) {
    if (audio.size() > CHUNK_SIZE * SAMPLING_RATE) {
        throw std::runtime_error("Audio is too long");
    }
    auto padding = (CHUNK_SIZE * SAMPLING_RATE) - audio.size();
    auto melTorch = TorchTensor(mMel.extract(audio, padding).contiguous());
    auto mel = tle::Tensor(melTorch);

    // Create the request
    auto request = tle::Request(prompt, MAX_NEW_TOKENS);
    request.setEncoderInputFeatures(mel);

    auto requestId = mExecutor.enqueueRequest(request);
    auto response = mExecutor.awaitResponses(requestId)[0];
    auto result = response.getResult();
    auto output = result.outputTokenIds[0];
    auto transcribeResult = TranscribeResult{output};
    
    return transcribeResult;
}