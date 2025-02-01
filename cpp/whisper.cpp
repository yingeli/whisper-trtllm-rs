#include "whisper.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/runtime/torchView.h"

//#include <span>

namespace tlr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;

namespace tensorrt_llm::whisper {

    tle::ExecutorConfig executorConfig(const Config config) {
        tle::ExecutorConfig executorConfig = tle::ExecutorConfig(1);
        executorConfig.setBatchingType(config.batchingType);
        return executorConfig;
    }

    Whisper::Whisper(
        std::filesystem::path const& modelPath, 
        const Config config
    ) : mMel(modelPath / "mel_filters.npz"),
        mExecutor(
            modelPath / "encoder",
            modelPath / "decoder",
            tle::ModelType::kENCODER_DECODER,
            executorConfig(config)) {
    }

    TranscribeResult Whisper::transcribe(
        //std::span<float> audio,
        std::vector<float> audio,
        tle::VecTokens prompt
    ) {
        if (audio.size() > CHUNK_SIZE * SAMPLING_RATE) {
            throw std::runtime_error("Audio is too long");
        }
        auto padding = (CHUNK_SIZE * SAMPLING_RATE) - audio.size();
        auto mel = mMel.extract(audio, padding).contiguous();

        // Create the request
        auto request = tle::Request(prompt, MAX_NEW_TOKENS);
        request.setEncoderInputFeatures(tle::detail::ofITensor(tlr::TorchView::of(mel)));

        auto requestId = mExecutor.enqueueRequest(request);
        auto response = mExecutor.awaitResponses(requestId)[0];
        auto result = response.getResult();
        auto output = result.outputTokenIds[0];
        auto transcribeResult = TranscribeResult{output};
        
        return transcribeResult;
    }

}