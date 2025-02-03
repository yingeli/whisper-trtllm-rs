#include "whisper.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/runtime/torchView.h"
#include "tensorrt_llm/runtime/torch.h"

#include <torch/torch.h>

#include <cuda_fp16.h>

namespace tlr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;

namespace tensorrt_llm::whisper {

    tle::ExecutorConfig executorConfig(const Config config) {
        tle::KvCacheConfig kvCacheConfig;
        kvCacheConfig.setFreeGpuMemoryFraction(0.9);
        kvCacheConfig.setCrossKvCacheFraction(0.5);

        std::string logitsPostProcessorName = "MyLogitsPP";
        auto logitsPostProcessorFn = [](
            tle::IdType reqId, 
            tle::Tensor& logits, 
            tle::BeamTokens const& tokens,
            tle::StreamPtr const& streamPtr, 
            std::optional<tle::IdType> clientId)
        {
            auto const tensorOptions = torch::device(tlr::TorchUtils::device(logits.getData()))
                .pinned_memory(logits.getMemoryType() == tle::MemoryType::kCPU_PINNEDPOOL)
                //.dtype(tlr::TorchUtils::dataType(logits.getDataType()))
                .dtype(torch::kFloat16)
                .layout(torch::kStrided);
            //auto shape = tlr::TorchUtils::shape(logits.getShape());
            //std::vector<tle::SizeType32> shape = {1, 1, logits.getShape()[2]};
            //auto torch_logits = torch::for_blob(logits.getData(), {1, 1, logits.getShape()[2]})
            //    .options(tensorOptions)
            //    .make_tensor();
            auto torch_logits = torch::from_blob(logits.getData(), {1, 1, logits.getShape()[2]}, tensorOptions);

            auto logprobs = torch::nn::functional::log_softmax(torch_logits, 2);
            // auto logprobs = torch::nn::functional::softmax(torch_logits, 2);

            auto text_logprobs = logprobs.slice(2, 0, 50365);
            auto timestampe_logprobs = logprobs.slice(2, 50365, 50866);

            auto max_text_logprob = std::get<0>(text_logprobs.max(2)).item<torch::Half>();
            auto timestampe_logprob = torch::logsumexp(timestampe_logprobs, 2).item<torch::Half>();
            //auto timestampe_logprob = timestampe_logprobs.sum(2).item<torch::Half>();

            //if (timestampe_logprob > -0.5) {

            std::cout << "max_text_logprob: " << max_text_logprob << " timestampe_logprob: " << timestampe_logprob << std::endl;
            std::cout << "tokens: " << tokens << std::endl;
            std::cout << std::endl;
        };
        auto logitsProcConfig = tle::LogitsPostProcessorConfig();
        auto logitsProcMap = std::unordered_map<std::string, tensorrt_llm::executor::LogitsPostProcessor>{
            {logitsPostProcessorName, logitsPostProcessorFn}
        };
        logitsProcConfig.setProcessorMap(logitsProcMap);

        tle::ExecutorConfig executorConfig = tle::ExecutorConfig(1);
        executorConfig.setBatchingType(config.batchingType);
        executorConfig.setKvCacheConfig(kvCacheConfig);
        executorConfig.setEnableChunkedContext(false);
        executorConfig.setLogitsPostProcessorConfig(logitsProcConfig);

        std::cout << "registered MyLogitsPP" << std::endl;
        
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

    IdType Whisper::enqueueTranscribeRequest(
        const std::span<const float> audio,
        const tle::VecTokens prompt
    ) {
        if (audio.size() > CHUNK_SIZE * SAMPLING_RATE) {
            throw std::runtime_error("Audio is too long");
        }
        // auto padding = (CHUNK_SIZE * SAMPLING_RATE) - audio.size();
        auto mel = mMel.extract(audio).toType(torch::kFloat16);
        
        int padding = 3000 - mel.size(1);
        if (padding > 0) {
            mel = torch::nn::functional::pad(
                mel, 
                torch::nn::functional::PadFuncOptions({0, padding}).mode(torch::kConstant).value(0));
        }
        
        // IFB
        mel = mel.transpose(0, 1).contiguous();

        // Static
        // mel = mel.unsqueeze(0).contiguous();

        int encoderOutputLength = mel.size(0) / 2;

        // Create the request
        auto request = tle::Request(prompt, MAX_NEW_TOKENS);
        request.setEncoderInputFeatures(tle::detail::ofITensor(tlr::TorchView::of(mel)));
        request.setEncoderOutputLength(encoderOutputLength);
        request.setEndId(50257);
        request.setPadId(50257);
        std::string logitsPostProcessorName = "MyLogitsPP";
        request.setLogitsPostProcessorName(logitsPostProcessorName);

        return mExecutor.enqueueRequest(request);
    }

    TranscribeResult Whisper::awaitTranscribeResponse(
        IdType const &requestId        
    ) {
        auto response = mExecutor.awaitResponses(requestId)[0];
        auto result = response.getResult();
        auto output = result.outputTokenIds[0];
        auto transcribeResult = TranscribeResult{output};
        return transcribeResult;
    }

    bool Whisper::isResponseReady(
        IdType const &requestId
    ) const {
        return mExecutor.getNumResponsesReady(requestId) > 0;
    }
}