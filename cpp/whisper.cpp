#include "whisper.h"
#include "token.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/runtime/torchView.h"
#include "tensorrt_llm/runtime/torch.h"

#include <torch/torch.h>
#include <mutex>

namespace tlr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;

namespace tensorrt_llm::whisper {
    const torch::Half NEG_INF = static_cast<torch::Half>(-std::numeric_limits<float>::infinity());

    torch::Tensor toTorch(
        tle::Tensor& logits
    ) {
        auto const tensorOptions = torch::device(tlr::TorchUtils::device(logits.getData()))
        .pinned_memory(logits.getMemoryType() == tle::MemoryType::kCPU_PINNEDPOOL)
        //.dtype(tlr::TorchUtils::dataType(logits.getDataType()))
        .dtype(torch::kFloat16)
        .layout(torch::kStrided);

        return torch::from_blob(logits.getData(), {1, 1, logits.getShape()[2]}, tensorOptions);
    }

    void processDetectionLogits(
        tle::IdType reqId,
        tle::Tensor& logits, 
        tle::BeamTokens const& tokens,
        tle::StreamPtr const& streamPtr
    ) {
        auto torchLogits = toTorch(logits);

        for (auto beam = 0; beam < torchLogits.size(1); beam++) {
            auto beam_logits = torchLogits.index({0, beam});
            auto beam_tokens = tokens[beam];

            // Set everything before BEGIN_OF_LANGUAGE to NEG_INF
            beam_logits.slice(0, 0, token::BEGIN_OF_LANGUAGE).fill_(NEG_INF);
        
            // Set everything after END_OF_LANGUAGE to NEG_INF 
            beam_logits.slice(0, token::END_OF_LANGUAGE).fill_(NEG_INF);
        };
    }

    tle::ExecutorConfig executorConfig(const Config config, 
        TranscribeLogitsProcessor& transcribeLogitsProcessor
    ) {
        tle::KvCacheConfig kvCacheConfig;
        kvCacheConfig.setFreeGpuMemoryFraction(0.9);
        kvCacheConfig.setCrossKvCacheFraction(0.5);

        auto transcribeLogitsProcessorFn = [&transcribeLogitsProcessor](
            tle::IdType reqId, 
            tle::Tensor& logits, 
            tle::BeamTokens const& tokens,
            tle::StreamPtr const& streamPtr, 
            std::optional<tle::IdType> clientId)
        {
            transcribeLogitsProcessor.process(reqId, logits, tokens, streamPtr);
        };

        auto detectLogitsProcessorFn = [](
            tle::IdType reqId, 
            tle::Tensor& logits, 
            tle::BeamTokens const& tokens,
            tle::StreamPtr const& streamPtr, 
            std::optional<tle::IdType> clientId)
        {
            processDetectionLogits(reqId, logits, tokens, streamPtr);
        };

        auto logitsProcConfig = tle::LogitsPostProcessorConfig();
        auto logitsProcMap = std::unordered_map<std::string, tensorrt_llm::executor::LogitsPostProcessor>{
            {"transcribe", transcribeLogitsProcessorFn},
            {"detect", detectLogitsProcessorFn}
        };
        logitsProcConfig.setProcessorMap(logitsProcMap);

        tle::ExecutorConfig executorConfig = tle::ExecutorConfig(1);
        executorConfig.setBatchingType(config.batchingType);
        executorConfig.setKvCacheConfig(kvCacheConfig);
        executorConfig.setEnableChunkedContext(false);
        executorConfig.setLogitsPostProcessorConfig(logitsProcConfig);
        
        return executorConfig;
    }

    Whisper::Whisper(
        std::filesystem::path const& modelPath, 
        const Config config
    ) : mMel(modelPath / "mel_filters.npz"),
        mTranscribeLogitsProcessor(),
        mExecutor(
            modelPath / "encoder",
            modelPath / "decoder",
            tle::ModelType::kENCODER_DECODER,
            executorConfig(config, mTranscribeLogitsProcessor)) {
    }

    IdType Whisper::enqueueDetectLanguageRequest(
        const std::span<const float> audio
    ) {
        auto chunk = audio.size() > CHUNK_SIZE * SAMPLING_RATE ? audio.first(CHUNK_SIZE * SAMPLING_RATE) : audio;
        
        auto mel = mMel.extract(chunk).toType(torch::kFloat16);
        
        int padding = 3000 - mel.size(1);
        if (padding > 0) {
            mel = torch::nn::functional::pad(
                mel, 
                torch::nn::functional::PadFuncOptions({0, padding}).mode(torch::kConstant).value(0));
        }
        
        mel = mel.transpose(0, 1).contiguous();

        int encoderOutputLength = mel.size(0) / 2;

        // Create the request
        auto request = tle::Request({50258}, 1);
        request.setEncoderInputFeatures(tle::detail::ofITensor(tlr::TorchView::of(mel)));
        request.setEncoderOutputLength(encoderOutputLength);
        request.setEndId(50257);
        request.setPadId(50257);
        request.setLogitsPostProcessorName("detect");

        // tle::OutputConfig outputConfig;
        // outputConfig.returnLogProbs = true;
        // request.setOutputConfig(outputConfig);

        return mExecutor.enqueueRequest(request);
    }

    TokenIdType Whisper::awaitDetectLanguageResponse(
        IdType const &requestId        
    ) {
        auto response = mExecutor.awaitResponses(requestId)[0];
        auto result = response.getResult();
        return result.outputTokenIds[0].back();
    }

    IdType Whisper::enqueueTranscribeRequest(
        const std::span<const float> audio,
        const tle::VecTokens prompt
    ) {
        auto chunk = audio.size() > CHUNK_SIZE * SAMPLING_RATE ? audio.first(CHUNK_SIZE * SAMPLING_RATE) : audio;
        
        auto mel = mMel.extract(chunk).toType(torch::kFloat16);
        
        int padding = 3000 - mel.size(1);
        if (padding > 0) {
            mel = torch::nn::functional::pad(
                mel, 
                torch::nn::functional::PadFuncOptions({0, padding}).mode(torch::kConstant).value(0));
        }
        
        mel = mel.transpose(0, 1).contiguous();

        int encoderOutputLength = mel.size(0) / 2;

        // Create the request
        auto request = tle::Request(prompt, MAX_NEW_TOKENS);
        request.setEncoderInputFeatures(tle::detail::ofITensor(tlr::TorchView::of(mel)));
        request.setEncoderOutputLength(encoderOutputLength);
        request.setEndId(token::END_OF_TEXT);
        request.setPadId(token::END_OF_TEXT);
        request.setLogitsPostProcessorName("transcribe");

        return mExecutor.enqueueRequest(request);
    }

    TranscribeResult Whisper::awaitTranscribeResponse(
        IdType const &requestId        
    ) {
        auto response = mExecutor.awaitResponses(requestId)[0];
        auto result = response.getResult();

        return TranscribeResult {
            result.outputTokenIds[0]
        };
    }

    bool Whisper::isResponseReady(
        IdType const &requestId
    ) const {
        return mExecutor.getNumResponsesReady(requestId) > 0;
    }

    void TranscribeLogitsProcessor::process(
        tle::IdType reqId,
        tle::Tensor& logits, 
        tle::BeamTokens const& tokens,
        tle::StreamPtr const& streamPtr
    ) {
        auto const tensorOptions = torch::device(tlr::TorchUtils::device(logits.getData()))
            .pinned_memory(logits.getMemoryType() == tle::MemoryType::kCPU_PINNEDPOOL)
            //.dtype(tlr::TorchUtils::dataType(logits.getDataType()))
            .dtype(torch::kFloat16)
            .layout(torch::kStrided);
        auto shape = logits.getShape();
        auto torch_logits = torch::from_blob(logits.getData(), {shape[0], shape[1], shape[2]}, tensorOptions);

        std::cout << "torch_logits: " << torch_logits.sizes() << std::endl;

        auto neg_inf = -std::numeric_limits<float>::infinity();

        for (auto beam = 0; beam < torch_logits.size(1); beam++) {
            auto beam_logits = torch_logits.index({0, beam});
            auto beam_tokens = tokens[beam];

            // std::cout << "beam_logits: " << beam_logits.sizes() << std::endl;

            // suppress notimestamps
            beam_logits.index_put_({50364}, neg_inf);

            auto n_tokens = beam_tokens.size();
            bool last_was_timestamp = beam_tokens[n_tokens - 1] >= 50365 && beam_tokens[n_tokens - 1] <= 51865;
            bool penultimate_was_timestamp = beam_tokens[n_tokens - 2] >= 50365 && beam_tokens[n_tokens - 2] <= 51865;

            if (last_was_timestamp) {
                if (penultimate_was_timestamp) {
                    beam_logits.slice(0, 50365, 51866).fill_(neg_inf);
                } else {
                    beam_logits.slice(0, 0, 50257).fill_(neg_inf);
                }
            }
        }

        /*

        // suppress notimestamps
        torch_logits.index_put_({0, 0, 50364}, neg_inf);

        auto n_tokens = tokens[0].size();
        


        auto logprobs = torch::nn::functional::log_softmax(torch_logits, 2);

        auto text_logprobs = logprobs.slice(2, 0, 50365);
        auto timestampe_logprobs = logprobs.slice(2, 50365, 50866);

        auto max_text_logprob = std::get<0>(text_logprobs.max(2)).item<torch::Half>();
        auto timestampe_logprob = torch::logsumexp(timestampe_logprobs, 2).item<torch::Half>();

        if (timestampe_logprob > max_text_logprob) {
            //std::cout << "max_text_logprob: " << max_text_logprob 
            //    << " timestampe_logprob: " << timestampe_logprob 
            //    << std::endl;            
            torch_logits.slice(2, 0, 50365).fill_(-std::numeric_limits<float>::infinity());
            //std::cout << "tokens: " << tokens << std::endl;
        }

        mRequestContextMapMutex.lock();
        auto it = mRequestContextMap.find(reqId);
        if (it != mRequestContextMap.end()) {
            it->second.mPrevTimestampLogProb = 0.0;
            if (timestampe_logprob > it->second.mPrevTimestampLogProb + 1.8) {
                //torch_logits.slice(2, 0, 50365).fill_(-std::numeric_limits<float>::infinity());
                //text_logprobs.fill_(static_cast<torch::Half>(-std::numeric_limits<float>::infinity()));
                //std::cout << "max_text_logprob: " << max_text_logprob 
                //    << " timestampe_logprob: " << timestampe_logprob 
                //    << " prevTimestampLogProb: " << *prevTimestampLogProb 
                //    << std::endl;
                std::cout << "tokens: " << tokens << std::endl;
                //std::cout << std::endl;
                it->second.mPrevTimestampLogProb = timestampe_logprob;
            }            
        }
        mRequestContextMapMutex.unlock();
        */
    }

    /*
    void DetectLogitsProcessor::process(
        tle::IdType reqId,
        tle::Tensor& logits, 
        tle::BeamTokens const& tokens,
        tle::StreamPtr const& streamPtr
    ) {
        auto const tensorOptions = torch::device(tlr::TorchUtils::device(logits.getData()))
            .pinned_memory(logits.getMemoryType() == tle::MemoryType::kCPU_PINNEDPOOL)
            //.dtype(tlr::TorchUtils::dataType(logits.getDataType()))
            .dtype(torch::kFloat16)
            .layout(torch::kStrided);

        auto torch_logits = torch::from_blob(logits.getData(), {1, 1, logits.getShape()[2]}, tensorOptions);

        auto lang_logits = torch_logits.slice(2, 50259, 50359);

        auto language_token_probs = torch::softmax(lang_logits, 2);
        auto deteted_lang = std::get<1>(language_token_probs.max(2)).item<TokenIdType>();

        std::cout << "deteted_lang: " << deteted_lang << std::endl;

        mDetectionMapMutex.lock();
        //auto it = mDetectionMap.find(reqId);
        //if (it != mDetectionMap.end()) {
        //    it->second.language = deteted_lang;   
        //}
        mDetectionMap[reqId] = Detection{deteted_lang};
        mDetectionMapMutex.unlock();
    }

    TokenIdType DetectLogitsProcessor::getResult(
        IdType const &requestId
    ) {
        auto language = 0;
        mDetectionMapMutex.lock();
        auto it = mDetectionMap.find(requestId);
        if (it != mDetectionMap.end()) {
            language = it->second.language;
            mDetectionMap.erase(it);
        }
        mDetectionMapMutex.unlock();
        return language;
    }
    */
}