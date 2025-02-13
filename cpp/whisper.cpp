#include "whisper.h"
#include "logits.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/runtime/torchView.h"
#include "tensorrt_llm/runtime/torch.h"

#include <mutex>

namespace tlr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;

namespace tensorrt_llm::whisper {
    void processDetectionLogits(
        tle::IdType reqId,
        tle::Tensor& tleLogits, 
        tle::BeamTokens const& tokens,
        tle::StreamPtr const& streamPtr
    ) {
        Logits logits(tleLogits, streamPtr);
        logits.suppressNonLanguage();
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
        auto request = tle::Request({token::START_OF_TRANSCRIPT}, 1);
        request.setEncoderInputFeatures(tle::detail::ofITensor(tlr::TorchView::of(mel)));
        request.setEncoderOutputLength(encoderOutputLength);
        request.setEndId(token::END_OF_TEXT);
        request.setPadId(token::END_OF_TEXT);
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

        auto requestId = mExecutor.enqueueRequest(request);
        mTranscribeLogitsProcessor.addRequest(requestId, prompt.size());

        return requestId;
    }

    TranscribeResult Whisper::awaitTranscribeResponse(
        IdType const &requestId        
    ) {
        auto response = mExecutor.awaitResponses(requestId)[0];
        auto result = response.getResult();

        mTranscribeLogitsProcessor.removeRequest(requestId);

        return TranscribeResult {
            result.outputTokenIds[0]
        };
    }

    bool Whisper::isResponseReady(
        IdType const &requestId
    ) const {
        return mExecutor.getNumResponsesReady(requestId) > 0;
    }

    void TranscribeLogitsProcessor::addRequest(
        const IdType reqId, 
        const std::size_t sampleBegin
    ) {
        std::lock_guard<std::mutex> lock(mMutex); 
        mTranscribeContextMap.emplace(reqId, TranscribeContext{sampleBegin});
    }

    void TranscribeLogitsProcessor::removeRequest(
        const IdType reqId
    ) {
        std::lock_guard<std::mutex> lock(mMutex); 
        mTranscribeContextMap.erase(reqId);
    }

    void TranscribeLogitsProcessor::process(
        tle::IdType reqId,
        tle::Tensor& tleLogits, 
        tle::BeamTokens const& tokens,
        tle::StreamPtr const& streamPtr
    ) {
        Logits logits(tleLogits, streamPtr);

        mMutex.lock();
        auto sampleBegin = mTranscribeContextMap[reqId].mSampleBegin;
        mMutex.unlock();

        // suppress notimestamps
        logits.suppressNoTimestamps();

        for (auto b = 0; b < tokens.size(); b++) {
            auto beamLogits = logits.beam(b);
            auto beamTokens = tokens[b];

            // suppress blank at the beginning
            if (beamTokens.size() == sampleBegin) {
                beamLogits.suppressBlank();
            }

            auto nTokens = beamTokens.size();
            bool lastWasTimestamp = nTokens > sampleBegin &&
                token::isTimestamp(beamTokens[nTokens - 1]);
            bool penultimateWasTimestamp = nTokens < sampleBegin + 2 ||
                nTokens > sampleBegin + 1 && token::isTimestamp(beamTokens[nTokens - 2]);

            if (lastWasTimestamp) {
                if (penultimateWasTimestamp) {
                    std::cout << "beamLogits: " << beamLogits.sizes() << std::endl;
                    std::cout << " tokens: " << beamTokens 
                        << std::endl;
                    beamLogits.suppressTimestamps();
                } else {
                    beamLogits.suppressText();
                }
            }

            for (auto i = nTokens - 1; i >= sampleBegin; i--) {
                auto token = beamTokens[i];
                if (token::isTimestamp(token)) {
                    tle::TokenIdType timestampLast;
                    if (lastWasTimestamp && !penultimateWasTimestamp) {
                        timestampLast = token;
                    } else {
                        timestampLast = token + 1;
                    }
                    beamLogits.suppressTimestamps(timestampLast);
                    break;
                }
            }

            // suppress generating non-timestamp tokens at the beginning
            if (beamTokens.size() == sampleBegin) {
                beamLogits.suppressNonTimestamp();
            }
        }

        // TODO: suppress max_initial_timestamp_index

        // if sum of probability over timestamps is above any other token, sample timestamp
        auto logprobs = logits.logprobs();
        
        for (auto b = 0; b < tokens.size(); b++) {
            auto beamLogprobs = logprobs.beam(b);

            auto timestampLogprob = beamLogprobs.timestamps().logsumexp();
            auto maxTextLogprob = beamLogprobs.nonTimestamps().max();

            if (timestampLogprob > maxTextLogprob) {
                std::cout << "timestampLogprob: " << timestampLogprob << " maxTextLogprob: " << maxTextLogprob << std::endl;
                //std::cout << "tokens: " << tokens[b] << std::endl;
                auto beamLogits = logits.beam(b);
                beamLogits.suppressNonTimestamp();
            }
        }
    }
}

/*

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
