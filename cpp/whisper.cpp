#include "whisper.h"
#include "logits.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/runtime/torchView.h"
#include "tensorrt_llm/runtime/torch.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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
        at::cuda::CUDAStreamGuard guard(tlr::TorchUtils::stream(*streamPtr));
        
        Logits logits(tleLogits);
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
            transcribeLogitsProcessor.process(reqId, logits, tokens, streamPtr, false);
        };

        auto transcribeSegmentLogitsProcessorFn = [&transcribeLogitsProcessor](
            tle::IdType reqId, 
            tle::Tensor& logits, 
            tle::BeamTokens const& tokens,
            tle::StreamPtr const& streamPtr, 
            std::optional<tle::IdType> clientId)
        {
            transcribeLogitsProcessor.process(reqId, logits, tokens, streamPtr, true);
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
            {"transcribeSegment", transcribeSegmentLogitsProcessorFn},
            {"detect", detectLogitsProcessorFn}
        };
        logitsProcConfig.setProcessorMap(logitsProcMap);

        auto decodingMode = tle::DecodingMode::Auto();
        decodingMode.useTemperature(true);
        auto decodingConfig = tle::DecodingConfig(decodingMode);
        //decodingConfig.setDecodingMode(decodingMode);

        tle::ExecutorConfig executorConfig = tle::ExecutorConfig(5);
        executorConfig.setBatchingType(config.batchingType);
        executorConfig.setKvCacheConfig(kvCacheConfig);
        executorConfig.setLogitsPostProcessorConfig(logitsProcConfig);
        executorConfig.setDecodingConfig(decodingConfig);
        //executorConfig.setEnableChunkedContext(false);
        
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
        const std::span<const float> first,
        const std::optional<std::span<const float>> second
    ) {
        //auto chunk = audio.size() > MAX_CHUNK_SIZE ? audio.first(MAX_CHUNK_SIZE) : audio;

        auto mel = mMel.extract(first, second).toType(torch::kFloat16);
        
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
        const std::span<const float> first,
        const std::optional<std::span<const float>> second,
        const tle::VecTokens prompt,
        const TranscribeOptions &options,
        const std::optional<bool> stopAfterTimestamp
    ) {      
        auto mel = mMel.extract(first, second).toType(torch::kFloat16);

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
        auto logitsPostProcessorName = "transcribe";
        if (stopAfterTimestamp.has_value() && stopAfterTimestamp.value()) {
            logitsPostProcessorName = "transcribeSegment";
        }
        request.setLogitsPostProcessorName(logitsPostProcessorName);

        auto samplingConfig = tle::SamplingConfig(options.beamWidth, options.topK);
        if (options.topP > 0) {
            samplingConfig.setTopP(options.topP);
        }
        samplingConfig.setTemperature(options.temperature);
        //samplingConfig.setEarlyStopping(std::nullopt);
        samplingConfig.setEarlyStopping(0);
        samplingConfig.setLengthPenalty(1);
        //samplingConfig.setNoRepeatNgramSize(0);
        //samplingConfig.setBeamSearchDiversityRate(0.2);
        samplingConfig.setNumReturnSequences(1);
        request.setSamplingConfig(samplingConfig);

        tle::OutputConfig outputConfig;
        outputConfig.returnLogProbs = true;
        request.setOutputConfig(outputConfig);

        auto requestId = mExecutor.enqueueRequest(request);
        mTranscribeLogitsProcessor.addRequest(requestId, prompt.size());

        return requestId;
    }

    TranscribeResult Whisper::awaitTranscribeResponse(
        IdType const &requestId        
    ) {
        auto response = mExecutor.awaitResponses(requestId)[0];
        auto result = response.getResult();
    
        if (result.isFinal) {
            mTranscribeLogitsProcessor.removeRequest(requestId);
        }

        //std::cout << "result.isFinal: " << result.isFinal << std::endl;
        //std::cout << "result.isSequenceFinal: " << result.isSequenceFinal << std::endl;
        //std::cout << "result.outputTokenIds.size(): " << result.outputTokenIds.size() << std::endl;

        return TranscribeResult {
            result.isFinal,
            result.isSequenceFinal,
            result.outputTokenIds[0],
            result.cumLogProbs.value()[0] / static_cast<float>(result.logProbs.value()[0].size() + 1)
        };
    
        /*
        // Find the beam with highest average log probability
        size_t bestBeamIdx = 0;
        float maxAvgLogProb = -std::numeric_limits<float>::infinity();
        
        const auto& cumLogProbs = result.cumLogProbs.value();
        
        // Calculate average log probabilities for each beam
        for (size_t i = 0; i < result.outputTokenIds.size(); i++) {
            const auto& logProbs = result.logProbs.value()[i];
            auto nNewTokens = logProbs.size();
            //float sumLogProbs = std::accumulate(logProbs.begin(), logProbs.end(), 0.0f);
            float avgLogProb = cumLogProbs[i] / static_cast<float>(nNewTokens + 1);

            if (avgLogProb > maxAvgLogProb) {
                maxAvgLogProb = avgLogProb;
                bestBeamIdx = i;
            }

            std::vector<int> newTokens(result.outputTokenIds[i].end() - nNewTokens, result.outputTokenIds[i].end());

            std::cout << "beam: " << i
                << " cumLogProb: " << cumLogProbs[i]
                //<< " sumLogProbs: " << sumLogProbs
                << " avgLogProb: " << avgLogProb 
                << " nNewTokens: " << nNewTokens
                << std::endl;
            std::cout << "tokens: " << newTokens << std::endl;
        }
        
        return TranscribeResult {
            result.isFinal,
            result.isSequenceFinal,
            result.outputTokenIds[bestBeamIdx],
            maxAvgLogProb
        };
        */
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
        tle::StreamPtr const& streamPtr,
        bool stopAfterTimestamp
    ) {
        at::cuda::CUDAStreamGuard guard(tlr::TorchUtils::stream(*streamPtr));

        Logits logits(tleLogits);

        mMutex.lock();
        auto sampleBegin = mTranscribeContextMap[reqId].sampleBegin;
        mMutex.unlock();

        // suppress notimestamps
        logits.suppressNoTimestamps();

        // suppress blank at the beginning
        if (tokens[0].size() == sampleBegin) {
            logits.suppressBlank();
            logits.suppressNonTimestamp();
            return;
        }
        
        bool checkTimestampsProb = false;

        for (auto b = 0; b < tokens.size(); b++) {
            auto beamLogits = logits.beam(b);
            auto beamTokens = tokens[b];
    
            auto nTokens = beamTokens.size();
            bool lastWasTimestamp = nTokens > sampleBegin &&
                token::isTimestamp(beamTokens[nTokens - 1]);
            bool penultimateWasTimestamp = nTokens < sampleBegin + 2 ||
                nTokens > sampleBegin + 1 && token::isTimestamp(beamTokens[nTokens - 2]);
    
            if (lastWasTimestamp) {
                if (penultimateWasTimestamp) {
                    beamLogits.suppressTimestamps();
                    //beamLogits.suppressEndOfText();
                } else {
                    if (stopAfterTimestamp) {
                        // suppress all except non-endoftext
                        beamLogits.suppressNonEndOfText();
                        continue;
                    }

                    beamLogits.suppressText();
                    beamLogits.suppressTimestamps(beamTokens[nTokens - 1]);
                    checkTimestampsProb = true;
                }
            } else {
                for (auto i = nTokens - 1; i >= sampleBegin; i--) {
                    auto token = beamTokens[i];
                    if (token::isTimestamp(token)) {
                        beamLogits.suppressTimestamps(token + 1);
                        break;
                    }
                }
                checkTimestampsProb = true;
            }
        }

        if (checkTimestampsProb) {
            auto logprobs = logits.logprobs();
            for (auto b = 0; b < tokens.size(); b++) {
                auto beamLogprobs = logprobs.beam(b);
                auto timestampsLogprob = beamLogprobs.timestamps().logsumexp();

                auto maxTextLogprob = beamLogprobs.nonTimestamps().max();

                if (timestampsLogprob > maxTextLogprob) {
                    logits.beam(b).suppressNonTimestamp();
                        //std::cout << tokens[b] << std::endl;
                }
            }
        }
    }
}


            /*




            bool sampleTimestamp = false;
            mMutex.lock();
            if (timestampLogprob > mTranscribeContextMap[reqId].prevTimestampLogprob + 5) {
                sampleTimestamp = true;
            }
            if (!timestampSuppressed) {
                mTranscribeContextMap[reqId].prevTimestampLogprob = timestampLogprob;
            }
            mMutex.unlock();
                
            if (sampleTimestamp) {
                logits.beam(b).suppressNonTimestamp();
            }
            */

            /*
                if (tokens[b].size() < 30 && !token::isClauseEnd(tokens[b])) {
                    logits.beam(b).suppressTimestamps();
                    // std::cout << " text";
                } else if (timestampLogprob > maxTextLogprob) {
                    //logits.beam(b).suppressNonTimestamp();
                    logits.beam(b).suppressText();
                    //std::cout << "timestampLogprob: " << timestampLogprob 
                    //    << " maxTextLogprob: " << maxTextLogprob
                    //    << " token: " << tokens[b].back()
                    //    << std::endl;
                    // std::cout << " timestamp";
                }
                //std::cout << std::endl;
                //if (timestampLogprob > maxTextLogprob) {
                //    logits.beam(b).suppressNonTimestamp();
                //}
            */
        // TODO: suppress max_initial_timestamp_index



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
