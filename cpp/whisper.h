#include <chrono>
#include "mel.h"
#include "tensorrt_llm/executor/executor.h"

#include <cuda_fp16.h>
#include <mutex>
#include <filesystem>
#include <span>

namespace tle = tensorrt_llm::executor;

const int MAX_CHUNK_SIZE = 30 * 16000;
const int MAX_NEW_TOKENS = 224;

using tle::BatchingType;
using tle::VecTokens;
using tle::BeamTokens;
using tle::VecLogProbs;
using tle::IdType;
using tle::TokenIdType;
using tle::SizeType32;
using tle::FloatType;

namespace tensorrt_llm::whisper {
    struct Config {
        BatchingType batchingType = BatchingType::kINFLIGHT;
    };

    struct TranscribeOptions {
        SizeType32 beamWidth;
        SizeType32 topK;
        FloatType topP;
        FloatType temperature;
    };

    struct TranscribeResult {
        bool isFinal;
        bool isSequenceFinal;
        VecTokens tokens;
        FloatType avgLogProb;
    };

    struct TranscribeContext {
        std::size_t sampleBegin;
        torch::Half prevTimestampLogprob;
    };
    
    class TranscribeLogitsProcessor {
        public:
            void addRequest(
                const IdType reqId, 
                const std::size_t sampleBegin
            );
    
            void removeRequest(
                const IdType reqId
            );

            void process(
                tle::IdType reqId,
                tle::Tensor& logits, 
                tle::BeamTokens const& tokens,
                tle::StreamPtr const& streamPtr,
                bool stopAfterTimestamp
            );
    
        private:
            std::mutex mMutex;
            std::unordered_map<IdType, TranscribeContext> mTranscribeContextMap;
    };

    class Whisper {
        public:
            Whisper(
                std::filesystem::path const& modelPath,
                const Config config
            );

            IdType enqueueDetectLanguageRequest(
                const std::span<const float> first,
                const std::optional<std::span<const float>> second
            );

            TokenIdType awaitDetectLanguageResponse(
                IdType const &requestId        
            );

            IdType enqueueTranscribeRequest(
                const std::span<const float> first,
                const std::optional<std::span<const float>> second,
                const tle::VecTokens prompt,
                const TranscribeOptions &options,
                const std::optional<bool> stopAfterTimestamp = std::nullopt
            );

            TranscribeResult awaitTranscribeResponse(
                IdType const &requestId            
            );
            
            bool isResponseReady(
                IdType const &requestId
            ) const;

        private:
            LogMelSpectrogram mMel;
            TranscribeLogitsProcessor mTranscribeLogitsProcessor;
            tle::Executor mExecutor;
    };
}