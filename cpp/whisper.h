#include "mel.h"
#include "tensorrt_llm/executor/executor.h"

#include <cuda_fp16.h>
#include <mutex>
#include <filesystem>
#include <span>

namespace tle = tensorrt_llm::executor;

const int CHUNK_SIZE = 30;
const int SAMPLING_RATE = 16000;
const int MAX_NEW_TOKENS = 96;

using tle::BatchingType;
using tle::VecTokens;
using tle::IdType;
using tle::TokenIdType;

namespace tensorrt_llm::whisper {

    struct Config {
        BatchingType batchingType = BatchingType::kINFLIGHT;
    };

    struct TranscribeResult {
        VecTokens tokens;
    };

    struct RequestContext {
        torch::Half mPrevTimestampLogProb;
    };
    
    class TranscribeLogitsProcessor {
        public:
            void process(
                tle::IdType reqId,
                tle::Tensor& logits, 
                tle::BeamTokens const& tokens,
                tle::StreamPtr const& streamPtr
            );
    
        private:
            std::mutex mRequestContextMapMutex;
            std::unordered_map<IdType, RequestContext> mRequestContextMap;
    };

    class Whisper {
        public:
            Whisper(
                std::filesystem::path const& modelPath,
                const Config config
            );

            IdType enqueueDetectLanguageRequest(
                const std::span<const float> audio
            );

            TokenIdType awaitDetectLanguageResponse(
                IdType const &requestId        
            );

            IdType enqueueTranscribeRequest(
                const std::span<const float> audio,
                const tle::VecTokens prompt
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