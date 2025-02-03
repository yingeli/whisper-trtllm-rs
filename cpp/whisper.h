#include "mel.h"
#include "tensorrt_llm/executor/executor.h"
#include <filesystem>
#include <span>

namespace tle = tensorrt_llm::executor;

const int CHUNK_SIZE = 30;
const int SAMPLING_RATE = 16000;
const int MAX_NEW_TOKENS = 96;

using tle::BatchingType;
using tle::VecTokens;
using tle::IdType;

namespace tensorrt_llm::whisper {

    struct Config {
        BatchingType batchingType = BatchingType::kINFLIGHT;
    };

    struct TranscribeResult {
        VecTokens tokens;
    };

    class Whisper {
        public:
            Whisper(
                std::filesystem::path const& modelPath,
                const Config config
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
            tle::Executor mExecutor;
            float mPrevTimestampLogProb;
    };

}