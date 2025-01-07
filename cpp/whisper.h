#include "mel.h"
#include "tensorrt_llm/executor/executor.h"
#include <filesystem>
//#include <span>

namespace tle = tensorrt_llm::executor;

const int CHUNK_SIZE = 30;
const int SAMPLING_RATE = 16000;
const int MAX_NEW_TOKENS = 96;

using tle::BatchingType;

struct Config {
    BatchingType batchingType = BatchingType::kSTATIC;
};

struct TranscribeResult {
    tle::VecTokens output;
};

class Whisper {
    public:
        Whisper(
            std::filesystem::path const& modelPath,
            Config config
        );

        TranscribeResult transcribe(
            //std::span<float> audio,
            std::vector<float> audio,
            tle::VecTokens prompt
        );

    private:
        LogMelSpectrogram mMel;
        tle::Executor mExecutor;
};