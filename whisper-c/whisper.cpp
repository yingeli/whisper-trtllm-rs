#include "tensorrt_llm/executor/executor.h"

#include "whisper.h"

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

struct Whisper
{
    tle::Executor executor;
};

inline Whisper load(
    std::filesystem::path const& modelPath
) {
    tle::SizeType32 beamWidth = 1;
    auto config = tle::ExecutorConfig(beamWidth);
    auto executor = tle::Executor(modelPath, tle::ModelType::kENCODER_DECODER, config);
    return Whisper{executor};;
}