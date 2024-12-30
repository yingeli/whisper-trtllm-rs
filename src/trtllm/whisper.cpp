#include "tensorrt_llm/executor/executor.h"

#include "rust/cxx.h"

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

class Whisper {
private:
    tle::Executor executor;
    
public:
    Whisper(tle::Executor executor);
};

inline std::unique_ptr<Whisper> load(
    rust::Str model_path
) {
    tle::SizeType32 beamWidth = 1;
    auto config = tle::ExecutorConfig(beamWidth);
    auto path = static_cast<std::string>(model_path);
    auto executor = tle::Executor(path, tle::ModelType::kENCODER_DECODER, config);
    return std::make_unique<Whisper>(executor);
}