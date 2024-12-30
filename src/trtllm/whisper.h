#include <memory>

#include "rust/cxx.h"

#include "/home/coder/whisper-trt/TensorRT-LLM/cpp/include/tensorrt_llm/executor/executor.h"

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

inline tle::Executor load_executor(
    rust::Str model_path
) {
    tle::SizeType32 beamWidth = 1;
    auto config = tle::ExecutorConfig(beamWidth);
    auto path = static_cast<std::string>(model_path);
    auto executor = tle::Executor(path, tle::ModelType::kENCODER_DECODER, config);
    return executor;
}