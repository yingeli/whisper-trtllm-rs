#pragma once

#include <memory>
#include <filesystem>

#include "tensorrt_llm/executor/executor.h"

#include "rust/cxx.h"

xxx

//namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

const tle::SizeType32 BEAM_WIDTH = 1;

class Whisper
{
private:
    tle::Executor executor;

public:
    Whisper(std::filesystem::path const& modelPath)
        : executor(tle::Executor(modelPath, tle::ModelType::kENCODER_DECODER, tle::ExecutorConfig(BEAM_WIDTH)))
    {}
};

inline std::unique_ptr<Whisper> load(rust::Str modelPath) {
    auto str = static_cast<std::string>(modelPath);
    auto path = std::filesystem::path(str);
    return std::make_unique<Whisper>(path);
}