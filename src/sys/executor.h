#pragma once

#include <memory>
#include <filesystem>

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include "rust/cxx.h"

namespace tle = tensorrt_llm::executor;

const tle::SizeType32 BEAM_WIDTH = 4;

inline void initialize() {
    initTrtLlmPlugins();
}

inline std::unique_ptr<tle::Executor> executor(rust::Str modelPathStr, tle::BatchingType batchingType) {
    auto modelPath = static_cast<std::string>(modelPathStr);
    auto encoderModelPath = std::filesystem::path(modelPath + "/encoder");
    auto decoderModelPath = std::filesystem::path(modelPath + "/decoder");

    auto executorConfig = tle::ExecutorConfig(BEAM_WIDTH);
    executorConfig.setBatchingType(batchingType);

    return std::make_unique<tle::Executor>(
        encoderModelPath, 
        decoderModelPath, 
        tle::ModelType::kENCODER_DECODER, 
        executorConfig
    );
}