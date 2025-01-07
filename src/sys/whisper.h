#pragma once

#include "cpp/whisper.h"

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include <memory>
#include <filesystem>

#include "rust/cxx.h"

//const tle::SizeType32 BEAM_WIDTH = 4;

inline std::unique_ptr<Whisper> whisper(rust::Str modelPathStr, Config config) {
    auto modelPath = std::filesystem::path(static_cast<std::string>(modelPathStr));
    //auto encoderModelPath = std::filesystem::path(modelPath + "/encoder");
    //auto decoderModelPath = std::filesystem::path(modelPath + "/decoder");

    //auto executorConfig = tle::ExecutorConfig(BEAM_WIDTH);
    //executorConfig.setBatchingType(batchingType);

    return std::make_unique<Whisper>(
        modelPath,
        config
    );
}

inline bool init() {
    return initTrtLlmPlugins();
}