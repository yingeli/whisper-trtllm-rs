#pragma once

#include "whisper-trtllm-rs/cpp/whisper.h"

#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include <memory>
#include <filesystem>

#include "rust/cxx.h"

namespace tle = tensorrt_llm::executor;
namespace tlw = tensorrt_llm::whisper;

using tle::BatchingType;
using tle::Tensor;
using tle::VecTokens;
using tlw::Config;

class Whisper {
    public:
        /*
        Whisper(
            rust::Str const& model_path_str,
            Config config
        ) : Whisper(
            std::filesystem::path(static_cast<std::string>(model_path_str)),
            config
        ) {}
        */

        Whisper(
            std::filesystem::path const& model_path,
            Config config
        );

        /*
        TranscribeResult transcribe(
            //std::vector<float> audio,
            Tensor features,
            VecTokens prompts
        );
        */

    private:
        tlw::Whisper _inner;
};

inline std::unique_ptr<Whisper> whisper(const rust::Str model_path_str, const Config config) {
    auto model_path = std::filesystem::path(static_cast<std::string>(model_path_str));
    return std::make_unique<Whisper>(
        model_path,
        config
    );
}

inline bool init() {
    return initTrtLlmPlugins();
}