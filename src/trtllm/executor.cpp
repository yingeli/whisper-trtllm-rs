#include <string>

#include "tensorrt_llm/executor/executor.h"

#include "rust/cxx.h"

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

class Executor {
private:
    std::unique_ptr<tle::Executor> impl;
    
public:
    Executor(std::unique_ptr<tle::Executor> impl)
        : impl(std::move(impl)) { }
    
    inline void shutdown() const {
        return impl->shutdown();
    }
};

inline std::unique_ptr<Executor> executor(
    rust::Str model_path,
    std::unique_ptr<Config> config
) {
    return std::make_unique<Whisper>(
        std::make_unique<tle::Executor>(
            static_cast<std::string>(model_path),
            config->device,
            config->compute_type,
            std::vector<int>(config->device_indices.begin(), config->device_indices.end()),
            config->tensor_parallel,
            *config->replica_pool_config
        )
    );
}

    /*
int main(int argc, char* argv[])
{

    // Register the TRT-LLM plugins
    initTrtLlmPlugins();

    if (argc != 2)
    {
        TLLM_LOG_ERROR("Usage: %s <dir_with_engine_files>", argv[0]);
        return 1;
    }

    // Create the executor for this engine
    tle::SizeType32 beamWidth = 1;
    auto executorConfig = tle::ExecutorConfig(beamWidth);
    auto trtEnginePath = argv[1];
    auto executor = tle::Executor(trtEnginePath, tle::ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    tle::SizeType32 maxNewTokens = 5;
    tle::VecTokens inputTokens{1, 2, 3, 4};
    auto request = tle::Request(inputTokens, maxNewTokens);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(request);

    // Wait for the response
    auto responses = executor.awaitResponses(requestId);

    // Get outputTokens
    auto outputTokens = responses.at(0).getResult().outputTokenIds.at(beamWidth - 1);

    TLLM_LOG_INFO("Output tokens: %s", tlc::vec2str(outputTokens).c_str());


    return 0;
}
        */
