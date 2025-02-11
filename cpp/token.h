#include "tensorrt_llm/executor/executor.h"

namespace tle = tensorrt_llm::executor;

using tle::TokenIdType;

namespace tensorrt_llm::whisper::token {
    const TokenIdType END_OF_TEXT = 50257;

    const TokenIdType BEGIN_OF_LANGUAGE = 50259;
    const TokenIdType END_OF_LANGUAGE = 50359;

    const TokenIdType TRANSCRIBE  = 50360;

    const TokenIdType NO_TIMESTAMPS = 50363;

    const TokenIdType BEGIN_OF_TIMESTAMP = 50365;
}