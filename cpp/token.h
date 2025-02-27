#include "tensorrt_llm/executor/executor.h"

namespace tle = tensorrt_llm::executor;

using tle::TokenIdType;

namespace tensorrt_llm::whisper::token {
    const TokenIdType SPACE = 256;

    const TokenIdType END_OF_TEXT = 50257;

    const TokenIdType START_OF_TRANSCRIPT = 50258;

    const TokenIdType BEGIN_OF_LANGUAGE = 50259;
    const TokenIdType END_OF_LANGUAGE = 50359;

    const TokenIdType TRANSCRIBE  = 50360;

    const TokenIdType NO_TIMESTAMPS = 50364;

    const TokenIdType BEGIN_OF_TIMESTAMP = 50365;
    const TokenIdType END_OF_TIMESTAMP = 51866;

    bool isTimestamp(TokenIdType token) {
        return token >= BEGIN_OF_TIMESTAMP && token < END_OF_TIMESTAMP;
    }

    bool isClauseEnd(const std::vector<TokenIdType>& tokens) {
        const size_t n = tokens.size();
        return n > 0 && (tokens[n - 1] == 11 || tokens[n - 1] == 13 || tokens[n - 1] == 0 || tokens[n - 1] == 30 || tokens[n - 1] == 1543) || // ,.!?。
            n > 2 && tokens[n - 3] == 171 && tokens[n - 2] == 120 && (tokens[n - 1] == 234 || tokens[n - 1] == 223 || tokens[n - 1] == 253); // ，！？
    }
}