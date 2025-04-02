#pragma once

#include "whisper-trtllm-rs/src/sys/vocab.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/runtime/torchView.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <torch/torch.h>

namespace tlr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;

const torch::Half NEG_INF = static_cast<torch::Half>(-std::numeric_limits<float>::infinity());

class Logprobs {
    public:
        Logprobs(torch::Tensor tensor): tensor_(tensor) {}

        Logprobs beam(int64_t beam) {
            auto tensor = tensor_.index({0, beam});
            return Logprobs(tensor);
        }

        Logprobs timestamps() {
            return slice(token::START_OF_TIMESTAMP);
        }

        Logprobs non_timestamps() {
            return slice(0, token::START_OF_TIMESTAMP);
        }

        float max() {
            return std::get<0>(tensor_.max(-1)).item<float>();
        }

        float logsumexp() {
            return tensor_.logsumexp(-1).item<float>();
        }

    private:
        Logprobs slice(std::optional<int64_t> start = std::nullopt, std::optional<int64_t> end = std::nullopt) {
            return Logprobs(tensor_.slice(-1, start, end));
        }

        torch::Tensor tensor_;
};

class Logits {
    public:
        Logits(
            tle::Tensor& logits
        ) : tensor_(tlr::Torch::tensor(tle::detail::toITensor(logits)))
        {}

        Logits(
            torch::Tensor& logits
        ) : tensor_(logits)
        {}

        Logits beam(int64_t beam) {
            auto tensor = tensor_.index({0, beam});
            return Logits(tensor);
        }

        Logprobs logprobs() {
            auto tensor = torch::nn::functional::log_softmax(tensor_.to(torch::kFloat32), 2);
            return Logprobs(tensor);
        }

        void set_transcribe() {
            tensor_.fill_(NEG_INF);
            tensor_.select(-1, token::TRANSCRIBE).fill_(0);
        }

        void set_eot() {
            tensor_.fill_(NEG_INF);
            tensor_.select(-1, token::END_OF_TEXT).fill_(0);
        }

        void suppress_notimestamps() {
            suppress(token::NO_TIMESTAMPS);
        }

        void suppress_non_languages() {
            suppress_range(0, token::START_OF_LANGUAGE);
            suppress_range(token::END_OF_LANGUAGE);
        }

        void suppress_eot() {
            suppress(token::END_OF_TEXT);
        }

        void suppress_non_eot() {
            suppress_range(0, token::END_OF_TEXT);
            suppress_range(token::END_OF_TEXT + 1);
        }

        void suppress_timestamps(std::optional<tle::TokenIdType> end = std::nullopt) {
            suppress_range(token::START_OF_TIMESTAMP, end);
        }

        void suppress_non_timestamps() {
            suppress_range(0, token::START_OF_TIMESTAMP);
        }

        void suppress_text() {
            suppress_range(0, token::END_OF_TEXT);
        }

        void suppress_blank() {
            torch::Tensor indices = torch::tensor({token::SPACE, token::END_OF_TEXT}, torch::kLong);
            suppress_indices(indices);
        }

    private:
        void suppress(
            const tle::TokenIdType id
        ) {
            tensor_.select(-1, id).fill_(NEG_INF);
        }

        void suppress_range(
            const tle::TokenIdType begin, 
            const std::optional<const tle::TokenIdType> end = std::nullopt
        ) {
            tensor_.slice(-1, begin, end).fill_(NEG_INF);
        }

        void suppress_indices(
            torch::Tensor indices
        ) {
            int64_t n = tensor_.dim();
            std::vector<torch::indexing::TensorIndex> idx(n, torch::indexing::Slice());
            idx[n - 1] = indices;
            tensor_.index_put_(idx, NEG_INF);
        }

        torch::Tensor tensor_;
};