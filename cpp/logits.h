#include "token.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/runtime/torchView.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <torch/torch.h>

namespace tlr = tensorrt_llm::runtime;
namespace tle = tensorrt_llm::executor;

using tle::TokenIdType;

namespace tensorrt_llm::whisper {
    const torch::Half NEG_INF = static_cast<torch::Half>(-std::numeric_limits<float>::infinity());

    /*
    torch::Tensor toTorch(tle::Tensor& logits) {
        return tlr::Torch::tensor(tle::detail::toITensor(logits));
        /*
        auto const tensorOptions = torch::device(tlr::TorchUtils::device(logits.getData()))
            .pinned_memory(logits.getMemoryType() == tle::MemoryType::kCPU_PINNEDPOOL)
            //.dtype(tlr::TorchUtils::dataType(logits.getDataType()))
            .dtype(torch::kFloat16)
            .layout(torch::kStrided);

        return torch::from_blob(logits.getData(), {1, 1, logits.getShape()[2]}, tensorOptions);
    }
        */

    class SingleBatchTensor {
        public:
            SingleBatchTensor(
                torch::Tensor tensor
            ) : mTensor(tensor)
            {}

            void put(
                const TokenIdType id,
                torch::Half value
            ) {
                mTensor.index_put_({
                    torch::indexing::Slice(), 
                    torch::indexing::Slice(), 
                    id
                }, value);
            }

            void putRange(
                const TokenIdType begin, 
                const TokenIdType end,
                torch::Half value
            ) {
                mTensor.slice(2, begin, end).fill_(value);
            }

            void putRange(
                const TokenIdType begin, 
                torch::Half value
            ) {
                mTensor.slice(2, begin).fill_(value);
            }

            void putIndices(
                torch::Tensor indices,
                torch::Half value
            ) {
                mTensor.index_put_({
                    torch::indexing::Slice(), 
                    torch::indexing::Slice(), 
                    indices
                }, value);
            }

        protected:
            torch::Tensor mTensor;
    };

    class BeamTensor {
        public:
            BeamTensor(torch::Tensor tensor): mTensor(tensor) {}

            void put(
                const TokenIdType id,
                torch::Half value
            ) {
                mTensor.index_put_({id}, value);
            }

            void putRange(
                const TokenIdType begin, 
                const TokenIdType end,
                torch::Half value
            ) {
                mTensor.slice(0, begin, end).fill_(value);
            }

            void putRange(
                const TokenIdType begin, 
                torch::Half value
            ) {
                mTensor.slice(0, begin).fill_(value);
            }

            void putIndices(
                torch::Tensor indices,
                torch::Half value
            ) {
                mTensor.index_put_({indices}, value);
            }

            BeamTensor slice(std::optional<int64_t> start = std::nullopt, std::optional<int64_t> end = std::nullopt) {
                return BeamTensor(mTensor.slice(0, start, end));
            }

            float max() {
                return std::get<0>(mTensor.max(0)).item<float>();
            }

            float logsumexp() {
                return mTensor.logsumexp(0).item<float>();
                //return torch::logsumexp(mTensor, 0).item<torch::Half>();
            }

        protected:
            torch::Tensor mTensor;
    };

    class BeamLogprobs: BeamTensor {
        public:
            BeamLogprobs(torch::Tensor tensor): BeamTensor(tensor) {}

            BeamTensor timestamps() {
                return slice(token::BEGIN_OF_TIMESTAMP);
            }

            BeamTensor nonTimestamps() {
                return slice(0, token::BEGIN_OF_TIMESTAMP);
            }
    };

    class Logprobs: SingleBatchTensor {
        public:
            Logprobs(torch::Tensor tensor): SingleBatchTensor(tensor) {}

            BeamLogprobs beam(int64_t beam) {
                auto tensor = mTensor.index({0, beam});
                return BeamLogprobs(tensor);
            }
    };

    class BeamLogits: BeamTensor {
        public:
            BeamLogits(torch::Tensor tensor): BeamTensor(tensor) {}

            void suppressNoTimestamps() {
                put(token::NO_TIMESTAMPS, NEG_INF);
            }

            void suppressEndOfText() {
                put(token::END_OF_TEXT, NEG_INF);
            }

            void suppressNonLanguage() {
                putRange(0, token::BEGIN_OF_LANGUAGE, NEG_INF);
                putRange(token::END_OF_LANGUAGE, NEG_INF);
            }

            void suppressText() {
                putRange(0, token::END_OF_TEXT, NEG_INF);
            }

            void suppressTimestamps() {
                putRange(token::BEGIN_OF_TIMESTAMP, NEG_INF);
            }

            void suppressTimestamps(TokenIdType end) {
                putRange(token::BEGIN_OF_TIMESTAMP, end, NEG_INF);
            }

            void suppressNonTimestamp() {
                putRange(0, token::BEGIN_OF_TIMESTAMP, NEG_INF);
            }

            void suppressNonEndOfText() {
                putRange(0, token::END_OF_TEXT, NEG_INF);
                putRange(token::END_OF_TEXT + 1, NEG_INF);
            }

            void suppressBlank() {
                torch::Tensor indices = torch::tensor({token::SPACE, token::END_OF_TEXT}, torch::kLong);
                putIndices(indices, NEG_INF);
            }

            torch::IntArrayRef sizes() {
                return mTensor.sizes();
            }
    };

    class Logits: SingleBatchTensor  {
        public:
            Logits(
                tle::Tensor& logits
            ) : SingleBatchTensor(tlr::Torch::tensor(tle::detail::toITensor(logits)))
            {}

            void suppressNoTimestamps() {
                put(token::NO_TIMESTAMPS, NEG_INF);
            }

            void suppressNonLanguage() {
                putRange(0, token::BEGIN_OF_LANGUAGE, NEG_INF);
                putRange(token::END_OF_LANGUAGE, NEG_INF);
            }

            void suppressNonTimestamp() {
                putRange(0, token::BEGIN_OF_TIMESTAMP, NEG_INF);
            }

            void suppressBlank() {
                torch::Tensor indices = torch::tensor({token::SPACE, token::END_OF_TEXT}, torch::kLong);
                putIndices(indices, NEG_INF);
            }

            BeamLogits beam(int64_t beam) {
                auto tensor = mTensor.index({0, beam});
                return BeamLogits(tensor);
            }

            Logprobs logprobs() {
                auto tensor = torch::nn::functional::log_softmax(mTensor.to(torch::kFloat32), 2);
                return Logprobs(tensor);
            }
    };
} // namespace tensorrt_llm::whisper