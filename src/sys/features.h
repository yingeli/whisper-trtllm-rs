#pragma once

#include <torch/torch.h>

const int64_t LENGTH_DIM = 0;

class Features {
    public:
        Features(
            const torch::Tensor tensor
        ): tensor_(tensor) {
        }

        inline size_t len() const {
            return tensor_.size(LENGTH_DIM);
        }

        inline std::unique_ptr<Features> slice(
            const size_t start,
            const size_t end
        ) const {
            return std::make_unique<Features>(tensor_.slice(LENGTH_DIM, start, end));
        }

        inline std::unique_ptr<Features> slice_to_end(
            const size_t start
        ) const {
            return std::make_unique<Features>(tensor_.slice(LENGTH_DIM, start));
        }

        inline std::unique_ptr<Features> pad(
            const size_t padding
        ) const {
            auto tensor = torch::nn::functional::pad(
                tensor_, 
                torch::nn::functional::PadFuncOptions({0, 0, 0, padding}).mode(torch::kConstant).value(-1.5));
            return std::make_unique<Features>(tensor);
        }

        inline std::unique_ptr<Features> join(const Features& other) const {
            auto tensor = torch::cat({tensor_, other.tensor_}, LENGTH_DIM);
            return std::make_unique<Features>(tensor);
        }

        inline const torch::Tensor& tensor() const {
            return tensor_;
        }

        inline torch::Tensor into_tensor() && {
            return std::move(tensor_);
        }

    private:
        torch::Tensor tensor_;
};

//inline std::unique_ptr<Features> features() {
//    auto tensor = torch::Tensor();
//    return std::make_unique<Features>(tensor);
//}