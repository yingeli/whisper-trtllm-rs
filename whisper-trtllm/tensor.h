#include <torch/torch.h>
#include "tensorrt_llm/runtime/iTensor.h"

class TorchTensor : public ITensor
{
public:
    explicit TorchTensor(const torch::Tensor& tensor) : mTensor(tensor) {}

    ~TorchTensor() override = default;

    Shape const& getShape() const override
    {
        return mTensor.shape();
        /*
        shape_.nbDims = tensor_.dim();
        for (int i = 0; i < shape_.nbDims; ++i)
        {
            shape_.d[i] = tensor_.size(i);
        }
        return shape_;
        */
    }

    void reshape(Shape const& dims) override
    {
        //std::vector<int64_t> sizes(dims.d, dims.d + dims.nbDims);
        mTensor = mTensor.reshape(dims);
    }

    std::size_t getSize() const override
    {
        return mTensor.numel();
    }

    void* getData() override
    {
        return mTensor.data_ptr();
    }

    const void* getData() const override
    {
        return mTensor.data_ptr();
    }

    torch::Tensor& getTensor()
    {
        return mTensor;
    }

    const torch::Tensor& getTensor() const
    {
        return mTensor;
    }

private:
    torch::Tensor mTensor;
    mutable Shape mShape;
};