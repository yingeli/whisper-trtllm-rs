#include <torch/torch.h>
#include <filesystem>
#include <span>

class FeatureBuffer {
    public:
        FeatureBuffer(
            LogMelSpectrogram logMel,
            const int64_t chunkSize = 3000
        );

        torch::Tensor chunk() const {
            return mBuffer.slice(0, 0, mChuckSize);
        }

        void append(
            const std::span<const float> samples
        );

        torch::Tensor consume(const int64_t amt);

    private:
        LogMelSpectrogram mExtractor;

        torch::Tensor mBuffer;

        std::vector<const float> mPrev;
        int64_t mNPrevHops;
        
        int64_t mChuckSize;
};
