#include <torch/torch.h>
#include <filesystem>
#include <span>

const int N_MELS = 128;
const int N_FFT = 400;
const int HOP_LENGTH = 160;

class LogMelSpectrogram {
    public:
        LogMelSpectrogram(
            std::filesystem::path const& melFilterPath,
            const int nMels = N_MELS,
            const int nFFT = N_FFT,
            const int hopLength = HOP_LENGTH,
            torch::Device const& device = torch::kCUDA
        );

        torch::Tensor extract(
            const std::span<const float> first, 
            const std::optional<std::span<const float>> second
        ) const;
        //torch::Tensor extract(const std::vector<float> audio, const int padding) const;

        int nMels() const {
            return mFilters.size(0);
        }

        int nFFT() const {
            return mNFFT;
        }

        int hopLength() const {
            return mHopLength;
        }

    private:
        torch::Tensor mFilters;
        torch::Tensor mWindow;
        int mNFFT;
        int mHopLength;
        //torch::Device mDevice;
};