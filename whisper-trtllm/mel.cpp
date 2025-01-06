#include "mel.h"
#include "cnpy/cnpy.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <filesystem>

torch::Tensor load_mel_filters(std::filesystem::path const& melFilterPath, const int nMels) {
    // Load the npz file
    cnpy::npz_t file = cnpy::npz_load(melFilterPath);

    // Load the mel_80 filter array
    std::string melName = std::string("mel_") + std::to_string(nMels);
    cnpy::NpyArray melArray = file[melName];

    // Get the shape of the array
    std::vector<size_t> shape = melArray.shape;

    // Convert the data to a torch::Tensor
    float* data = melArray.data<float>();
    torch::Tensor tensor = torch::from_blob(data, {static_cast<long>(shape[0]), static_cast<long>(shape[1])}, torch::kFloat32);

    return tensor;
}

LogMelSpectrogram::LogMelSpectrogram(
    std::filesystem::path const& melFilterPath, 
    const int nMels,
    const int nFFT,
    const int hopLength,
    torch::Device const& device
) : mNFFT(nFFT), mHopLength(hopLength), mDevice(device) {
    mFilters = load_mel_filters(melFilterPath, nMels).to(mDevice);
    mWindow = torch::hann_window(nFFT).to(mDevice);
}

torch::Tensor LogMelSpectrogram::extract(std::span<float> audio, const int padding) {
    // Convert audio to tensor
    torch::Tensor audioTensor = torch::from_blob((void*)audio.data(), {1, (long)audio.size()}, torch::kFloat32).to(mDevice);

    if (padding > 0) {
        audioTensor = torch::nn::functional::pad(
            audioTensor, 
            torch::nn::functional::PadFuncOptions({padding, 0}).mode(torch::kConstant).value(0));
    }

    torch::Tensor stft = torch::stft(audioTensor, mNFFT, mHopLength, mNFFT, mWindow, true, false, true);
    auto magnitudes = stft.slice(-1, 0, mNFFT / 2 + 1).abs().pow(2);

    torch::Tensor mel_spec = torch::matmul(mFilters, magnitudes);

    torch::Tensor log_spec = torch::clamp(mel_spec, 1e-10).log10();
    log_spec = torch::maximum(log_spec, log_spec.max() - 8.0);
    log_spec = (log_spec + 4.0) / 4.0;

    return log_spec;
}