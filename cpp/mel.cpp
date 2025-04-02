#include "mel.h"

#include "cnpy/cnpy.h"

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>
#include <vector>
#include <filesystem>

LogMelSpectrogram::LogMelSpectrogram(
    std::filesystem::path const& melFilterPath, 
    const int nMels,
    const int nFFT,
    const int hopLength,
    torch::Device const& device
) : mNFFT(nFFT), 
    mHopLength(hopLength) {
    // Load the npz file
    cnpy::npz_t file = cnpy::npz_load(melFilterPath);

    // Load the mel_128 filter array
    std::string melName = std::string("mel_") + std::to_string(nMels);
    cnpy::NpyArray melArray = file[melName];
    std::vector<int64_t> shape(melArray.shape.begin(), melArray.shape.end());
    
    mFilters = torch::from_blob(melArray.data<void>(), shape, torch::kFloat32).to(device);
    
    mWindow = torch::hann_window(nFFT).to(device);
}

torch::Tensor LogMelSpectrogram::extract(
    const std::span<const float> first, 
    const std::optional<std::span<const float>> second
) const {
    auto totalSize = first.size() + (second.has_value() ? second.value().size() : 0);

    auto device = mFilters.device();
    
    /*
    torch::Tensor samples = torch::from_blob(
        (void*)first.data(), 
        (long)first.size(),
        torch::kFloat32).to(device);
    */

    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device);

    torch::Tensor samples = torch::empty({static_cast<long>(totalSize)}, options);

    float* data_ptr = samples.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    cudaMemcpyAsync(
        data_ptr,
        first.data(),
        first.size_bytes(),
        cudaMemcpyHostToDevice,
        stream
    );

    if (second.has_value()) { 
        auto s = second.value();
        if (s.size() > 0) {
            cudaMemcpyAsync(
                data_ptr + first.size(),
                s.data(),
                s.size_bytes(),
                cudaMemcpyHostToDevice,
                stream
            );
        }
    }

    cudaStreamSynchronize(stream);

    int padding = totalSize % mHopLength == 0 ? 0 : mHopLength - (totalSize % mHopLength);
    if (padding > 0) {
        samples = torch::nn::functional::pad(
            samples, 
            torch::nn::functional::PadFuncOptions({0, padding}).mode(torch::kConstant).value(0));
    }

    torch::Tensor stft = torch::stft(samples, 
        mNFFT, 
        mHopLength, 
        mNFFT, 
        mWindow, 
        true, // center
        "reflect", // pad_mode
        false, // normalized
        true, // onesided
        true // return_complex
    );

    std::cout << "STFT shape: " << stft.sizes() << std::endl;

    auto magnitudes = stft.slice(-1, 0, stft.size(stft.dim() - 1) - 1).abs().pow(2);

    torch::Tensor mel_spec = torch::matmul(mFilters, magnitudes);

    torch::Tensor log_spec = torch::clamp_min(mel_spec, 1e-10).log10();
    log_spec = torch::maximum(log_spec, log_spec.max() - 8.0);
    log_spec = (log_spec + 4.0) / 4.0;

    return log_spec;
}

/*
torch::Tensor LogMelSpectrogram::extract(
    const std::span<const float> samples
) const {
    int padding = samples.size() % mHopLength == 0 ? 0 : mHopLength - (samples.size() % mHopLength);
    
    // Convert audio to tensor
    auto device = mFilters.device();
    torch::Tensor tensor = torch::from_blob(
        (void*)samples.data(), 
        (long)samples.size(),
        torch::kFloat32).to(device);

    return compute(tensor, padding);
}
*/



/*
torch::Tensor LogMelSpectrogram::extract(const std::vector<float> audio, const int padding) const {
    // Convert audio to tensor
    torch::Tensor audioTensor = torch::from_blob((void*)audio.data(), 
        {1, (long)audio.size()}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

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
*/