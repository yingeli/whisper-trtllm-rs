#include "whisper-trtllm-rs/src/sys/mel.h"

#include "cnpy.h"

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>
#include <vector>
#include <filesystem>

LogMelSpectrogram::LogMelSpectrogram(
    const std::filesystem::path& mel_filter_path,
    const size_t n_mels,
    const size_t n_fft,
    const size_t hop_length,
    const torch::Device& device
) : n_fft_(n_fft),
    hop_length_(hop_length) {
    // Load the npz file
    cnpy::npz_t file = cnpy::npz_load(mel_filter_path);

    // Load the mel filter array
    std::string mel_name = std::string("mel_") + std::to_string(n_mels);
    cnpy::NpyArray mel_array = file[mel_name];
    std::vector<int64_t> shape(mel_array.shape.begin(), mel_array.shape.end());
    
    filters_ = torch::from_blob(mel_array.data<void>(), shape, torch::kFloat32).to(device);
    window_ = torch::hann_window(n_fft).to(device);
}

torch::Tensor LogMelSpectrogram::extract(
    const std::span<const float> first, 
    const std::optional<std::span<const float>> second,
    const std::optional<size_t> skip,
    const std::optional<bool> padding
) const {
    auto device = filters_.device();

    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device);

    auto total_size = first.size() + (second.has_value() ? second.value().size() : 0);
    
    auto padding_size = 0;
    if (padding.has_value() && padding.value()) {
        padding_size = n_fft_ / 2 - (total_size - 1) % hop_length_ - 1;
        total_size += padding_size;
    }

    torch::Tensor samples = torch::empty({static_cast<long>(total_size)}, options);

    float* data_ptr = samples.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (first.size() > 0) {
        cudaMemcpyAsync(
            data_ptr,
            first.data(),
            first.size_bytes(),
            cudaMemcpyHostToDevice,
            stream
        );
    }
    if (second.has_value() && second.value().size() > 0) { 
        cudaMemcpyAsync(
            data_ptr + first.size(),
            second.value().data(),
            second.value().size_bytes(),
            cudaMemcpyHostToDevice,
            stream
        );
    }
    if (padding_size > 0) {
        cudaMemsetAsync(
            data_ptr + total_size,  // Start position after all data
            0,                     // Value to set (0)
            padding_size * sizeof(float), // Size in bytes
            stream
        );
    }
    cudaStreamSynchronize(stream);

    torch::Tensor stft = torch::stft(samples, 
        n_fft_, 
        hop_length_, 
        n_fft_, 
        window_, 
        true, // center
        "reflect", // pad_mode
        false, // normalized
        true, // onesided
        true // return_complex
    );

    auto n_frames = total_size + hop_length_ > n_fft_ / 2 ? (total_size + hop_length_ - n_fft_ / 2) / hop_length_ : 0;
    auto start = 0;
    if (skip.has_value()) {
        start = skip.value();
    }
    auto magnitudes = stft.slice(-1, start, n_frames).abs().pow(2);
    // auto magnitudes = stft.slice(-1, 0, stft.size(stft.dim() - 1) - 1).abs().pow(2);

    torch::Tensor mel_spec = torch::matmul(filters_, magnitudes);

    torch::Tensor log_spec = torch::clamp_min(mel_spec, 1e-10).log10();
    log_spec = torch::maximum(log_spec, log_spec.max() - 8.0);
    log_spec = (log_spec + 4.0) / 4.0;

    log_spec = log_spec.toType(torch::kFloat16).transpose(0, 1);

    return log_spec;
}

/*
torch::Tensor LogMelSpectrogram::extract(
    const std::span<const float> first, 
    const std::optional<std::span<const float>> second,
    const std::optional<size_t> skip,
    const std::optional<size_t> pad_to
) const {
    auto device = filters_.device();

    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device);

    auto total_size = first.size() + (second.has_value() ? second.value().size() : 0);
    auto padding_size = 0;

    if (pad_to.has_value()) {
        auto n_skip = skip.has_value() ? skip.value() : 0;
        auto pad_to_size = (pad_to.value() + n_skip) * hop_length_ + n_fft_ / 2 - hop_length_;
        if (pad_to_size > total_size) {
            padding_size = pad_to_size - total_size;
            total_size = pad_to_size;
        }
    }

    torch::Tensor samples = torch::empty({static_cast<long>(total_size)}, options);

    float* data_ptr = samples.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (first.size() > 0) {
        cudaMemcpyAsync(
            data_ptr,
            first.data(),
            first.size_bytes(),
            cudaMemcpyHostToDevice,
            stream
        );
    }
    if (second.has_value() && second.value().size() > 0) { 
        cudaMemcpyAsync(
            data_ptr + first.size(),
            second.value().data(),
            second.value().size_bytes(),
            cudaMemcpyHostToDevice,
            stream
        );
    }
    if (padding_size > 0) {
        cudaMemsetAsync(
            data_ptr + total_size,  // Start position after all data
            0,                     // Value to set (0)
            padding_size * sizeof(float), // Size in bytes
            stream
        );
    }
    cudaStreamSynchronize(stream);

    //int padding = totalSize % mHopLength == 0 ? 0 : mHopLength - (totalSize % mHopLength);
    //if (padding > 0) {
    //    samples = torch::nn::functional::pad(
    //        samples, 
    //        torch::nn::functional::PadFuncOptions({0, padding}).mode(torch::kConstant).value(0));
    //}

    torch::Tensor stft = torch::stft(samples, 
        n_fft_, 
        hop_length_, 
        n_fft_, 
        window_, 
        true, // center
        "reflect", // pad_mode
        false, // normalized
        true, // onesided
        true // return_complex
    );

    auto n_frames = total_size + hop_length_ > n_fft_ / 2 ? (total_size + hop_length_ - n_fft_ / 2) / hop_length_ : 0;
    auto start = 0;
    if (skip.has_value()) {
        start = skip.value();
    }
    auto magnitudes = stft.slice(-1, start, n_frames).abs().pow(2);
    // auto magnitudes = stft.slice(-1, 0, stft.size(stft.dim() - 1) - 1).abs().pow(2);

    torch::Tensor mel_spec = torch::matmul(filters_, magnitudes);

    torch::Tensor log_spec = torch::clamp_min(mel_spec, 1e-10).log10();
    log_spec = torch::maximum(log_spec, log_spec.max() - 8.0);
    log_spec = (log_spec + 4.0) / 4.0;

    log_spec = log_spec.toType(torch::kFloat16).transpose(0, 1);

    std::cout << "LogMelSpectrogram: " << log_spec << std::endl;

    return log_spec;
}
*/