#pragma once

#include "whisper-trtllm-rs/src/sys/features.h"

#include "rust/cxx.h"

#include <torch/torch.h>
#include <filesystem>
#include <span>

const size_t N_MELS = 128;
const size_t N_FFT = 400;
const size_t HOP_LENGTH = 160;

class LogMelSpectrogram {
    public:
        LogMelSpectrogram(
            const std::filesystem::path& mel_filter_path,
            const size_t n_mels = N_MELS,
            const size_t n_fft = N_FFT,
            const size_t hop_length = HOP_LENGTH,
            const torch::Device& device = torch::kCUDA
        );

        torch::Tensor extract(
            const std::span<const float> first, 
            const std::optional<std::span<const float>> second,
            const std::optional<size_t> skip = std::nullopt,
            const std::optional<bool> padding = false
        ) const;

        inline std::unique_ptr<Features> extract(
            const rust::Slice<const float> first, 
            const rust::Slice<const float> second,
            const size_t skip = 0,
            const bool padding = false
        ) const {
            return std::make_unique<Features>(
                extract(
                    std::span<const float>(first.data(), first.size()), 
                    std::span<const float>(second.data(), second.size()),
                    skip,
                    padding
                )
            );
        }

        inline std::unique_ptr<Features> empty() const {
            // auto tensor = torch::empty({n_mels(), 0}, torch::TensorOptions().dtype(torch::kFloat32).device(filters_.device()));
            auto tensor = torch::empty({0, n_mels()}, torch::TensorOptions().dtype(torch::kFloat16).device(filters_.device()));
            return std::make_unique<Features>(tensor);
        }

        size_t n_mels() const {
            return filters_.size(0);
        }

        size_t n_fft() const {
            return n_fft_;
        }

        size_t hop_length() const {
            return hop_length_;
        }

    private:
        torch::Tensor filters_;
        torch::Tensor window_;
        size_t n_fft_;
        size_t hop_length_;
};

inline std::unique_ptr<LogMelSpectrogram> log_mel_spectrogram(
    const rust::Str mel_filter_path,
    const size_t n_mels = N_MELS,
    const size_t n_fft = N_FFT,
    const size_t hop_length = HOP_LENGTH
) {
    auto path = std::filesystem::path(static_cast<std::string>(mel_filter_path));
    return std::make_unique<LogMelSpectrogram>(path, n_mels, n_fft, hop_length);
}