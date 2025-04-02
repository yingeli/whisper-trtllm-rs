#pragma once

#include "whisper-trtllm-rs/src/sys/mel.h"
#include "whisper-trtllm-rs/src/sys/features.h"

#include "rust/cxx.h"

#include <torch/torch.h>
#include <filesystem>
#include <span>
#include <memory>

class FeatureBuffer {
    public:
        FeatureBuffer(
            const LogMelSpectrogram logMel
        );

        FeatureBuffer(
            std::filesystem::path const& melFilterPath,
            torch::Device const& device = torch::kCUDA,
            const int nMels = N_MELS,
            const int nFFT = N_FFT,
            const int hopLength = HOP_LENGTH
        );

        size_t nMels() const;
        
        size_t nFFT() const;
        
        size_t hopLength() const;

        size_t len() const {
            return mBuffer.size(0);
        }

        bool isEmpty() const {
            return (mBuffer.size(0) == 0 && mPrev.size() <= mNOverlapFrames * mExtractor.hopLength());
        }

        torch::Tensor getFeatures(const size_t amt) const;

        void append(
            const std::span<const float> samples
        );

        void consume(const size_t amt);

        // rust ffi
        inline std::unique_ptr<Features> features(const size_t amt) const {
            return std::make_unique<Features>(getFeatures(amt));
        }

        // rust ffi
        inline void append(
            const rust::Slice<const float> samples
        ) {
            append(std::span<const float>(samples.data(), samples.size()));
        }


    private:
        LogMelSpectrogram mExtractor;

        torch::Tensor mBuffer;

        std::vector<float> mPrev;
        size_t mNOverlapFrames;
        size_t mNMaxOverlapFrames;        
};

// rust ffi
inline std::unique_ptr<FeatureBuffer> feature_buffer(
    const rust::Str mel_filter_path
) {
    auto path = std::filesystem::path(static_cast<std::string>(mel_filter_path));
    return std::make_unique<FeatureBuffer>(
        path
    );
}