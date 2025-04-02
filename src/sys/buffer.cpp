#include "whisper-trtllm-rs/src/sys/buffer.h"

#include <algorithm>

FeatureBuffer::FeatureBuffer(
    const LogMelSpectrogram logMel
) : mBuffer(torch::empty({0, logMel.nMels()})),
    mPrev(std::vector<float>()),
    mExtractor(logMel),
    mNOverlapFrames(0)
{
    mNMaxOverlapFrames = (mExtractor.nFFT() / 2 + mExtractor.hopLength() - 1) / mExtractor.hopLength();
    mPrev.reserve(mNMaxOverlapFrames * mExtractor.hopLength() + mExtractor.nFFT() / 2);
}

FeatureBuffer::FeatureBuffer(
    std::filesystem::path const& melFilterPath,
    torch::Device const& device,
    const int nMels,
    const int nFFT,
    const int hopLength
) : FeatureBuffer(
    LogMelSpectrogram(melFilterPath, device, nMels, nFFT, hopLength)
) {
}

size_t FeatureBuffer::nMels() const {
    return mExtractor.nMels();
}

size_t FeatureBuffer::nFFT() const {
    return mExtractor.nFFT();
}

size_t FeatureBuffer::hopLength() const {
    return mExtractor.hopLength();
}

torch::Tensor FeatureBuffer::getFeatures(const size_t amt) const {
    if (mBuffer.size(0) >= amt) {
        return mBuffer.slice(0, 0, amt);
    } else {
        auto padTo = (amt - mBuffer.size(0)) * hopLength();
        auto features = mExtractor.extract(mPrev, std::nullopt, padTo).transpose(0, 1);
        if (mPrev.size() > 0) {
            features = features.slice(0, mNOverlapFrames);
        }
        return torch::cat({mBuffer, features}, 0);
    }
}

void FeatureBuffer::append(
    const std::span<const float> samples
) {
    auto nSamples = mPrev.size() + samples.size();
    size_t nFrames = 0;
    if (nSamples + hopLength() > nFFT() / 2) {
        nFrames = (nSamples + hopLength() - nFFT() / 2) / hopLength();
    }
    auto nTail = nSamples - nFrames * hopLength();

    auto features = mExtractor.extract(mPrev, samples).transpose(0, 1).slice(0, mNOverlapFrames, nFrames);
    mBuffer = torch::cat({mBuffer, features}, 0);

    mNOverlapFrames = std::min(nFrames, mNMaxOverlapFrames);
    auto n = mNOverlapFrames * hopLength() + nTail;
    if (n <= samples.size()) {
        mPrev.resize(n);
        std::copy(samples.end() - n, samples.end(), mPrev.begin());
    } else {
        auto m = n - samples.size();
        if (m <= mPrev.size()) {
            std::copy(mPrev.end() - m, mPrev.end(), mPrev.begin());
            mPrev.resize(n);
            std::copy(samples.begin(), samples.end(), mPrev.begin() + m);
        } else {
            mPrev.insert(mPrev.end(), samples.begin(), samples.end());
        }
    }
}

void FeatureBuffer::consume(
    const size_t amt
) {
    mBuffer = mBuffer.slice(0, amt);
}