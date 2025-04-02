#include "feature.h"

FeatureBuffer::FeatureBuffer(
    LogMelSpectrogram logMel,
    const int64_t chunkSize
) : mBuffer(torch::empty({0, logMel.n_mels()})),
    mLogMel(logMel),
    mChuckSize(chunkSize),
    mPrev(std::vector<float>()) {
    mNPrevHops = (mLogMel.nFFT() + mLogMel.hopLength() - 1) / mLogMel.hopLength();
}

void FeatureBuffer::append(
    const std::span<const float> samples
) {
    auto features = mLogMel.extract(mPrev, samples).transpose(0, 1).slice(0, mNPrevHops);
    
    mBuffer = torch::cat({mBuffer, features}, 0);

    auto n = mNPrevHops * mLogMel.hopLength();
    if (samples.size() >= n) {
        if mPrev.size() < n {
            mPrev.resize(n);
        }
        auto start_pos = samples.size() - n;
        std::copy(samples.begin() + start_pos, samples.end(), mPrev.begin());
    } else {
        if mPrev.size() < n {
            mPrev = std::vector(n, 0.0f);
        }
        auto start_pos = samples.size();
        std::copy(mPrev.begin() + start_pos, mPrev.end(), mPrev.begin());
        std::copy(samples.begin(), samples.end(), mPrev.begin() + n - start_pos);
    }
}

void FeatureBuffer::consume(
    const int64_t amt
) {
    mBuffer = mBuffer.slice(0, amt);
}