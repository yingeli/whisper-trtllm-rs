#include "kaldifeat/csrc/feature-fbank.h"

class Fbank {
    public:
        Fbank(
            const std::string& melFilterPath,
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
        
        bool compute_features() const {
            return (mBuffer.size(0) == 0 && mPrev.size() <= mNOverlapFrames * mExtractor.hopLength());
        }

    private:
        inner_ kaldifeat::Fbank;
}