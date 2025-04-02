use cxx::UniquePtr;

use std::path::Path;
use anyhow::{anyhow, Result};

use super::features::{self, Features};

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        type Features = super::features::ffi::Features;
        
        include!("whisper-trtllm-rs/src/sys/buffer.h");

        type FeatureBuffer;

        fn feature_buffer(
            mel_filter_path: &str,
        ) -> Result<UniquePtr<FeatureBuffer>>;

        fn len(self: &FeatureBuffer) -> usize;

        #[rust_name = "is_empty"]
        fn isEmpty(self: &FeatureBuffer) -> bool;

        fn features(
            self: &FeatureBuffer, 
            amt: usize
        ) -> UniquePtr<Features>;

        fn append(
            self: Pin<&mut FeatureBuffer>,
            samples: &[f32],
        ) -> Result<()>;

        fn consume(
            self: Pin<&mut FeatureBuffer>,
            amt: usize,
        ) -> Result<()>;
    }
}

pub(crate) struct FeatureBuffer {
    ptr: UniquePtr<ffi::FeatureBuffer>,
}

impl FeatureBuffer {
    pub fn open<P: AsRef<Path>>(mel_filter_path: P) -> Result<Self> {
        let mel_filter_path = mel_filter_path.as_ref();
        let path = mel_filter_path.to_str().ok_or_else(|| anyhow!("invalid path: {}", mel_filter_path.display()))?;
        let ptr = ffi::feature_buffer(path)
            .map_err(|e| anyhow!("failed to create feature buffer: {}", e))?;

        Ok(Self { ptr })
    }

    pub fn len(&self) -> usize {
        self.ptr.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ptr.is_empty()
    }

    pub fn features(&self, amt: usize) -> Features {
        Features::new(self.ptr.features(amt))
    }

    pub fn append(&mut self, samples: &[f32]) -> Result<()> {
        self.ptr.pin_mut().append(samples).map_err(|e| anyhow!("failed to append samples: {}", e))
    }

    pub fn consume(&mut self, amt: usize) -> Result<()> {
        self.ptr.pin_mut().consume(amt).map_err(|e| anyhow!("failed to consume features: {}", e))
    }
}

unsafe impl Send for FeatureBuffer {}
unsafe impl Sync for FeatureBuffer {}

#[cfg(test)]
mod tests {
    use super::{Features, FeatureBuffer};

    #[test]
    fn test_features_buffer() {
        let mut buffer = FeatureBuffer::open("models/whisper_turbo/mel_filters.npz").unwrap();
        assert_eq!(buffer.len(), 0); 
    }
}