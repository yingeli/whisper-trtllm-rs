use std::ops::Deref;

use cxx::UniquePtr;

use anyhow::{anyhow, Result};

#[cxx::bridge]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("whisper-trtllm-rs/src/diarization/sys/kaldifeat.h");

        type Fbank;

        fn fbank() -> UniquePtr<Fbank>;

        fn compute_features(
            self: &Fbank,
            samples: &[f32],
            vtln_warp: f32,
        ) -> Result<UniquePtr<Features>>;
    }
}