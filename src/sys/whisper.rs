use cxx::UniquePtr;

use std::path::Path;
use std::sync::Once;
use anyhow::{anyhow, Result};

use super::{
    tensor, Tensor, 
};

static INIT: Once = Once::new();

#[cxx::bridge]
mod ffi {
    #[derive(Copy, Clone, Debug)]
    #[repr(i32)]
    enum BatchingType {
        kSTATIC = 0,
        kINFLIGHT = 1,
    }

    #[derive(Copy, Clone, Debug)]
    struct Config {
        batchingType: BatchingType,
    }

    /*
    struct TranscribeResult {
        sequences: Vec<VecString>,
        sequences_ids: Vec<VecUSize>,
        scores: Vec<f32>,
        no_speech_prob: f32,
    }
    */

    unsafe extern "C++" {
        include!("whisper-trtllm-rs/src/sys/whisper.h");
        
        type BatchingType;

        type Config;

        type Tensor = super::Tensor;

        //type TranscribeResult;

        type Whisper;

        /*
        fn transcribe(
            self: &Whisper,
            features: Tensor,
            prompts: &[i32],
        ) -> Result<>
        */

        fn init() -> bool;

        fn whisper(model_path: &str, config: Config) -> UniquePtr<Whisper>;
    }
}

pub enum BatchingType {
    Static,
    Inflight,
}

impl Default for BatchingType {
    fn default() -> Self {
        Self::Static
    }
}

impl BatchingType {
    fn to_ffi(&self) -> ffi::BatchingType {
        match self {
            Self::Static => ffi::BatchingType::kSTATIC,
            Self::Inflight => ffi::BatchingType::kINFLIGHT,
        }
    }
}

pub struct Config {
    pub batching_type: BatchingType,
}

impl Config {
    fn to_ffi(&self) -> ffi::Config {
        ffi::Config {
            batchingType: self.batching_type.to_ffi(),
        }
    }
}

pub struct Whisper {
    inner: UniquePtr<ffi::Whisper>,
}

impl Whisper {
    pub fn load<P: AsRef<Path>>(model_path: P, config: Config) -> Result<Self> {
        INIT.call_once(|| {
            ffi::init();
        });

        let model_path = model_path.as_ref();
        let path = model_path.to_str().ok_or_else(|| anyhow!("invalid path: {}", model_path.display()))?;
        let inner = ffi::whisper(path, config.to_ffi());

        Ok(Self { inner })
    }
}