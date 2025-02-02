use cxx::UniquePtr;

use std::path::Path;
use std::sync::Once;
use anyhow::{anyhow, Result};

use super::{
    tensor, Tensor, 
};

pub(crate) use self::ffi::TranscribeResult;

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

    struct TranscribeResult {
        tokens: Vec<u32>,
    }

    unsafe extern "C++" {
        include!("whisper-trtllm-rs/src/sys/whisper.h");
        
        type BatchingType;

        type Config;

        type Tensor = super::Tensor;

        type Whisper;

        fn enqueue_transcribe_request(
            self: Pin<&mut Whisper>,
            audio: &[f32],
            prompt: &[u32],
        ) -> Result<u64>;

        fn await_transcribe_response(
            self: Pin<&mut Whisper>,
            request_id: &u64,
        ) -> Result<TranscribeResult>;

        fn is_response_ready(
            self: &Whisper,
            request_id: &u64,
        ) -> Result<bool>;

        fn init() -> bool;

        fn whisper(model_path: &str, config: Config) -> UniquePtr<Whisper>;
    }
}

unsafe impl Send for ffi::Whisper {}
unsafe impl Sync for ffi::Whisper {}

pub enum BatchingType {
    Static,
    Inflight,
}

impl Default for BatchingType {
    fn default() -> Self {
        Self::Inflight
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
    ptr: UniquePtr<ffi::Whisper>,
}

impl Whisper {
    pub fn load<P: AsRef<Path>>(model_path: P, config: Config) -> Result<Self> {
        INIT.call_once(|| {
            ffi::init();
        });

        let model_path = model_path.as_ref();
        let path = model_path.to_str().ok_or_else(|| anyhow!("invalid path: {}", model_path.display()))?;
        let ptr = ffi::whisper(path, config.to_ffi());

        Ok(Self { ptr })
    }

    pub fn enqueue_transcribe_request(&mut self, audio: &[f32], prompt: &[u32]) -> Result<u64> {
        self.ptr.pin_mut().enqueue_transcribe_request(audio, prompt)
            .map_err(|e| anyhow!("failed to enqueue transcribe request: {e}"))
    }

    pub fn await_transcribe_response(&mut self, request_id: &u64) -> Result<TranscribeResult> {
        self.ptr.pin_mut().await_transcribe_response(request_id)
            .map_err(|e| anyhow!("failed to get transcribe response: {e}"))
    }

    pub fn is_response_ready(&self, request_id: &u64) -> Result<bool> {
        self.ptr.is_response_ready(request_id)
            .map_err(|e| anyhow!("failed to query if response is ready: {e}"))
    }
}