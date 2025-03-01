use cxx::UniquePtr;

use core::prelude::v1;
use std::path::Path;
use std::sync::Once;
use anyhow::{anyhow, Result};

//use super::{
//    tensor, Tensor, 
//};

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
    struct TranscribeOptions {
        beamWidth: u32,
        topK: u32,
        topP: f32,
        temperature: f32,
    }

    struct TranscribeResult {
        tokens: Vec<u32>,
        avgLogProb: f32,
    }

    unsafe extern "C++" {
        include!("whisper-trtllm-rs/src/sys/whisper.h");
        
        //type Tensor = super::Tensor;

        type BatchingType;

        type Config;

        type Whisper;

        type TranscribeOptions;

        fn enqueue_detect_language_request(
            self: Pin<&mut Whisper>,
            first: &[f32],
            second: &[f32],
        ) -> Result<u64>;

        fn await_detect_language_response(
            self: Pin<&mut Whisper>,
            request_id: &u64,
        ) -> Result<u32>;

        fn enqueue_transcribe_request(
            self: Pin<&mut Whisper>,
            first: &[f32],
            second: &[f32],
            prompt: &[u32],
            option: &TranscribeOptions,
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

pub struct TranscribeOptions {
    pub beam_width: u32,
    pub top_k: u32,
    pub top_p: f32,
    pub temperature: f32,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            beam_width: 1,
            top_k: 0,
            top_p: 0.0,
            temperature: 1.0,
        }
    }
}

impl TranscribeOptions {
    fn to_ffi(&self) -> ffi::TranscribeOptions {
        ffi::TranscribeOptions {
            beamWidth: self.beam_width,
            topK: self.top_k,
            topP: self.top_p,
            temperature: self.temperature,
        }
    }
}

pub(crate) struct TranscribeResult {
    pub tokens: Vec<u32>,
    pub avg_logprob: f32,
}

impl TranscribeResult {
    fn from_ffi(result: ffi::TranscribeResult) -> Self {
        Self {
            tokens: result.tokens,
            avg_logprob: result.avgLogProb,
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

    pub fn enqueue_detect_language_request(&mut self, first: &[f32], second: &[f32]) -> Result<u64> {
        self.ptr.pin_mut().enqueue_detect_language_request(first, second)
            .map_err(|e| anyhow!("failed to enqueue transcribe request: {e}"))
    }

    pub fn await_detect_language_response(&mut self, request_id: &u64) -> Result<u32> {
        self.ptr.pin_mut().await_detect_language_response(request_id)
            .map_err(|e| anyhow!("failed to get transcribe response: {e}"))
    }

    pub fn enqueue_transcribe_request(&mut self, first: &[f32], second: &[f32], prompt: &[u32], options: &TranscribeOptions) -> Result<u64> {
        let ffi_options = options.to_ffi();
        self.ptr.pin_mut().enqueue_transcribe_request(first, second, prompt, &ffi_options)
            .map_err(|e| anyhow!("failed to enqueue transcribe request: {e}"))
    }

    pub fn await_transcribe_response(&mut self, request_id: &u64) -> Result<TranscribeResult> {
        let result = self.ptr.pin_mut().await_transcribe_response(request_id)
            .map_err(|e| anyhow!("failed to get transcribe response: {e}"))?;
        Ok(TranscribeResult::from_ffi(result))
    }

    pub fn is_response_ready(&self, request_id: &u64) -> Result<bool> {
        self.ptr.is_response_ready(request_id)
            .map_err(|e| anyhow!("failed to query if response is ready: {e}"))
    }
}