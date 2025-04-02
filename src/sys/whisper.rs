use cxx::UniquePtr;

use std::path::Path;
use std::sync::Once;
use anyhow::{anyhow, Result};

use super::features::{self, Features};

pub use ffi::{Config, TranscribeOptions, TranscribeResult};

static INIT: Once = Once::new();

#[cxx::bridge]
mod ffi {
    /*
    #[derive(Copy, Clone, Debug)]
    #[repr(i32)]
    enum BatchingType {
        kSTATIC = 0,
        kINFLIGHT = 1,
    }
    */

    #[derive(Copy, Clone, Debug)]
    pub struct Config {
        pub max_beam_width: u32,
        // batching_type: BatchingType,
    }

    #[derive(Copy, Clone, Debug)]
    struct TranscribeOptions {
        beam_width: u32,
        top_k: u32,
        top_p: f32,
        temperature: f32,
    }

    #[derive(Clone, Debug)]
    struct TranscribeResult {
        is_final: bool,
        is_sequence_final: bool,
        tokens: Vec<u32>,
        avg_logprob: f32,
    }

    unsafe extern "C++" {
        type Features = super::features::ffi::Features;

        include!("whisper-trtllm-rs/src/sys/whisper.h");
        
        type Whisper;

        fn init() -> bool;

        fn whisper(model_path: &str, config: &Config) -> UniquePtr<Whisper>;

        fn enqueue_detect_language_request(
            self: Pin<&mut Whisper>,
            features: &Features,
        ) -> Result<u64>;

        fn await_detect_language_response(
            self: Pin<&mut Whisper>,
            request_id: &u64,
        ) -> Result<u32>;

        fn enqueue_transcribe_request(
            self: Pin<&mut Whisper>,
            features: &Features,
            prompt: &[u32],
            option: &TranscribeOptions,
            stop_on_timestamp: bool,
        ) -> Result<u64>;

        fn await_transcribe_response(
            self: Pin<&mut Whisper>,
            request_id: &u64,
        ) -> Result<TranscribeResult>;

        fn is_response_ready(
            self: &Whisper,
            request_id: &u64,
        ) -> Result<bool>;
    }
}

unsafe impl Send for ffi::Whisper {}
unsafe impl Sync for ffi::Whisper {}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_beam_width: 1,
            // batching_type: BatchingType::default().to_ffi(),
        }
    }
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            beam_width: 1,
            top_k: 0,
            top_p: 0.0,
            temperature: 0.0,
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
        let ptr = ffi::whisper(path, &config);

        Ok(Self { ptr })
    }

    pub fn enqueue_detect_language_request(&mut self, features: &Features) -> Result<u64> {
        self.ptr.pin_mut().enqueue_detect_language_request(features)
            .map_err(|e| anyhow!("failed to enqueue transcribe request: {e}"))
    }

    pub fn await_detect_language_response(&mut self, request_id: &u64) -> Result<u32> {
        self.ptr.pin_mut().await_detect_language_response(request_id)
            .map_err(|e| anyhow!("failed to get transcribe response: {e}"))
    }

    pub fn enqueue_transcribe_request(&mut self, 
        features: &Features,
        prompt: &[u32], 
        options: &TranscribeOptions,
        stop_on_timestamp: bool,
    ) -> Result<u64> {
        self.ptr.pin_mut().enqueue_transcribe_request(
            features, 
            prompt, 
            &options, 
            stop_on_timestamp,
        ).map_err(|e| anyhow!("failed to enqueue transcribe request: {e}"))
    }

    pub fn await_transcribe_response(&mut self, request_id: &u64) -> Result<TranscribeResult> {
        let result = self.ptr.pin_mut().await_transcribe_response(request_id)
            .map_err(|e| anyhow!("failed to get transcribe response: {e}"))?;
        Ok(result)
    }

    pub fn is_response_ready(&self, request_id: &u64) -> Result<bool> {
        self.ptr.is_response_ready(request_id)
            .map_err(|e| anyhow!("failed to query if response is ready: {e}"))
    }
}