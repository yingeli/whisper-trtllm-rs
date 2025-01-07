use std::path::Path;
use anyhow::{anyhow, Result};
use super::sys;
use tokenizers::tokenizer::Tokenizer;

const TOKENIZER_FILENAME: &str = "tokenizer.json";



pub struct Whisper {
    inner: sys::Whisper,
    tokenizer: Tokenizer,
}

impl Whisper {
    pub fn load<T: AsRef<Path>>(model_path: T) -> Result<Self> {
        let config = sys::Config {
            batching_type: sys::BatchingType::Static,
        };
        let inner = sys::Whisper::load(&model_path, config)?;
        let tokenizer = Tokenizer::from_file(model_path.as_ref().join(TOKENIZER_FILENAME))
            .map_err(|e| anyhow!(e))?;
        Ok(Self { inner, tokenizer })
    }

    pub fn transcribe(&self, audio: &[f32]) {
    }
}
