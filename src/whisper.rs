use std::path::Path;
use anyhow::{anyhow, Result};
use super::sys;
use super::tokenizer::Tokenizer;
use std::sync::Mutex;

const TOKENIZER_FILENAME: &str = "tokenizer.json";

pub struct Whisper {
    inner: Mutex<sys::Whisper>,
    tokenizer: Tokenizer,
}

impl Whisper {
    pub fn load<T: AsRef<Path>>(model_path: T) -> Result<Self> {
        let config = sys::Config {
            batching_type: sys::BatchingType::Inflight,
        };
        let inner = Mutex::new(sys::Whisper::load(&model_path, config)?);
        let tokenizer = Tokenizer::from_file(model_path.as_ref().join(TOKENIZER_FILENAME))?;
        Ok(Self { inner, tokenizer })
    }

    pub fn detect_language(&self, audio: &[f32]) -> Result<i32> {
        let mut inner = self.inner.lock().unwrap();
        let request_id = inner.enqueue_detect_language_request(audio)?;
        drop(inner);

        loop {
            let inner = self.inner.lock().unwrap();
            if inner.is_response_ready(&request_id)? {
                break;
            }
            drop(inner);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let mut inner = self.inner.lock().unwrap();
        let token_id = inner.await_detect_language_response(&request_id)?;

        //let output = self.tokenizer.decode(result.tokens.as_slice(), false)?;

        Ok(token_id)
    }

    pub fn transcribe(&self, audio: &[f32]) -> Result<String> {
        let lang_token = self.tokenizer.language_token_id("th")?;

        // Transcribe.
        let mut prompt = vec![
            self.tokenizer.start_of_prev(),
            //self.tokenizer.timestamp_token_id(0.0)?,
        ];
        prompt.extend_from_slice(self.tokenizer.encode("Hi,")?.get_ids());
        //prompt.push(self.tokenizer.timestamp_token_id(0.5)?);

        prompt.push(self.tokenizer.start_of_transcript());
        prompt.push(lang_token);
        prompt.push(self.tokenizer.transcribe());
        //prompt.push(self.tokenizer.no_timestamp());

        let mut inner = self.inner.lock().unwrap();
        let request_id = inner.enqueue_transcribe_request(audio, prompt.as_slice())?;
        drop(inner);

        loop {
            let inner = self.inner.lock().unwrap();
            if inner.is_response_ready(&request_id)? {
                break;
            }
            drop(inner);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let mut inner = self.inner.lock().unwrap();
        let result = inner.await_transcribe_response(&request_id)?;

        let output = self.tokenizer.decode(result.tokens.as_slice(), false)?;

        Ok(output)
    }
}
