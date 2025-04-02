use super::sys::{self, Config, Features, TranscribeOptions, TranscribeResult};
use std::sync::{Mutex, RwLock};
use anyhow::{anyhow, Result};
use std::path::Path;
use std::future::Future;
use tokio::time::{sleep, Duration};

pub(crate) struct Model {
    inner: RwLock<sys::Whisper>,
}

impl Model {
    pub fn load<P: AsRef<Path>>(model_path: P, config: Config) -> Result<Self> {
        let inner = RwLock::new(sys::Whisper::load(&model_path, config)?);
        Ok(Self { inner })
    }

    pub fn detect_language<'a>(&'a self, features: &'a Features) -> impl Future<Output = Result<u32>> + 'a {
        let mut whisper = self.inner.write().unwrap();
        let result = whisper.enqueue_detect_language_request(features);

        async move {
            let request_id = result?;
            
            loop {
                let whisper = self.inner.read().unwrap();
                if whisper.is_response_ready(&request_id)? {
                    break;
                }
                drop(whisper);
                sleep(Duration::from_millis(5)).await;
            }
            
            let mut whisper = self.inner.write().unwrap();
            whisper.await_detect_language_response(&request_id)
        }
    }

    /*
    //pub fn transcribe<'a>(&'a self, features: Features, prompt: Option<&str>) -> impl Future<Output = Result<(String, impl Iterator<Item = Segment> + 'a)>> + 'a {
    pub async fn transcribe<'a>(&'a self, features: Features, input: &[u32]) -> Result<Vec<u32>> {
        /*
        let mut input = vec![];
        if let Some(p) = prompt {
            if !p.is_empty() {
                input.push(self.tokenizer.start_of_prev());
                let encoder = self.tokenizer.encode(p).unwrap();
                let tokens = encoder.get_ids();
                input.extend_from_slice(tokens);  
            }
        }
        input.push(self.tokenizer.start_of_transcript());
        // input.push(language);
        // input.push(50259);
        // input.push(self.tokenizer.transcribe());
        // input.push(self.tokenizer.no_timestamp());
        */

        let mut whisper = self.inner.write().unwrap();
        let request_id = whisper.enqueue_transcribe_request(
            &features, 
            &input, 
            &TranscribeOptions::default(),
            false,
        )?;
            
        loop {
            let whisper = self.inner.read().unwrap();
            if whisper.is_response_ready(&request_id)? {
                break;
            }
            drop(whisper);
            sleep(Duration::from_millis(5)).await;
        }
            
        let mut whisper = self.inner.write().unwrap();
        let result = whisper.await_transcribe_response(&request_id)?;

        let tokens = result.tokens;

        println!("Tokens: {:?}", tokens);

        let language = self.tokenizer.language(tokens[input.len()])?;

        let segments = SegmentIterator::new(tokens, &self.tokenizer, input.len() + 2);

        Ok((language, segments))
    }
    */

    pub async fn transcribe_segment<'a>(&'a self, 
        features: Features, 
        input: &[u32],
    ) -> Result<Vec<u32>> {
        let mut whisper = self.inner.write().unwrap();
        let request_id = whisper.enqueue_transcribe_request(
            &features, 
            &input, 
            &TranscribeOptions::default(),
            true, // stop_on_timestamp
        )?;
        drop(whisper);
            
        loop {
            let whisper = self.inner.read().unwrap();
            if whisper.is_response_ready(&request_id)? {
                break;
            }
            drop(whisper);
            sleep(Duration::from_millis(5)).await;
        }
            
        let mut whisper = self.inner.write().unwrap();
        let result = whisper.await_transcribe_response(&request_id)?;

        Ok(result.tokens)
    }
}