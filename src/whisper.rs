use std::path::Path;
use anyhow::{anyhow, Result};
use futures::executor;
use crate::sys::TranscribeResult;
use super::tokenizer::Tokenizer;
use super::sys::{self};
use super::model::Model;
use tokio::sync::Mutex;
//use super::audio::Audio;
use tokio::io::AsyncRead;
use tokio::time::{sleep, Duration};
//use super::transcript::{Transcript, Segment};
use futures::stream::{Stream, StreamExt};
use super::features::Audio;
use super::sys::LogMelSpectrogram;
//use super::transcript::{Segment};
use async_stream::stream;

pub use sys::Config;

const MEL_FILTER_FILENAME: &str = "mel_filters.npz";
const TOKENIZER_FILENAME: &str = "tokenizer.json";

pub struct Whisper {
    extractor: LogMelSpectrogram,
    tokenizer: Tokenizer,
    model: Model,
}

impl Whisper {
    const CHUNK_SIZE: usize = 3000;
    const N_MEL: usize = 128;
    const N_FFT: usize = 400;
    const HOP_LENGTH: usize = 160;

    const DELTA: usize = 100;

    pub fn load<T: AsRef<Path>>(model_path: T, config: Config) -> Result<Self> {
        let extractor = LogMelSpectrogram::open(
            model_path.as_ref().join(MEL_FILTER_FILENAME),
            Self::N_MEL,
            Self::N_FFT,
            Self::HOP_LENGTH,
        )?;

        let tokenizer = Tokenizer::from_file(model_path.as_ref().join(TOKENIZER_FILENAME))?;

        let model = Model::load(&model_path, config)?;

        Ok(Self { 
            extractor,
            tokenizer,
            model,
        })
    }

    pub async fn detect_language<S>(&self, stream: S) -> Result<String> 
    where 
        S: Stream<Item = Vec<f32>> + Unpin,
    {
        let mut audio = Audio::new(&self.extractor, stream);

        let features = audio.features(Self::CHUNK_SIZE).await?
            .ok_or_else(|| anyhow!("No audio data"))?;

        let language_token = self.model.detect_language(&features).await?;

        let language = self.tokenizer.language(language_token)?;
        Ok(language)
    }

    /*
    pub fn transcribe<'a, S>(&'a self, 
        stream: S, 
        initial_prompt: Option<&'a str>
    ) -> impl Stream<Item = Result<(String, Segment)>> + 'a
    where 
        S: Stream<Item = Vec<f32>> + Unpin + 'a,
    {
        let stream = stream! {
            let mut audio = Audio::new(&self.extractor, stream);

            while let Some(chunk) = audio.chunk(Self::CHUNK_SIZE).await? {
                let mut input = vec![];
                /*
                if let Some(p) = prompt {
                    if !p.is_empty() {
                        input.push(self.tokenizer.start_of_prev());
                        let encoder = self.tokenizer.encode(p).unwrap();
                        let tokens = encoder.get_ids();
                        input.extend_from_slice(tokens);  
                    }
                }
                */
                input.push(self.tokenizer.start_of_transcript());

                let tokens = self.model.transcribe_segment(chunk, &input).await?;

                let language = self.tokenizer.language(tokens[input.len()])?;

                let start = self.tokenizer.timestamp_to_millis(tokens[input.len() + 2]).unwrap_or(0);
                let end;
                let text;
                if let Some(millis) = self.tokenizer.timestamp_to_millis(tokens[tokens.len() - 1]) {
                    end = millis;
                    text = self.tokenizer.decode(&tokens[input.len() + 3..tokens.len() - 1], false)?;
                } else {
                    end = 30000;
                    text = self.tokenizer.decode(&tokens[input.len() + 3..], false)?;
                }
                audio.consume_millis(end);
                
                let segment = Segment::new(start, end, text);
                yield Ok((language, segment));
            }
        };

        stream
    }
    */

    pub fn log_mel(&self) {
        let first = vec![0.0; 30 * 16000 + 1];
        let second = vec![0.0; 30 * 16000];
        let features = self.extractor.extract(&first, &second).unwrap().slice_to_end(2);
        println!("features: {:?}", features.len());

        let features = self.extractor.extract_final(&first, &second).unwrap().slice_to_end(2);
        println!("features: {:?}", features.len());
    }
}