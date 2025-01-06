use std::path::Path;
use anyhow::{anyhow, Result};
use super::sys::executor::{Executor, BatchingType};
use tokenizers::tokenizer::Tokenizer;
use mel_spec::mel::{log_mel_spectrogram, mel, norm_mel};
use mel_spec::stft::Spectrogram;
use ndarray::{s, stack, Array2, Axis};
use ndarray_npy::NpzReader;
use std::fs::File;
use super::spectrogram::LogMelSpectrogram;

const TOKENIZER_FILENAME: &str = "tokenizer.json";
const MEL_FILTERS_NAME: &str = "mel_80";

pub struct Whisper {
    executor: Executor,
    tokenizer: Tokenizer,
    spectrogram: LogMelSpectrogram,
}

impl Whisper {
    pub fn load<T: AsRef<Path>>(model_path: T) -> Result<Self> {
        let executor = Executor::load(&model_path, BatchingType::Static)?;
        let tokenizer = Tokenizer::from_file(model_path.as_ref().join(TOKENIZER_FILENAME))
            .map_err(|e| anyhow!(e))?;
        let spectrogram = LogMelSpectrogram::default();
        Ok(Self { executor, tokenizer, spectrogram })
    }

    pub fn transcribe(&self, audio: &[f32]) {
        let features = self.spectrogram.extract_features(audio);
        println!("features len: {:?}", features.len());
        for f in features.iter().take(10) {
            print!("{} ", f);
        }
        println!();
    }
}
