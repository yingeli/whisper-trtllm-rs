use tokenizers::{self, Decoder, Encoding};
use anyhow::{anyhow, Result};
use std::path::Path;
use scan_fmt::scan_fmt; 

pub(crate) struct Tokenizer {
    inner: tokenizers::Tokenizer,
    no_timestamp: u32,
    start_of_prev: u32,
    start_of_transcript: u32,
    transcribe: u32,
}

impl Tokenizer {
    pub fn from_file<T: AsRef<std::path::Path>>(path: T) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|err| anyhow!("failed to load a tokenizer: {err}"))?;
        
        let no_timestamp = inner.token_to_id("<|notimestamps|>")
            .ok_or_else(|| anyhow!("failed to get the no_timestamp token"))?;

        let start_of_prev = inner.token_to_id("<|startofprev|>")
            .ok_or_else(|| anyhow!("failed to get the start_of_preview token"))?;

        let start_of_transcript = inner.token_to_id("<|startoftranscript|>")
            .ok_or_else(|| anyhow!("failed to get the start_of_transcript token"))?;

        let transcribe = inner.token_to_id("<|transcribe|>")
            .ok_or_else(|| anyhow!("failed to get the transcribe token"))?;

        Ok(Self { 
            inner,
            no_timestamp,
            start_of_prev,
            start_of_transcript,
            transcribe,
        })
    }

    pub fn no_timestamp(&self) -> u32 {
        self.no_timestamp
    }

    pub fn start_of_prev(&self) -> u32 {
        self.start_of_prev
    }

    pub fn start_of_transcript(&self) -> u32 {
        self.start_of_transcript
    }

    pub fn transcribe(&self) -> u32 {
        self.transcribe
    }

    pub fn is_timestamp(&self, id: u32) -> bool {
        id > self.no_timestamp()
    }

    pub fn token_to_id(&self, token: &str) -> Result<u32> {
        self.inner.token_to_id(token)
            .ok_or_else(|| anyhow!("failed to find the token"))
    }

    pub fn language_token_id(&self, lang: &str) -> Result<u32> {
        self.token_to_id(format!("<|{}|>", lang).as_str())
    }

    pub fn timestamp_token_id(&self, timestamp: f32) -> Result<u32> {
        self.token_to_id(format!("<|{:.2}|>", timestamp).as_str())
    }

    //pub fn decode(&self, ids: &[usize]) -> Result<String> {
    //    let ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
    //    self.inner.decode(ids.as_slice(), false)
    //        .map_err(|err| anyhow!("failed to decode: {err}"))
    //}

    /*
    pub fn decode(&self, ids: &[usize]) -> Result<String> {
        let mut text = String::new();
        for id in ids {
            let token = self.inner.id_to_token(*id as u32)
                .ok_or_else(|| anyhow!("failed to decode the given input"))?;
            println!("id: {}, token: {}", id, token);
            text.push_str(&token);
        }
        Ok(text)
    }
    */

    pub fn encode(&self, text: &str) -> Result<Encoding> {
        self.inner.encode(text, false)
            .map_err(|err| anyhow!("failed to encode: {err}"))
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(tokens, skip_special_tokens)
            .map_err(|err| anyhow!("failed to decode: {err}"))
    }

    pub fn id_to_timestamp(&self, id: usize) -> Result<f32> {
        let token = self.inner.id_to_token(id as u32)
            .ok_or_else(|| anyhow!("failed to decode the given input"))?;
        let timestamp = scan_fmt!(&token, "<{}>", f32)?;
        Ok(timestamp)
    }

    pub fn extract_timestamp(&self, token: &str) -> Result<f32> {
        let timestamp = scan_fmt!(&token, "<|{}|>", f32)?;
        Ok(timestamp)
    }
}