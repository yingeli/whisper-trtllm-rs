use std::path::Path;
use anyhow::{anyhow, Result};
use super::sys::{self, TranscribeOptions};
use super::tokenizer::Tokenizer;
use tokio::sync::Mutex;
use super::audio::Audio;
use tokio::io::AsyncRead;
use tokio::time::{sleep, Duration};
use super::transcript::{Transcript, Segment};

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

    pub async fn detect_language<R: AsyncRead + Unpin>(&self, reader: R) -> Result<String> {
        let mut audio = Audio::new(reader);
        audio.fill_chunk().await?;
        let (first, second) = audio.chunk();

        let mut inner = self.inner.lock().await;
        let request_id = inner.enqueue_detect_language_request(first, second)?;
        drop(inner);

        loop {
            let inner = self.inner.lock().await;
            if inner.is_response_ready(&request_id)? {
                break;
            }
            drop(inner);
            sleep(std::time::Duration::from_millis(10)).await;
        }

        let mut inner = self.inner.lock().await;
        let token = inner.await_detect_language_response(&request_id)?;

        let lang = self.tokenizer.language(token)?;

        Ok(lang)
    }

    pub async fn transcribe<R: AsyncRead + Unpin>(&self, 
        reader: R, 
        language: Option<String>, 
        initial_prompt: Option<String>,
        options: &TranscribeOptions,
    ) -> Result<Transcript>
    where
        R: AsyncRead + Unpin,
    {
        let mut audio = Audio::new(reader);
        
        //let start = std::time::Instant::now();
        let (first, second) = audio.fill_chunk().await?;
        //println!("fill_chunk: {:?}", start.elapsed());
        
        let lang = match language {
            Some(lang) => self.tokenizer.language_token_id(&lang)?,
            None => self.detect_chunk_language(first, second).await?,
        };

        let mut segments = Vec::new();
        let mut prompt = None;
        if let Some(initial_prompt) = initial_prompt {
            prompt = Some(self.tokenizer.encode(&initial_prompt)?.get_ids().to_vec());
        }

        loop {
            //println!("first: {:?}, second {:?}", first.len(), second.len());

            let tokens = self.transcribe_chunk(&mut audio, lang, prompt, options).await?;
            //println!("Tokens: {:?}", tokens);
            //println!("Chunk: {:?}", self.tokenizer.decode(&tokens, false)?);

            let mut start = 0;
            let mut end = 0;
            let mut start_pos = None;
            let mut end_pos = 0;

            for (i, token) in tokens.iter().enumerate() {
                if let Some(millis) = self.tokenizer.timestamp_to_millis(*token) {
                    match start_pos {
                        Some(pos) => {
                            if i > pos {
                                end = millis;
                                end_pos = i;
                                
                                // New segment
                                if end_pos - pos > 0 {
                                    println!("[{:?} -> {:?}] {:?}", audio.offset() + start, audio.offset() + end, &tokens[pos..end_pos]);
                                    let text = self.tokenizer.decode(&tokens[pos..end_pos], false)?;
                                    let segment = Segment::new(audio.offset() + start, audio.offset() + end, text);
                                    segments.push(segment);
                                }

                                start_pos = None;
                            }
                        },
                        None => {
                            start = millis;
                            start_pos = Some(i + 1);
                        }
                    }
                }
            }

            if end_pos == 0 {
                end = audio.chunk_duration();
                end_pos = tokens.len() - 1;

                if tokens.len() > 1 {
                    // New segment
                    println!("[{:?} -> {:?}] {:?}", audio.offset() + start, audio.chunk_end(), &tokens[1..]);
                    let text = self.tokenizer.decode(&tokens[1..], false)?;
                    let segment = Segment::new(audio.offset() + start, audio.chunk_end(), text);
                    segments.push(segment);
                }
                start_pos = None;
            }

            if audio.is_end() {
                if let Some(pos) = start_pos {
                    if pos < tokens.len() {
                        end = audio.chunk_end();
                        // New segment
                        println!("[{:?} -> {:?}] {:?}", audio.offset() + start, audio.duration(), &tokens[pos..]);
                        let text = self.tokenizer.decode(&tokens[pos..], false)?;
                        let segment = Segment::new(audio.offset() + start, audio.duration(), text);
                        segments.push(segment);
                    }
                }
                break;
            }

            audio.consume(end);
            //println!("Consumed: {:?}", audio.offset());

            //(first, second) = audio.fill_chunk().await?;

            let s = if end_pos < 210 { 0 } else { end_pos - 210 };
            //println!("Prompt: {:?}", self.tokenizer.decode(&tokens[s..end_pos + 1], true)?);
            prompt = Some(tokens[s..end_pos + 1].to_vec());
        }

        let transcript = Transcript::new(self.tokenizer.language(lang)?, segments);

        Ok(transcript)
    }

    pub async fn 

    pub async fn detect_chunk_language(&self, first: &[f32], second: &[f32]) -> Result<u32> {
        let mut inner = self.inner.lock().await;
        let request_id = inner.enqueue_detect_language_request(first, second)?;
        drop(inner);

        loop {
            let inner = self.inner.lock().await;
            if inner.is_response_ready(&request_id)? {
                break;
            }
            drop(inner);
            sleep(Duration::from_millis(5)).await;
        }

        let mut inner = self.inner.lock().await;
        let token_id = inner.await_detect_language_response(&request_id)?;

        Ok(token_id)
    }

    async fn transcribe_chunk<R: AsyncRead + Unpin>(&self, 
        audio: &mut Audio<R>, 
        language: u32, 
        prompt: Option<Vec<u32>>,
        options: &TranscribeOptions,
    ) -> Result<Vec<u32>> {
        let mut input = vec![];
        
        if let Some(p) = prompt {
            input.push(self.tokenizer.start_of_prev());
            input.extend_from_slice(&p);
        }

        input.push(self.tokenizer.start_of_transcript());
        //input.push(language);
        //input.push(50259);
        input.push(50260);        
        input.push(self.tokenizer.transcribe());
        //prompt.push(self.tokenizer.no_timestamp());

        let (first, second) = audio.fill_chunk().await?;
        let mut inner = self.inner.lock().await;
        let request_id = inner.enqueue_transcribe_request(first, second, input.as_slice(), options, false)?;
        drop(inner);

        audio.fill().await?;
       
        let mut output = vec![];
        let mut max_avg_logprob = f32::MIN;

        loop {
            loop {
                let inner = self.inner.lock().await;
                if inner.is_response_ready(&request_id)? {
                    break;
                }
                drop(inner);
                //std::thread::sleep(std::time::Duration::from_millis(10));
                sleep(Duration::from_millis(5)).await;
            }

            let mut inner = self.inner.lock().await;
            let result = inner.await_transcribe_response(&request_id)?;
            drop(inner);

            if result.is_sequence_final {
                if result.avg_logprob > max_avg_logprob {
                    output = result.tokens[input.len()..].to_vec();
                    max_avg_logprob = result.avg_logprob;
                }
                if result.is_final {
                    break;
                }
            }
        }

        //let output = self.tokenizer.decode(result.tokens.as_slice(), false)?;

        //println!("Output: {:?}", self.tokenizer.decode(&result.tokens, false)?);
        //let output = result.tokens[input.len()..].to_vec();
        Ok(output)
    }
}
