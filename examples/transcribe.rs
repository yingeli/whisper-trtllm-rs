use std::path::Path;
use anyhow::{anyhow, Result};
use whisper_trtllm_rs::{Whisper, Config};
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt, AsyncSeekExt, SeekFrom, BufReader};
use futures::{Stream, StreamExt};
use hound::{WavReader, WavSpec, SampleFormat};
use std::io::Read;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::sync::{Arc, Mutex};

const CHUNK_SIZE: usize = 16000 * 1;

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::default();
    let whisper = Arc::new(Whisper::load("/home/coder/whisper-trtllm-rs/models/whisper_turbo", config)?);
    
    whisper.log_mel();
    /*
    let audio_stream = wav_to_stream("/home/coder/whisper-trtllm-rs/audio/fr.wav", CHUNK_SIZE).await?;
    let start = std::time::Instant::now();
    //let (lang, segments) = whisper.transcribe(audio_stream, None).await?;
    //println!("Transcription took: {:?}", start.elapsed());
    //println!("Result: {:?}", result);

    let audio_stream = wav_to_stream("/home/coder/whisper-trtllm-rs/audio/oppo-en-us.wav", CHUNK_SIZE).await?;
    let start = std::time::Instant::now();
    let mut stream = Box::pin(whisper.transcribe(audio_stream, Some("<|0.00|>Hi,<|0.36|>")));
    while let Some(segment) = stream.next().await {
        match segment {
            Ok((lang, segment)) => println!("Language: {:?}, {:?}", lang, segment),
            Err(e) => eprintln!("Error: {:?}", e),
        }
    }
    println!("Transcription took: {:?}", start.elapsed());
    //println!("Lang: {:?}, Segments: {:?}", lang, segments.collect::<Vec<Segment>>());
    */

    Ok(())
}

struct WavStream {
    samples: Vec<f32>,
    position: usize,
    chunk_size: usize,
}

impl WavStream {
    pub fn new(samples: Vec<f32>, chunk_size: usize) -> Self {
        Self {
            samples,
            position: 0,
            chunk_size,
        }
    }
}

impl Stream for WavStream {
    type Item = Vec<f32>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        
        if this.position >= this.samples.len() {
            return Poll::Ready(None);
        }
        
        let end = (this.position + this.chunk_size).min(this.samples.len());
        let chunk = this.samples[this.position..end].to_vec();
        this.position = end;
        
        Poll::Ready(Some(chunk))
    }
}

// 从WAV文件加载音频并创建流
async fn wav_to_stream<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<impl Stream<Item = Vec<f32>>> {
    // 使用hound读取WAV文件
    let reader = WavReader::open(path.as_ref())?;
    let spec = reader.spec();
    
    // 检查采样率和通道数
    if spec.channels != 1 {
        return Err(anyhow!("Only mono WAV files are supported"));
    }
    
    // 将样本转换为f32
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.into_samples::<f32>().collect::<Result<Vec<f32>, _>>()?,
        SampleFormat::Int => {
            match spec.bits_per_sample {
                16 => reader.into_samples::<i16>()
                    .map(|s| s.map(|s| s as f32 / i16::MAX as f32))
                    .collect::<Result<Vec<f32>, _>>()?,
                24 | 32 => reader.into_samples::<i32>()
                    .map(|s| s.map(|s| s as f32 / i32::MAX as f32))
                    .collect::<Result<Vec<f32>, _>>()?,
                8 => reader.into_samples::<i8>()
                    .map(|s| s.map(|s| s as f32 / i8::MAX as f32))
                    .collect::<Result<Vec<f32>, _>>()?,
                b => return Err(anyhow!("Unsupported bits per sample: {}", b)),
            }
        },
    };
    
    // 如果采样率不是16kHz，需要进行重采样(简化起见，这里省略了重采样逻辑)
    if spec.sample_rate != 16000 {
        println!("Warning: Sample rate is {}Hz, not 16kHz. Resampling needed.", spec.sample_rate);
        // 这里应该添加重采样逻辑
    }
    
    // 创建音频流
    Ok(WavStream::new(samples, chunk_size))
}