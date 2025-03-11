use std::path::Path;
use anyhow::Result;
use whisper_trtllm_rs::{Whisper, TranscribeOptions};
use hound::WavReader;
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt, AsyncSeekExt, SeekFrom, BufReader};

#[tokio::main]
async fn main() -> Result<()> {
    let whisper = Arc::new(Whisper::load("/home/coder/whisper-trtllm-rs/models/whisper_turbo")?);
    //let audio = read_audio("/home/coder/whisper-trtllm-rs/models/assets/meeting-30s.wav", 16000)?;
    let mut audio = File::open("/home/coder/whisper-trtllm-rs/models/assets/oppo-zh-cn.wav").await?;
    audio.seek(SeekFrom::Start(44)).await?;
    let reader = BufReader::new(audio);

    let mut options = TranscribeOptions::default();
    options.beam_width = 1;
    //options.top_k = 0;
    //options.top_p = 1.0;
    //options.temperature = 1.0;

    let start = std::time::Instant::now();
    let transcript = whisper.transcribe(reader, None, None, &options).await?;
    //println!("Transcript: {:?}", transcript);
    //println!("Time elapsed: {:?}", start.elapsed());

    let mut audio = File::open("/home/coder/whisper-trtllm-rs/models/assets/yunxi-en-zh.wav").await?;
    audio.seek(SeekFrom::Start(44)).await?;
    let reader = BufReader::new(audio);

    let mut options = TranscribeOptions::default();
    options.beam_width = 1;
    //options.top_k = 0;
    //options.top_p = 1.0;
    //options.temperature = 1.0;

    let start = std::time::Instant::now();
    let transcript = whisper.transcribe(reader, None, None, &options).await?;
    println!("Transcript: {:?}", transcript);
    println!("Time elapsed: {:?}", start.elapsed());

    /*
    let n = 1; // Number of threads
    let mut handles = Vec::new();
    let start = std::time::Instant::now();
    for i in 0..n {
        //let audio_clone = audio.clone();
        let whisper_clone = whisper.clone();
        handles.push(tokio::spawn( async move {
            for i in 0..20 {
                let mut audio = File::open("/home/coder/whisper-trtllm-rs/models/assets/oppo-en-us.wav").await.unwrap();
                audio.seek(SeekFrom::Start(44)).await.unwrap();
                let reader = BufReader::new(audio);
                let mut options = TranscribeOptions::default();
                options.beam_width = 1;
                //options.top_k = 0;
                //options.top_p = 1.0;
                //options.temperature = 1.0;
                let start_time = std::time::Instant::now();
                //let transcript = whisper_clone.transcribe(reader, None, Some("Hi,".to_string()), &options).await.unwrap();
                let transcript = whisper_clone.transcribe(reader, None, None, &options).await.unwrap();
                // println!("Transcript: {:?}", transcript);
                println!("Time elapsed: {:?}", start_time.elapsed());
                //break;
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }
    println!("Time: {:?}", start.elapsed());
    println!("RPS: {:?}", n as f32 * 20.0 / start.elapsed().as_millis() as f32 * 1000 as f32);
    */

    Ok(())
}

fn read_audio<T: AsRef<Path>>(path: T, sample_rate: usize) -> Result<Vec<f32>> {
    // Should use a better resampling algorithm.
    fn resample(samples: Vec<f32>, src_rate: usize, target_rate: usize) -> Vec<f32> {
        samples
            .into_iter()
            .step_by(src_rate / target_rate)
            .collect()
    }

    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let max = 2_i32.pow((spec.bits_per_sample - 1) as u32) as f32;
    let samples = reader
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 / max)
        .collect::<Vec<f32>>();

    if spec.channels == 1 {
        return Ok(resample(samples, spec.sample_rate as usize, sample_rate));
    }

    let mut mono = vec![];
    for chunk in samples.chunks(2) {
        if chunk.len() == 2 {
            mono.push((chunk[0] + chunk[1]) / 2.);
        }
    }

    Ok(resample(mono, spec.sample_rate as usize, sample_rate))
}
