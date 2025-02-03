use std::path::Path;

use anyhow::Result;

use whisper_trtllm_rs::Whisper;

use hound::WavReader;

use std::sync::Arc;

fn main() -> Result<()> {
    let whisper = Arc::new(Whisper::load("/home/coder/whisper-trtllm-rs/models/whisper_turbo_int8")?);
    let audio = read_audio("/home/coder/whisper-trtllm-rs/models/assets/meeting-30s.wav", 16000)?;
    let result = whisper.transcribe(&audio)?;
    println!("Result: {:?}", result);

    /*
    let n = 2; // Number of threads
    let mut handles = Vec::new();
    for i in 0..n {
        let audio_clone = audio.clone();
        let whisper_clone = whisper.clone();
        handles.push(std::thread::spawn(move || {
            let start = std::time::Instant::now();
            let result = whisper_clone.transcribe(&audio_clone).unwrap();
            println!("Thread {} time: {:?}", i, start.elapsed());
            println!("Result: {:?}", result);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }*/

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