use std::path::{Path, PathBuf};

use anyhow::Result;

use whisper_trt::Whisper;

fn main() -> Result<()> {
    let whisper = Whisper::load("models/whisper_turbo_int8")?;

    Ok(())
}