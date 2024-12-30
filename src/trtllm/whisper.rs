use cxx::UniquePtr;

use std::path::Path;
use anyhow::{anyhow, Result};

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("whisper.h");

        type Whisper;

        fn load(model_path: &str) -> UniquePtr<Whisper>;
    }
}

pub struct Whisper {
    inner: UniquePtr<ffi::Whisper>,
}

impl Whisper {
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref();
        let path = model_path.to_str().ok_or_else(|| anyhow!("invalid path: {}", model_path.display()))?;
        let inner = ffi::load(path);

        Ok(Self { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load() {
        let model_path = "models/whisper_turbo_int8";
        let whisper = Whisper::load(model_path).unwrap();
    }
}