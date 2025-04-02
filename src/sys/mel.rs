use super::features::{self, Features};

use cxx::UniquePtr;

use anyhow::{anyhow, Result};
use std::path::Path;

#[cxx::bridge]
pub(crate) mod ffi {
    unsafe extern "C++" {
        type Features = super::features::ffi::Features;

        include!("whisper-trtllm-rs/src/sys/mel.h");

        type LogMelSpectrogram;

        fn log_mel_spectrogram(
            mel_filter_path: &str,
            n_mels: usize,
            n_fft: usize,
            hop_length: usize,
        ) -> Result<UniquePtr<LogMelSpectrogram>>;

        fn extract(
            self: &LogMelSpectrogram,
            first: &[f32],
            second: &[f32],
            skip: usize,
            padding: bool,
        ) -> Result<UniquePtr<Features>>;

        fn empty(
            self: &LogMelSpectrogram
        ) -> UniquePtr<Features>;

        fn n_fft(
            self: &LogMelSpectrogram
        ) -> usize;

        fn hop_length(
            self: &LogMelSpectrogram
        ) -> usize;
    }
}

pub(crate) struct LogMelSpectrogram {
    ptr: UniquePtr<ffi::LogMelSpectrogram>,
    required_overlap_frames: usize,
}

impl LogMelSpectrogram {
    pub fn open(
        mel_filter_path: impl AsRef<Path>,
        n_mels: usize,
        n_fft: usize,
        hop_length: usize,
    ) -> Result<Self> {
        let required_overlap_frames = (n_fft / 2 + hop_length - 1) / hop_length;
        let path = mel_filter_path.as_ref().to_str()
            .ok_or_else(|| anyhow!("failed to convert mel filter path to str"))?;
        let ptr = ffi::log_mel_spectrogram(
            path,
            n_mels,
            n_fft,
            hop_length,
        ).map_err(|e| anyhow!("failed to create log mel spectrogram: {}", e))?;

        Ok(Self { 
            ptr,
            required_overlap_frames,
        })
    }

    pub fn n_fft(&self) -> usize {
        self.ptr.n_fft()
    }

    pub fn hop_length(&self) -> usize {
        self.ptr.hop_length()
    }

    pub fn required_overlap_frames(&self) -> usize {
        self.required_overlap_frames
    }

    pub fn extract(&self, first: &[f32], second: &[f32], skip: usize) -> Result<Features> {
        let ptr = self.ptr.extract(first, second, skip, false)
            .map_err(|e| anyhow!("failed to extract log mel spectrogram: {}", e))?;

        Ok(ptr.into())
    }

    pub fn extract_final(&self, first: &[f32], second: &[f32], skip: usize) -> Result<Features> {
        let ptr = self.ptr.extract(first, second, skip, true)
            .map_err(|e| anyhow!("failed to extract log mel spectrogram: {}", e))?;

        Ok(ptr.into())
    }

    pub fn empty(&self) -> Features {
        self.ptr.empty().into()
    }
}

unsafe impl Send for LogMelSpectrogram {}
unsafe impl Sync for LogMelSpectrogram {}