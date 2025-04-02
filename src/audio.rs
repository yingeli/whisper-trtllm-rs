use tokio::io::{AsyncRead, AsyncReadExt};
use anyhow::{anyhow, Ok, Result};
use std::collections::VecDeque;
use super::sys::{Features, LogMelSpectrogram};
use futures::stream::{Stream, StreamExt};

pub(crate) struct Audio<'a, S> {
    extractor: &'a LogMelSpectrogram,
    stream: S,
    prev: Vec<f32>,
    overlap_frames: usize,
    features: Features,
    offset: usize,
    eof: bool,
}

impl<'a, S> Audio<'a, S> 
where 
    S: Stream<Item = Vec<f32>> + Unpin,
{  
    const MILLIS_PER_FRAME: usize = 10;

    pub fn new(extractor: &'a LogMelSpectrogram, stream: S) -> Self {
        let features = extractor.empty();
        Self {
            extractor,
            stream,
            prev: vec![],
            overlap_frames: 0,
            features,
            offset: 0,
            eof: false,
        }
    }

    pub fn len(&self) -> usize {
        self.offset + self.features.len()
    }

    pub fn eof(&self) -> bool {
        self.eof
    }

    pub fn offset(&self) -> usize {
        self.offset + Self::MILLIS_PER_FRAME
    }

    fn overlap_samples(&self) -> usize {
        self.overlap_frames * self.samples_per_frame()
    }

    fn samples_per_frame(&self) -> usize {
        self.extractor.hop_length()
    }

    fn required_overlap_frames(&self) -> usize {
        self.extractor.required_overlap_frames()
    }

    // pub fn duration(&self) -> usize {
    //    self.offset + self.buffer.len() / Self::SAMPLES_PER_MILLIS
    // }

    pub async fn chunk(&mut self, chunk_size: usize) -> Result<Option<Features>> {
        loop {
            if self.features.len() >= chunk_size {
                let chuck = self.features.slice(0, chunk_size);
                return Ok(Some(chuck));
            }

            if self.eof {
                
                && self.features.len() == 0 {
                return Ok(None);
            }

            if !self.fill().await? {
                let n = chunk_size - self.features.len();
                let features = self.extractor.extract(&self.prev, &[], self.overlap_frames, n).unwrap();
                self.eof = true;
                return Ok(Some(self.features.join(&features)));            
            };
        }
    }

    pub async fn fill(&mut self) -> Result<bool> {
        let Some(mut samples) = self.stream.next().await else {
            return Ok(false);
        };

        let features = self.extractor.extract(
            &self.prev,
            &samples,
            self.overlap_frames,
            0,
        )?;

        let mut n = self.prev.len() + samples.len() - features.len() * self.samples_per_frame();
        if self.overlap_frames < self.required_overlap_frames() {
            n -= self.overlap_samples();

            self.overlap_frames = std::cmp::min(
                self.overlap_frames + features.len(),
                self.required_overlap_frames(),
            );

            n += self.overlap_samples();

            let mut combined = self.prev.clone();
            combined.extend(samples);
            samples = combined;            
        }
        self.prev = samples[samples.len() - n..].to_vec();

        self.features = self.features.join(&features);

        Ok(true)
    }

    pub fn features(&self, n_frames: usize) -> Features {
        if self.features.len() >= n_frames {
            self.features.slice(0, n_frames)
        } else {
            let n = n_frames - self.features.len();
            let features = self.extractor.extract(&self.prev, &[], self.overlap_frames, n).unwrap();
            self.features.join(&features)
        }
    }

    pub fn consume(&mut self, n_frames: usize) {
        self.features = self.features.slice_to_end(n_frames);
        self.offset += n_frames;
    }

    pub fn consume_millis(&mut self, millis: usize) {
        self.consume(millis / Self::MILLIS_PER_FRAME);
    }
}