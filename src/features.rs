use tokio::io::{AsyncRead, AsyncReadExt};
use anyhow::{anyhow, Ok, Result};
use std::collections::VecDeque;
use super::sys::{Features, LogMelSpectrogram};
use futures::stream::{Stream, StreamExt};

pub(crate) struct FeatureBuffer<'a, S> {
    extractor: &'a LogMelSpectrogram,
    stream: S,
    prev: Vec<f32>,
    overlap_frames: usize,
    features: Features,
    offset: usize,
    eof: bool,
}

impl<'a, S> FeatureBuffer<'a, S> 
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

    pub async fn features(&mut self, chunk_size: usize) -> Result<Option<Features>> {
        loop {
            if self.features.len() >= chunk_size {
                let chuck = self.features.slice(0, chunk_size);
                return Ok(Some(chuck));
            }

            if self.eof {
                if self.features.len() == 0 {
                    return Ok(None);
                } else {
                    let n = chunk_size - self.features.len();
                    let features = self.features.pad(n);
                    return Ok(Some(features));
                }
            }

            self.fill().await?;
        }
    }

    pub async fn fill(&mut self) -> Result<()> {
        let Some(mut samples) = self.stream.next().await else {
            let features = self.extractor.extract_final(&self.prev, &[]).unwrap();
            self.features.join(&features.slice_to_end(self.overlap_frames));
            self.eof = true;
            return Ok(());
        };

        let features = self.extractor.extract(
            &self.prev,
            &samples,
        )?.slice_to_end(self.overlap_frames);

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

        Ok(())
    }

    pub fn consume(&mut self, n_frames: usize) {
        self.features = self.features.slice_to_end(n_frames);
        self.offset += n_frames;
    }

    pub fn consume_millis(&mut self, millis: usize) {
        self.consume(millis / Self::MILLIS_PER_FRAME);
    }
}