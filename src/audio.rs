use tokio::io::{AsyncRead, AsyncReadExt};
use anyhow::{anyhow, Result};
use std::collections::VecDeque;

pub(crate) struct Audio<R> {
    reader: R,
    buffer: VecDeque<f32>,
    offset: usize,
}

impl<R: AsyncRead + Unpin> Audio<R> {
    const CHUNK_SIZE: usize = 30 * 16000;
    const SAMPLES_PER_MILLIS: usize = 16;
    
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: VecDeque::with_capacity(Self::CHUNK_SIZE),
            offset: 0,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn duration(&self) -> usize {
        self.offset + self.buffer.len() / Self::SAMPLES_PER_MILLIS
    }

    pub fn chunk_duration(&self) -> usize {
        self.buffer.len() / Self::SAMPLES_PER_MILLIS
    }

    pub async fn fill_chunk(&mut self) -> Result<(&[f32], &[f32])> {
        while self.buffer.len() < Self::CHUNK_SIZE {
            match self.reader.read_i16_le().await {
                Ok(sample) => self.buffer.push_back(sample as f32 / i16::MAX as f32),
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::UnexpectedEof {
                        return Err(anyhow!(e));
                    }
                    break;
                }
            }
        }
        Ok(self.buffer.as_slices())
    }

    pub fn consume(&mut self, millis: usize) {
        self.buffer.drain(..millis * Self::SAMPLES_PER_MILLIS);
        self.offset += millis;
    }

    pub fn is_end(&self) -> bool {
        self.buffer.len() < Self::CHUNK_SIZE
    }
}