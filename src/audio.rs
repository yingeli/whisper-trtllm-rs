use tokio::io::{AsyncRead, AsyncReadExt};
use anyhow::{anyhow, Result};
use std::collections::VecDeque;

pub(crate) struct Audio<R> {
    reader: R,
    buffer: VecDeque<f32>,
    offset: usize,
    is_end: bool,
}

impl<R: AsyncRead + Unpin> Audio<R> {
    const CHUNK_SIZE: usize = 30 * 16000;
    const SAMPLES_PER_MILLIS: usize = 16;
    
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: VecDeque::with_capacity(Self::CHUNK_SIZE * 2),
            offset: 0,
            is_end: false,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn duration(&self) -> usize {
        self.offset + self.buffer.len() / Self::SAMPLES_PER_MILLIS
    }

    pub fn chunk(&self) -> (&[f32], &[f32]) {
        let (first, second) = self.buffer.as_slices();
        let total_len = self.buffer.len().min(Self::CHUNK_SIZE);
        
        if first.len() >= total_len {
            // If first slice has enough elements, return first slice and empty second slice
            (&first[..total_len], &[])
        } else {
            // If we need elements from both slices
            let second_len = total_len - first.len();
            (first, &second[..second_len])
        }
    }

    pub fn chunk_duration(&self) -> usize {
        self.buffer.len().min(Self::CHUNK_SIZE)  / Self::SAMPLES_PER_MILLIS
    }

    pub fn chunk_start(&self) -> usize {
        self.offset
    }

    pub fn chunk_end(&self) -> usize {
        self.offset + self.chunk_duration()
    }

    pub async fn fill_chunk(&mut self) -> Result<()> {
        while self.buffer.len() < Self::CHUNK_SIZE && !self.is_end {
            match self.reader.read_i16_le().await {
                Ok(sample) => self.buffer.push_back(sample as f32 / i16::MAX as f32),
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::UnexpectedEof {
                        return Err(anyhow!(e));
                    }
                    self.is_end = true;
                    break;
                }
            }
        }
        // k(self.buffer.as_slices())
        Ok(())
    }

    pub async fn fill(&mut self) -> Result<()> {
        while self.buffer.len() < Self::CHUNK_SIZE * 2 && !self.is_end {
            match self.reader.read_i16_le().await {
                Ok(sample) => self.buffer.push_back(sample as f32 / i16::MAX as f32),
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::UnexpectedEof {
                        return Err(anyhow!(e));
                    }
                    self.is_end = true;
                    break;
                }
            }
        }
        Ok(())
    }

    pub fn consume(&mut self, millis: usize) {
        self.buffer.drain(..millis * Self::SAMPLES_PER_MILLIS);
        self.offset += millis;
    }

    pub fn is_end(&self) -> bool {
        self.is_end && self.buffer.len() <= Self::CHUNK_SIZE
    }
}