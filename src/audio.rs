use tokio::io::{AsyncRead, AsyncReadExt};
use anyhow::{anyhow, Result};
use std::{collections::VecDeque, f32::MIN};
use futures::stream::Stream;

pub(crate) struct Audio<R> {
    reader: R,
    eof: bool,
    buffer: VecDeque<f32>,
    offset: usize,
    samples_per_millis: usize,
}

impl<R: AsyncRead + Unpin> Audio<R> {
    const CHUNK_SIZE: usize = 30 * 16000;
    
    pub fn new(reader: R, sample_rate: usize) -> Self {
        Self {
            reader,
            samples_per_millis: sample_rate / 1000,
            eof: false,
            buffer: VecDeque::with_capacity(Self::CHUNK_SIZE * 2),
            offset: 0,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn duration(&self) -> usize {
        self.offset + self.buffer.len() / self.samples_per_millis
    }

    pub async fn samples(&self, n: usize) -> Result<(&[f32], &[f32])> {
        loop {
            if self.buffer.len() >= n {
                let (first, second) = self.buffer.as_slices();
                if first.len() >= n {
                    return Ok((&first[..n], &second[..0]));
                } else {
                    let second_len = n - first.len();
                    return Ok((first, &second[..second_len]));
                }
            }

            if self.eof {
                return Ok((self.buffer.as_slices().0, self.buffer.as_slices().1)); // Return whatever is left in the buffer
            }

            // If we don't have enough samples, fill the buffer
        }
    }

    async fn fill_until(&mut self, n: usize) -> Result<(&[f32], &[f32])> {
        while !self.eof && self.buffer.len() < n {
            match self.reader.read_i16_le().await {
                Ok(sample) => self.buffer.push_back(sample as f32 / i16::MAX as f32),
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::UnexpectedEof {
                        return Err(anyhow!(e));
                    }
                    self.eof = true;
                    break;
                }
            }
        }
        let (first, second) = self.buffer.as_slices();
        Ok((&[f32], &[f32]))
    }

    pub fn consume(&mut self, n: usize) {
        self.buffer.drain(..n);
        self.offset += n;
    }

    pub fn consume_millis(&mut self, millis: usize) {
        self.consume(millis * self.samples_per_millis);
    }

    pub fn eof(&self) -> bool {
        self.eof
    }

    /*
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

    pub fn chunk_end(&self) -> usize {
        self.offset + self.chunk_duration()
    }

    pub async fn fill_chunk(&mut self) -> Result<(&[f32], &[f32])> {
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
        Ok(self.chunk())
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
    */
}