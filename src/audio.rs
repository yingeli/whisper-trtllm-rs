use tokio::io::{AsyncRead};
use anyhow::{anyhow, Result};

pub(crate) struct Audio<R> {
    reader: R,
    buffer: Vec<f32>,
}

impl<R: AsyncRead> Audio<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: vec![],
        }
    }

    pub async fn read_chunk(&mut self) -> Result<&[f32]> {
        self.buffer.resize(n, 0.0);
        self.reader.read_f32_into::<LittleEndian>(&mut self.buffer)?;
        Ok(&self.buffer)
    }
}