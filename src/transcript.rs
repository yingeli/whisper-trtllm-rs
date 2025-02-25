use futures::{Stream, StreamExt};
use async_stream::stream;
use tokio::time::{sleep, Duration};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Transcript {
    language: String,
    segments: Vec<Segment>,
}

impl Transcript {
    pub fn new(language: String, segments: Vec<Segment>) -> Self {
        Self {
            language,
            segments,
        }
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    /*
    pub fn segments(&self) -> impl Stream<Item = Segment> {
        stream! {
            for i in 0..10 {
                sleep(Duration::from_millis(100)).await;
                yield Segment::new(i, i + 1, format!("Segment {}", i));
            }
        }
    }
    */
}

#[derive(Debug, Serialize)]
pub struct Segment {
    start: usize,
    end: usize,
    text: String,
}

impl Segment {
    pub fn new(start: usize, end: usize, text: String) -> Self {
        Self {
            start,
            end,
            text,
        }
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}