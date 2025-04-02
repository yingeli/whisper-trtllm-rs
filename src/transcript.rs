use serde::Serialize;
use super::tokenizer::Tokenizer;

/*
pub struct SegmentIterator<'a> {
    tokens: Vec<u32>,
    tokenizer: &'a Tokenizer,
    position: usize,
}

impl <'a> SegmentIterator<'a> {
    pub fn new(tokens: Vec<u32>, tokenizer: &'a Tokenizer, position: usize) -> Self {
        Self {
            tokens,
            tokenizer,
            position,
        }
    }

    fn current_token(&self) -> Option<u32> {
        if self.position < self.tokens.len() {
            Some(self.tokens[self.position])
        } else {
            None
        }
    }

    pub fn next(&mut self) -> Option<Segment> {
        loop {
            let Some(token) = self.current_token() else {
                return None;
            };

            let Some(start) = self.tokenizer.timestamp_to_millis(token) else {
                self.position += 1;
                continue;
            };
            let start_pos = self.position;
                
            self.position += 1;

            loop {
                let Some(token) = self.current_token() else {
                    // end of tokens
                    if self.position <= start_pos + 1 {
                        return None;
                    }
                        
                    let tokens = &self.tokens[start_pos..self.position];
                    let segment = Segment::no_end(start, tokens);
                    return Some(segment)
                };

                let Some(end) = self.tokenizer.timestamp_to_millis(token) else {
                    self.position += 1;
                    continue;                    
                };
                        
                //let text = self.tokenizer.decode(&self.tokens[start_pos + 1..self.position], false).unwrap();    
                self.position += 1;
        
                if self.position <= start_pos + 1 {
                    // No tokens between start and end
                    break;
                }

                // Use a direct slice from self.tokens to preserve the lifetime 'a
                let tokens = &self.tokens[start_pos..self.position];
                let segment = Segment::new(start, end, tokens);
                return Some(segment);
            }
        }
    }
}
*/

#[derive(Debug, Serialize)]
pub struct Segment {
    start: usize,
    end: usize,
    text: String,
    // tokens: &'a [u32],
}

impl Segment {
    const NO_END: usize = usize::MAX;

    pub fn new(start: usize, end: usize, text: String) -> Self {
        Self {
            start,
            end,
            text,
        }
    }

    pub fn no_end(start: usize, text: String) -> Self {
        Self {
            start,
            end: Self::NO_END,
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

    pub fn has_end(&self) -> bool {
        self.end < usize::MAX
    }

    /*
    pub fn tokens(&self) -> &[u32] {
        self.tokens
    }

    pub fn text_tokens(&self) -> &[u32] {
        if self.has_end() {
            &self.tokens[1..self.tokens.len() - 1]
        } else {
            &self.tokens[1..]
        }
    }
    */
}

/*

#[derive(Debug, Serialize)]
pub struct Transcript {
    language: String,
    segments: Vec<Segment>,
}

impl Transcript {
    pub fn new(tokens: Vec<u32>, tokenizer: &Tokenizer) -> Self {
        let token = tokens[1];
        let language = 
        Self {
            language,
            segments,
        }
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn segments(&self) -> impl Stream<Item = Segment> {
        stream! {
            for i in 0..10 {
                sleep(Duration::from_millis(100)).await;
                yield Segment::new(i, i + 1, format!("Segment {}", i));
            }
        }
    }
    */

