mod sys;
mod tokenizer;
mod model;
mod audio;
mod whisper;
//mod transcript;
//pub use sys::TranscribeOptions;
pub use whisper::{Whisper, Config};
//pub use transcript::{Segment};