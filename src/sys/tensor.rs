use cxx::UniquePtr;

use anyhow::{anyhow, Result};

pub(crate) use self::ffi::Tensor;

#[cxx::bridge]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("whisper-trtllm-rs/src/sys/tensor.h");
        
        type Tensor;
    }
}