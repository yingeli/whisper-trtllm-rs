use cxx::UniquePtr;

use std::path::Path;
use anyhow::{anyhow, Result};

#[cxx::bridge]
pub mod ffi {
    #[namespace = "tensorrt_llm::executor"]
    unsafe extern "C++" {
        include!("tensorrt_llm/executor/executor.h");
        
        type Executor = ffi::Executor;
    }

}

unsafe impl cxx::ExternType for ffi::Executor {
    type Id = cxx::type_id!("tensorrt_llm::executor::Executor");
    type Kind = cxx::kind::Trivial;
}