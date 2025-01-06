use cxx::UniquePtr;

use std::path::Path;
use std::sync::Once;
use anyhow::{anyhow, Result};

static INIT: Once = Once::new();

#[cxx::bridge]
mod ffi {
    #[namespace = "tensorrt_llm::executor"]
    #[derive(Copy, Clone, Debug)]
    #[repr(i32)]
    enum BatchingType {
        kSTATIC = 0,
        kINFLIGHT = 1,
    }

    unsafe extern "C++" {
        include!("tensorrt_llm/executor/types.h");
        
        #[namespace = "tensorrt_llm::executor"]
        type BatchingType;

        #[namespace = "tensorrt_llm::executor"]
        type Executor;

        include!("executor.h");

        fn initialize();

        fn executor(model_path: &str, batching_type: BatchingType) -> UniquePtr<Executor>;
    }
}

pub enum BatchingType {
    Static,
    Inflight,
}

impl Default for BatchingType {
    fn default() -> Self {
        Self::Static
    }
}

impl BatchingType {
    fn to_ffi(&self) -> ffi::BatchingType {
        match self {
            Self::Static => ffi::BatchingType::kSTATIC,
            Self::Inflight => ffi::BatchingType::kINFLIGHT,
        }
    }
}

pub struct Executor {
    inner: UniquePtr<ffi::Executor>,
}

impl Executor {
    fn initialize() {
        ffi::initialize();
    }

    pub fn load<P: AsRef<Path>>(model_path: P, batching_type: BatchingType) -> Result<Self> {
        INIT.call_once(|| {
            Self::initialize();
        });

        let model_path = model_path.as_ref();
        let path = model_path.to_str().ok_or_else(|| anyhow!("invalid path: {}", model_path.display()))?;
        let inner = ffi::executor(path, batching_type.to_ffi());

        Ok(Self { inner })
    }
}