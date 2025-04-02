use std::ops::Deref;

use cxx::UniquePtr;

use anyhow::{anyhow, Result};

#[cxx::bridge]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("whisper-trtllm-rs/src/sys/features.h");

        type Features;

        // fn features() -> UniquePtr<Features>;

        fn len(self: &Features) -> usize;

        fn slice(self: &Features, start: usize, end: usize) -> UniquePtr<Features>;

        fn slice_to_end(self: &Features, start: usize) -> UniquePtr<Features>;

        fn pad(self: &Features, padding: usize) -> UniquePtr<Features>;

        fn join(self: &Features, other: &Features) -> UniquePtr<Features>;
    }
}

pub(crate) struct Features {
    ptr: UniquePtr<ffi::Features>,
}

impl From<UniquePtr<ffi::Features>> for Features {
    fn from(ptr: UniquePtr<ffi::Features>) -> Self {
        Self {
            ptr,
        }
    }
}

impl Deref for Features {
    type Target = ffi::Features;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl Features {
    fn from_ffi(ptr: UniquePtr<ffi::Features>) -> Self {
        Self {
            ptr,
        }
    }

    pub fn len(&self) -> usize {
        self.ptr.len()
    }

    pub fn slice(&self, start: usize, end: usize) -> Self {
        let ptr = self.ptr.slice(start, end);
        Self::from_ffi(ptr)
    }

    pub fn slice_to_end(&self, start: usize) -> Self {
        let ptr = self.ptr.slice_to_end(start);
        Self::from_ffi(ptr)
    }

    pub fn pad(&self, padding: usize) -> Self {
        let ptr = self.ptr.pad(padding);
        Self::from_ffi(ptr)
    }

    pub fn join(&self, other: &Self) -> Self {
        self.ptr.join(&other.ptr).into()
    }
}

unsafe impl Send for Features {}
unsafe impl Sync for Features {}