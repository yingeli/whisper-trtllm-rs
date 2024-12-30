extern crate bindgen;

use std::env;
use std::path::PathBuf;

#[cfg(not(target_os = "windows"))]
const PATH_SEPARATOR: char = ':';

#[cfg(target_os = "windows")]
const PATH_SEPARATOR: char = ';';

fn add_search_paths(key: &str) {
    println!("cargo:rerun-if-env-changed={}", key);
    if let Ok(library_path) = env::var(key) {
        library_path
            .split(PATH_SEPARATOR)
            .filter(|v| !v.is_empty())
            .for_each(|v| {
                println!("cargo:rustc-link-search={}", v);
            });
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/trtllm/whisper.h");
    println!("cargo:rerun-if-changed=build.rs");
    add_search_paths("LIBRARY_PATH");
    add_search_paths("CMAKE_LIBRARY_PATH");

    cxx_build::bridges([
        "src/trtllm/whisper.rs",
    ])
    .file("src/trtllm/whisper.cpp")
    .include("TensorRT-LLM/cpp/include")
    .include("/usr/local/cuda/include")
    .std("c++17")
    .static_crt(cfg!(target_os = "windows"))
    .flag_if_supported("/EHsc")
    .compile("whisper-trt");
}