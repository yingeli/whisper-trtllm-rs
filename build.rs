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
    println!("cargo:rerun-if-changed=src/sys/whisper.h");
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rustc-link-search=cpp/build");
    println!("cargo:rustc-link-lib=whisper-trtllm");
    
    println!("cargo:rustc-link-search=/app/tensorrt_llm/lib");
    println!("cargo:rustc-link-lib=tensorrt_llm");
    println!("cargo:rustc-link-lib=nvinfer_plugin_tensorrt_llm");

    println!("cargo:rustc-link-search=/usr/local/lib/python3.12/dist-packages/torch/lib");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=torch_cpu");
    
    //add_search_paths("LIBRARY_PATH");
    //add_search_paths("LD_LIBRARY_PATH");
    //add_search_paths("CMAKE_LIBRARY_PATH");

    cxx_build::bridges([
        "src/sys/whisper.rs",
    ])
    //.file("src/sys/whisper.cpp")
    .include("src/sys")
    .include(".")
    .include("/app/tensorrt_llm/include")
    .include("/usr/local/cuda/include")
    .include("/opt/pytorch/pytorch/torch/csrc/api/include")
    .include("/opt/pytorch/pytorch")
    .include("/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include")
    .include("/usr/local/lib/python3.12/dist-packages/torch/include")
    //.cpp(true)
    .std("c++17")
    //.cuda(true)
    //.static_flag(true)
    //.static_crt(cfg!(target_os = "windows"))
    .flag_if_supported("/EHsc")
    .compile("whisper-trt");
}