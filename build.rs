use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/sys");
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=build.rs");

    //println!("cargo:rustc-link-search=/app/tensorrt_llm/lib");
    println!("cargo:rustc-link-search=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs");
    println!("cargo:rustc-link-lib=nvinfer_plugin_tensorrt_llm");
    println!("cargo:rustc-link-lib=tensorrt_llm");

    println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=z");

    println!("cargo:rustc-link-search=/usr/local/lib/python3.12/dist-packages/torch/lib");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=c10_cuda");
    println!("cargo:rustc-link-lib=c10");
    
    cxx_build::bridges([
        "src/sys/features.rs",
        "src/sys/mel.rs",
        "src/sys/whisper.rs",
    ])
    .file("cpp/cnpy/cnpy.cpp")
    .file("src/sys/mel.cpp")
    .file("src/sys/whisper.cpp")
    .include("cpp/cnpy")
    .include("/home/coder/whisper-trtllm-rs/cpp/TensorRT-LLM/cpp/include")
    .include("/home/coder/whisper-trtllm-rs/cpp/TensorRT-LLM/cpp")    
    .include("/usr/local/lib/python3.12/dist-packages/torch/include")
    .include("/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include")
    .include("/usr/local/tensorrt/include")
    //.cpp(true)
    .std("c++20")
    .cuda(true)
    //.static_flag(true)
    //.static_crt(cfg!(target_os = "windows"))
    .flag_if_supported("/EHsc")
    .flag("-w")
    .compile("whisper-trtllm");
}