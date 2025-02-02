use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/sys");
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rustc-link-search=/app/tensorrt_llm/lib");
    println!("cargo:rustc-link-lib=tensorrt_llm");
    println!("cargo:rustc-link-lib=nvinfer_plugin_tensorrt_llm");

    println!("cargo:rustc-link-search=/usr/local/cuda-12.8/NsightSystems-cli-2024.6.2/target-linux-x64");
    println!("cargo:rustc-link-lib=z");

    println!("cargo:rustc-link-search=/usr/local/lib/python3.12/dist-packages/torch/lib");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=torch_cuda");
    
    cxx_build::bridges([
        "src/sys/tensor.rs",
        "src/sys/whisper.rs",
    ])
    .file("src/sys/whisper.cpp")
    .file("cpp/whisper.cpp")
    .file("cpp/mel.cpp")
    .file("cpp/cnpy/cnpy.cpp")
    .include("/usr/local/lib/python3.12/dist-packages/torch/include")
    .include("/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include")
    .include("/app/tensorrt_llm/include")
    .include("/home/coder/whisper-trtllm-rs/cpp/TensorRT-LLM/cpp")
    .include("/usr/local/tensorrt/include")
    //.cpp(true)
    .std("c++20")
    .cuda(true)
    //.static_flag(true)
    //.static_crt(cfg!(target_os = "windows"))
    .flag_if_supported("/EHsc")
    .flag("-w")
    .compile("whisper-trt");
}