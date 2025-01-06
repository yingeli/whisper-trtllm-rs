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
    //println!("cargo:rustc-link-lib=/app/tensorrt_llm/lib");
    //println!("cargo:rustc-link-search=/home/coder/whisper-trt/TensorRT-LLM/cpp/build/tensorrt_llm");
    println!("cargo:rustc-link-search=/app/tensorrt_llm/lib");
    println!("cargo:rustc-link-lib=tensorrt_llm");
    //println!("cargo:rustc-link-search=/home/coder/whisper-trt/TensorRT-LLM/cpp/build/tensorrt_llm/plugins");
    println!("cargo:rustc-link-lib=nvinfer_plugin_tensorrt_llm");    
    //println!("cargo:rustc-link-search=/app/tensorrt_llm/lib");
    //println!("cargo:rustc-link-lib=th_common");
    //println!("cargo:rustc-link-lib=decoder_attention");
    //println!("cargo:rustc-link-lib=tensorrt_llm_ucx_wrapper");
    //println!("cargo:rustc-link-lib=nvinfer_plugin_tensorrt_llm");
    //println!("cargo:rustc-link-lib=tensorrt_llm_nvrtc_wrapper");   
    //println!("cargo:rustc-link-search=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs");
    //println!("cargo:rustc-link-search=/home/coder/whisper-trt/TensorRT-LLM/cpp/tensorrt_llm/executor/x86_64-linux-gnu");
    //println!("cargo:rustc-link-lib=static=tensorrt_llm_executor_static");
    //println!("cargo:rustc-link-search=/home/coder/whisper-trt/TensorRT-LLM/cpp/tensorrt_llm/batch_manager/x86_64-linux-gnu");
    //println!("cargo:rustc-link-lib=static=tensorrt_llm_batch_manager_static");
    //println!("cargo:rustc-link-search=/opt/hpcx/ompi/lib");
    //println!("cargo:rustc-link-lib=mpi");     
    
    //add_search_paths("LIBRARY_PATH");
    //add_search_paths("LD_LIBRARY_PATH");
    //add_search_paths("CMAKE_LIBRARY_PATH");

    cxx_build::bridges([
        "src/sys/executor.rs",
    ])
    //.file("src/trtllm/whisper.cpp")
    //.file("TensorRT-LLM/cpp/tensorrt_llm/executor/x86_64-linux-gnu/libtensorrt_llm_executor_static.pre_cxx11.a")
    //.file("TensorRT-LLM/cpp/tensorrt_llm/executor/x86_64-linux-gnu/libtensorrt_llm_executor_static.a")
    .include("src/sys")
    .include("/app/tensorrt_llm/include")
    .include("/usr/local/cuda/include")
    .include("/opt/pytorch/pytorch/torch/csrc/api/include")
    //.cpp(true)
    .std("c++17")
    //.cuda(true)
    //.static_flag(true)
    //.static_crt(cfg!(target_os = "windows"))
    .flag_if_supported("/EHsc")
    .compile("whisper-trt");
}