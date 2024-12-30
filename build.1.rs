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
    add_search_paths("LIBRARY_PATH");
    add_search_paths("CMAKE_LIBRARY_PATH");

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("cargo:rustc-link-lib=src/trtllm");
    let build_dir = format!("{}/src/trtllm/build", crate_dir);
    println!("cargo:rustc-link-search=native={}", build_dir);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir);

    println!("cargo:rerun-if-changed={}/src/trtllm/whisper.h", crate_dir);
    println!("cargo:rerun-if-changed={}/src/trtllm/build/libtrtllm.a", crate_dir);

    let bindings = bindgen::Builder::default()
        .header("src/trtllm/whisper.h")
        .clang_arg("-I/usr/include/c++/13")
        .generate()
        .expect("Unable to generate bindings");

    let bindings = format!("use serde::{{Deserialize, Serialize}};\n{}", bindings);

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Write the bindings to $OUT_DIR/bindings.rs
    std::fs::write(out_path.join("bindings.rs"), bindings).expect("Couldn't write bindings!");
}