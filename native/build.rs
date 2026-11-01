fn main() {
    println!("cargo:rerun-if-changed=cuda/CMakeLists.txt");
    println!("cargo:rerun-if-changed=cuda/include/cnot103/cnot_cuda.h");
    println!("cargo:rerun-if-changed=cuda/src/cnot_cuda.cu");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let destination = cmake::Config::new("cuda")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();

    println!(
        "cargo:rustc-link-search=native={}/lib",
        destination.display()
    );
    println!("cargo:rustc-link-lib=static=cnot103_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
