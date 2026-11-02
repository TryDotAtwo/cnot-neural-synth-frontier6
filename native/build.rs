use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=cuda/CMakeLists.txt");
    println!("cargo:rerun-if-changed=cuda/include/cnot103/cnot_cuda.h");
    println!("cargo:rerun-if-changed=cuda/src/cnot_cuda.cu");
    for name in [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDAToolkit_ROOT",
        "CUDACXX",
        "CMAKE_CUDA_COMPILER",
        "CNOT103_CUDA_ARCHITECTURES",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let mut config = cmake::Config::new("cuda");
    config.define("CMAKE_BUILD_TYPE", "Release");
    if let Ok(compiler) = env::var("CMAKE_CUDA_COMPILER").or_else(|_| env::var("CUDACXX")) {
        config.define("CMAKE_CUDA_COMPILER", compiler);
    }
    if let Ok(architectures) = env::var("CNOT103_CUDA_ARCHITECTURES") {
        config.define("CMAKE_CUDA_ARCHITECTURES", architectures);
    }
    let destination = config.build();

    println!(
        "cargo:rustc-link-search=native={}/lib",
        destination.display()
    );
    println!("cargo:rustc-link-lib=static=cnot103_cuda");
    let runtime_lib_dir = cuda_runtime_lib_dir().unwrap_or_else(|| {
        panic!(
            "CUDA feature enabled, but the CUDA runtime library directory could not be found; set CUDA_PATH or CUDAToolkit_ROOT"
        )
    });
    println!(
        "cargo:rustc-link-search=native={}",
        runtime_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");
}

fn cuda_runtime_lib_dir() -> Option<PathBuf> {
    let target_os = env::var("CARGO_CFG_TARGET_OS").ok()?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").ok()?;
    let candidates: &[&str] = match (target_os.as_str(), target_arch.as_str()) {
        ("windows", "x86_64") => &["lib/x64", "lib"],
        ("windows", "x86") => &["lib/Win32", "lib"],
        ("linux", "x86_64") => &["lib64", "targets/x86_64-linux/lib", "lib"],
        ("linux", "aarch64") => &["lib64", "targets/aarch64-linux/lib", "lib"],
        ("macos", _) => &["lib"],
        _ => return None,
    };
    for root in cuda_roots() {
        if let Some(path) = candidates
            .iter()
            .map(|relative| root.join(relative))
            .find(|path| path.is_dir())
        {
            return Some(path);
        }
    }
    None
}

fn cuda_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for name in ["CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT"] {
        if let Some(value) = env::var_os(name) {
            let path = PathBuf::from(value);
            if path.is_dir() && !roots.contains(&path) {
                roots.push(path);
            }
        }
    }
    for name in ["CUDACXX", "CMAKE_CUDA_COMPILER"] {
        if let Some(value) = env::var_os(name) {
            if let Some(root) = Path::new(&value).parent().and_then(Path::parent) {
                let root = root.to_owned();
                if root.is_dir() && !roots.contains(&root) {
                    roots.push(root);
                }
            }
        }
    }
    roots
}
