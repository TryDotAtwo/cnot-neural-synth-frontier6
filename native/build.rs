use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/CMakeLists.txt");
    println!("cargo:rerun-if-changed=cuda/include/cnot103/cnot_cuda.h");
    println!("cargo:rerun-if-changed=cuda/src/cnot_cuda.cu");
    println!("cargo:rerun-if-changed=cuda/src/packed_block_scratch.cuh");
    println!("cargo:rerun-if-changed=cuda/src/matrix_boundary.cu");
    println!("cargo:rerun-if-changed=cuda/src/block_histogram.cu");
    println!("cargo:rerun-if-changed=cuda/include/cnot103/column_diffusion.h");
    println!("cargo:rerun-if-changed=cuda/src/column_diffusion.cu");
    println!("cargo:rerun-if-changed=cuda/src/diffusion_chain.cuh");
    println!("cargo:rerun-if-changed=cuda/src/pair_diffusion.cu");
    println!("cargo:rerun-if-changed=cuda/src/shared_word.cu");
    println!("cargo:rerun-if-changed=cuda/src/shared_word_rng.cuh");
    println!("cargo:rerun-if-changed=cuda/src/shared_word_moments.cuh");
    println!("cargo:rerun-if-changed=cuda/include/cnot103/shared_word.h");
    println!("cargo:rerun-if-changed=cuda/include/cnot103/pair_diffusion.h");
    println!("cargo:rerun-if-changed=cuda/src/basis_gradient.cu");
    println!("cargo:rerun-if-changed=cuda/src/general_compiler.cu");
    println!("cargo:rerun-if-changed=cuda/src/stage26.cu");
    println!("cargo:rerun-if-changed=cuda/src/frequency.cu");
    println!("cargo:rerun-if-changed=cuda/src/shared_relation.cu");
    println!("cargo:rerun-if-changed=cuda/src/bilinear.cu");
    println!("cargo:rerun-if-changed=cuda/src/positive_packet.cu");
    println!("cargo:rerun-if-changed=cuda/src/quadratic.cu");
    println!("cargo:rerun-if-changed=cuda/src/endpoint_response.cu");
    println!("cargo:rerun-if-changed=cuda/src/endpoint_response_math.cuh");
    println!("cargo:rerun-if-changed=cuda/src/paid_basis_atlas.cu");
    println!("cargo:rerun-if-changed=cuda/src/paid_basis_atlas.cuh");
    println!("cargo:rerun-if-changed=cuda/src/paid_shear.cu");
    println!("cargo:rerun-if-changed=cuda/src/paid_basis_features.cu");
    println!("cargo:rerun-if-changed=cuda/src/template_response.cu");
    println!("cargo:rerun-if-changed=cuda/src/guarded_pair.cu");
    println!("cargo:rerun-if-changed=cuda/src/zero_excess.cu");
    println!("cargo:rerun-if-changed=cuda/src/one_excess_math.cuh");
    println!("cargo:rerun-if-changed=cuda/src/one_excess_structure.cuh");
    println!("cargo:rerun-if-changed=cuda/src/one_excess_structure.cu");
    println!("cargo:rerun-if-changed=cuda/src/one_excess_count.cuh");
    println!("cargo:rerun-if-changed=cuda/src/one_excess_count.cu");
    println!("cargo:rerun-if-changed=cuda/src/one_excess.cuh");
    println!("cargo:rerun-if-changed=cuda/src/one_excess.cu");
    println!("cargo:rerun-if-changed=cuda/src/template_response_math.cuh");
    println!("cargo:rerun-if-changed=cuda/src/torch_stream.cpp");
    for name in [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDAToolkit_ROOT",
        "CUDACXX",
        "CMAKE_CUDA_COMPILER",
        "CNOT103_CUDA_ARCHITECTURES",
        "LIBTORCH",
        "PYTHON",
        "CMAKE_PREFIX_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let mut config = cmake::Config::new("cuda");
    config.define("CMAKE_BUILD_TYPE", "Release");
    config.define("BUILD_TESTING", "OFF");
    config.define(
        "CMAKE_PREFIX_PATH",
        torch_cmake_prefix().display().to_string(),
    );
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
    println!("cargo:rustc-link-lib=dylib=c10_cuda");
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

fn torch_cmake_prefix() -> PathBuf {
    if let Some(root) = env::var_os("LIBTORCH").map(PathBuf::from) {
        for relative in ["share/cmake", "lib/cmake"] {
            let prefix = root.join(relative);
            if prefix.join("Torch/TorchConfig.cmake").is_file() {
                return prefix;
            }
        }
        panic!("LIBTORCH is set, but its CMake TorchConfig.cmake was not found");
    }

    let python = env::var("PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let output = Command::new(&python)
        .args(["-c", "import torch; print(torch.utils.cmake_prefix_path)"])
        .output()
        .unwrap_or_else(|error| {
            panic!("failed to query {python} for the Torch CMake prefix: {error}")
        });
    if !output.status.success() {
        panic!(
            "failed to query PyTorch CMake prefix: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let prefix = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    if !prefix.join("Torch/TorchConfig.cmake").is_file() {
        panic!(
            "PyTorch reported {}, but Torch/TorchConfig.cmake is missing",
            prefix.display()
        );
    }
    prefix
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
