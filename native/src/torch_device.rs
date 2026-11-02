//! Explicit loading and validation of the LibTorch CUDA backend.
use std::env;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::OnceLock;
use tch::Device;

static CUDA_LIBRARY: OnceLock<Result<usize, String>> = OnceLock::new();

pub fn require_device(device: Device) -> Result<(), String> {
    if let Device::Cuda(index) = device {
        require_cuda_device(index)?;
    }
    Ok(())
}

pub fn require_cuda_device(index: usize) -> Result<(), String> {
    load_cuda_library()?;
    if !tch::Cuda::is_available() {
        return Err("LibTorch CUDA backend loaded, but CUDA reports no available device".into());
    }
    let count = tch::Cuda::device_count();
    let requested =
        i64::try_from(index).map_err(|_| "CUDA device index is too large".to_string())?;
    if count <= requested {
        return Err(format!(
            "CUDA device {index} requested, but LibTorch reports {count} device(s)"
        ));
    }
    Ok(())
}

fn load_cuda_library() -> Result<(), String> {
    CUDA_LIBRARY
        .get_or_init(load_cuda_library_once)
        .as_ref()
        .map(|_| ())
        .map_err(Clone::clone)
}

fn candidate_paths() -> Vec<PathBuf> {
    let filename = if cfg!(target_os = "windows") {
        "torch_cuda.dll"
    } else {
        "libtorch_cuda.so"
    };
    let mut paths = Vec::new();
    if let Some(root) = env::var_os("LIBTORCH") {
        paths.push(PathBuf::from(&root).join("lib").join(filename));
        paths.push(PathBuf::from(root).join(filename));
    }
    if let Some(lib) = env::var_os("LIBTORCH_LIB") {
        paths.push(PathBuf::from(lib).join(filename));
    }
    if let Some(search) = env::var_os("LD_LIBRARY_PATH") {
        paths.extend(env::split_paths(&search).map(|dir| dir.join(filename)));
    }
    paths.push(PathBuf::from(filename));
    paths.sort();
    paths.dedup();
    paths
}

#[cfg(target_os = "linux")]
fn load_cuda_library_once() -> Result<usize, String> {
    const RTLD_NOW: c_int = 2;
    const RTLD_GLOBAL: c_int = 0x100;
    let mut errors = Vec::new();
    for path in candidate_paths() {
        let encoded = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| "CUDA library path contains an interior NUL byte".to_string())?;
        unsafe {
            let _ = dlerror();
            let handle = dlopen(encoded.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
            if !handle.is_null() {
                return Ok(handle as usize);
            }
            let detail = dlerror();
            if !detail.is_null() {
                errors.push(format!(
                    "{}: {}",
                    path.display(),
                    CStr::from_ptr(detail).to_string_lossy()
                ));
            }
        }
    }
    Err(format!(
        "could not load libtorch_cuda.so from configured paths: {}",
        errors.join(" | ")
    ))
}

#[cfg(target_os = "linux")]
#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

#[cfg(target_os = "windows")]
fn load_cuda_library_once() -> Result<usize, String> {
    use std::os::windows::ffi::OsStrExt;
    let mut errors = Vec::new();
    for path in candidate_paths() {
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        unsafe {
            let handle = LoadLibraryW(wide.as_ptr());
            if !handle.is_null() {
                return Ok(handle as usize);
            }
            errors.push(format!("{}: LoadLibraryW failed", path.display()));
        }
    }
    Err(format!(
        "could not load torch_cuda.dll: {}",
        errors.join(" | ")
    ))
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(filename: *const u16) -> *mut c_void;
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn load_cuda_library_once() -> Result<usize, String> {
    let _ = candidate_paths();
    Err("automatic LibTorch CUDA loading is supported on Linux and Windows".into())
}
