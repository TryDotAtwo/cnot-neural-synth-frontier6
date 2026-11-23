use std::fs;
use std::path::{Path, PathBuf};

pub fn ordered_toolkit_roots(
    explicit_root: Option<&Path>,
    compiler: Option<&Path>,
    cuda_home: Option<&Path>,
    cuda_path: Option<&Path>,
) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(root) = explicit_root {
        push_existing_root(&mut roots, root.to_owned());
    }
    if let Some(compiler) = compiler {
        if let Some(root) = compiler_root(compiler) {
            push_existing_root(&mut roots, root);
        }
    }
    if let Some(root) = cuda_home {
        push_existing_root(&mut roots, root.to_owned());
    }
    if let Some(root) = cuda_path {
        push_existing_root(&mut roots, root.to_owned());
    }
    roots
}

fn push_existing_root(roots: &mut Vec<PathBuf>, root: PathBuf) {
    if root.is_dir() && !roots.contains(&root) {
        roots.push(root);
    }
}

fn compiler_root(compiler: &Path) -> Option<PathBuf> {
    let compiler = fs::canonicalize(compiler).unwrap_or_else(|_| compiler.to_owned());
    let bin = compiler.parent()?;
    let is_bin_dir = bin
        .file_name()
        .map(|name| name.to_string_lossy().eq_ignore_ascii_case("bin"))
        .unwrap_or(false);
    if is_bin_dir {
        bin.parent().map(Path::to_path_buf)
    } else {
        None
    }
}

pub fn find_runtime_lib_dir(
    roots: &[PathBuf],
    target_os: &str,
    target_arch: &str,
) -> Option<PathBuf> {
    let candidates: &[&str] = match (target_os, target_arch) {
        ("windows", "x86_64") => &["lib/x64", "lib"],
        ("windows", "x86") => &["lib/Win32", "lib"],
        ("linux", "x86_64") => &["lib64", "targets/x86_64-linux/lib", "lib"],
        ("linux", "aarch64") => &["lib64", "targets/aarch64-linux/lib", "lib"],
        ("macos", _) => &["lib"],
        _ => return None,
    };

    roots.iter().find_map(|root| {
        candidates
            .iter()
            .map(|relative| root.join(relative))
            .find(|directory| {
                directory.is_dir() && contains_linkable_cudart(directory, target_os)
            })
    })
}

fn contains_linkable_cudart(directory: &Path, target_os: &str) -> bool {
    let Ok(entries) = fs::read_dir(directory) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        if !path.is_file() {
            return false;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        match target_os {
            "linux" => name == "libcudart.so" || name.starts_with("libcudart.so."),
            "windows" => name.eq_ignore_ascii_case("cudart.lib"),
            "macos" => name == "libcudart.dylib",
            _ => false,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempTree(PathBuf);

    impl TempTree {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "cnot103-cuda-paths-{}-{}",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir_all(&path).expect("create isolated CUDA discovery fixture");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn skips_an_existing_lib64_without_cudart_for_the_real_target_runtime() {
        let tree = TempTree::new();
        let root = tree.path().join("cuda");
        let empty_lib64 = root.join("lib64");
        let target_lib = root.join("targets/x86_64-linux/lib");
        fs::create_dir_all(&empty_lib64).unwrap();
        fs::create_dir_all(&target_lib).unwrap();
        fs::write(target_lib.join("libcudart.so.13"), b"test runtime").unwrap();

        assert_eq!(
            find_runtime_lib_dir(&[root], "linux", "x86_64"),
            Some(target_lib),
        );
    }

    #[test]
    fn explicit_toolkit_root_precedes_compiler_and_stale_home() {
        let tree = TempTree::new();
        let explicit = tree.path().join("approved-toolkit");
        let compiler_toolkit = tree.path().join("compiler-toolkit");
        let compiler_bin = compiler_toolkit.join("bin");
        let stale_home = tree.path().join("stale-home");
        let fallback_path = tree.path().join("fallback-path");
        fs::create_dir_all(&explicit).unwrap();
        fs::create_dir_all(&compiler_bin).unwrap();
        fs::create_dir_all(&stale_home).unwrap();
        fs::create_dir_all(&fallback_path).unwrap();
        let compiler = compiler_bin.join("nvcc");
        fs::write(&compiler, b"compiler").unwrap();

        assert_eq!(
            ordered_toolkit_roots(
                Some(&explicit),
                Some(&compiler),
                Some(&stale_home),
                Some(&fallback_path),
            ),
            vec![explicit, compiler_toolkit, stale_home, fallback_path],
        );
    }

    #[test]
    fn does_not_accept_an_existing_directory_without_a_linkable_runtime() {
        let tree = TempTree::new();
        let root = tree.path().join("cuda");
        fs::create_dir_all(root.join("lib64")).unwrap();
        fs::create_dir_all(root.join("targets/x86_64-linux/lib")).unwrap();
        assert_eq!(
            find_runtime_lib_dir(&[root], "linux", "x86_64"),
            None,
        );
    }
}
