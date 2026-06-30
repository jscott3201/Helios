use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let native_dir = manifest_dir.join("../../native/gemma4_mlx");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
    let build_dir = out_dir.join("gemma4_mlx-build");

    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("include/gemma4_mlx.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("src/runtime.cc").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("src/model_manifest.cc").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("src/model_manifest.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("src/native_model.cc").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("src/native_model.h").display()
    );
    println!("cargo:rerun-if-env-changed=GEMMA4D_REQUIRE_MLX");

    fs::create_dir_all(&build_dir).expect("native build directory can be created");

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&native_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Debug");

    let require_mlx = env::var_os("GEMMA4D_REQUIRE_MLX").is_some();
    if require_mlx {
        configure.arg("-DGEMMA4D_REQUIRE_MLX=ON");
    } else {
        configure.arg("-DGEMMA4D_REQUIRE_MLX=OFF");
    }

    run(&mut configure, "configure native gemma4_mlx");
    run(
        Command::new("cmake").arg("--build").arg(&build_dir),
        "build native gemma4_mlx",
    );

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=gemma4_mlx");
    if require_mlx {
        println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
        println!("cargo:rustc-link-lib=dylib=mlx");
    }

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=c++");
    }
}

fn run(command: &mut Command, action: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to {action}: {error}"));
    if !status.success() {
        panic!("{action} failed with status {status}");
    }
}
