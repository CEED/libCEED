extern crate bindgen;
extern crate pkg_config;

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env("OUT_DIR").unwrap());
    let statik = env("CARGO_FEATURE_STATIC").is_some();
    let system = env("CARGO_FEATURE_SYSTEM").is_some();

    let ceed_pc = if system {
        "ceed".to_string()
    } else {
        // Install libceed.a or libceed.so to $OUT_DIR/lib
        let makeflags = env("CARGO_MAKEFLAGS").unwrap();
        let mut make = Command::new("make");
        make.arg("install")
            .arg(format!("prefix={}", out_dir.to_string_lossy()))
            .env("MAKEFLAGS", makeflags)
            .current_dir("..");
        if statik {
            make.arg("STATIC=1");
        }
        run(&mut make);
        out_dir
            .join("lib")
            .join("pkgconfig")
            .join("ceed.pc")
            .to_string_lossy()
            .into_owned()
    };
    pkg_config::Config::new()
        .statik(statik)
        .atleast_version("0.8")
        .probe(&ceed_pc)
        .unwrap();

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=../include/ceed/ceed.h");
    println!("cargo:rerun-if-changed=../Makefile");
    println!("cargo:rerun-if-changed=../config.mk");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("../include/ceed/ceed.h")
        // Tell cargo to not mangle the function names
        .trust_clang_mangling(false)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok()
}

fn run(command: &mut Command) {
    println!("Running: `{:?}`", command);
    assert!(command.status().unwrap().success());
}
