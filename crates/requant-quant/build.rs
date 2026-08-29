fn main() {
    // Only link libggml when the ggml-oracle feature is enabled (test-time bit-exactness oracle).
    if std::env::var("CARGO_FEATURE_GGML_ORACLE").is_ok() {
        let libdir = std::env::var("GGML_LIB_DIR").unwrap_or_else(|_| "/opt/homebrew/lib".to_string());
        println!("cargo:rustc-link-search={libdir}");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
        // Help the runtime linker find the dylib when tests execute.
        if std::env::var("DOCS_RS").is_err() {
            println!("cargo:rustc-env=DYLD_FALLBACK_LIBRARY_PATH={libdir}");
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}
