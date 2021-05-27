use std::path::{Path, PathBuf};

fn get_rel_path(rel: impl AsRef<Path>) -> PathBuf {
    let path = std::env::current_dir().unwrap();
    if path.ends_with("rust/libceed-sys") {
        rel.as_ref().to_path_buf()
    } else {
        Path::new("rust/libceed-sys").join(rel)
    }
}

#[test]
fn test_doc_version() {
    version_sync::assert_contains_regex!(
        get_rel_path("src/lib.rs").to_str().unwrap(),
        "{name} = \"{version}\""
    );
}
