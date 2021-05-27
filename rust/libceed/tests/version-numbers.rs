use std::path::{Path, PathBuf};

#[test]
fn test_readme_deps() {
    let path = std::env::current_dir().unwrap();
    version_sync::assert_markdown_deps_updated!(if path.ends_with("rust/libceed") {
        "../../README.rst"
    } else {
        "README.rst"
    });
}

fn get_rel_path(rel: impl AsRef<Path>) -> PathBuf {
    let path = std::env::current_dir().unwrap();
    if path.ends_with("rust/libceed") {
        rel.as_ref().to_path_buf()
    } else {
        Path::new("rust/libceed").join(rel)
    }
}

#[test]
fn test_html_root_url() {
    version_sync::assert_html_root_url_updated!(get_rel_path("src/lib.rs").to_str().unwrap());
}

#[test]
fn test_doc_version() {
    version_sync::assert_contains_regex!(
        get_rel_path("src/lib.rs").to_str().unwrap(),
        "{name} = \"{version}\""
    );
}
