#[test]
fn test_readme_deps() {
    let path = std::env::current_dir().unwrap();
    let path = path.to_str().unwrap();
    let length = path.len();
    if &path[length-4..length] == "rust" {
        version_sync::assert_markdown_deps_updated!("../README.rst");
    } else {
        version_sync::assert_markdown_deps_updated!("README.rst");
    }
}

#[test]
fn test_html_root_url() {
    let path = std::env::current_dir().unwrap();
    let path = path.to_str().unwrap();
    let length = path.len();
    if &path[length-4..length] == "rust" {
        version_sync::assert_html_root_url_updated!("src/lib.rs");
    } else {
        version_sync::assert_html_root_url_updated!("rust/src/lib.rs");
    }
}
