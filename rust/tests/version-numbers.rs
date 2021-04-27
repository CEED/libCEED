#[test]
fn test_readme_deps() {
    let path = std::env::current_dir().unwrap();
    let path = path.to_str().unwrap();
    version_sync::assert_markdown_deps_updated!(if &path[path.len() - 4..path.len()] == "rust" {
        "../README.rst"
    } else {
        "README.rst"
    });
}

#[test]
fn test_html_root_url() {
    let path = std::env::current_dir().unwrap();
    let path = path.to_str().unwrap();
    version_sync::assert_html_root_url_updated!(if &path[path.len() - 4..path.len()] == "rust" {
        "src/lib.rs"
    } else {
        "rust/src/lib.rs"
    });
}
