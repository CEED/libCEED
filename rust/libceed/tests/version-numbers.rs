// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors
// All Rights Reserved. See the top-level COPYRIGHT and NOTICE files for details.
//
// SPDX-License-Identifier: (BSD-2-Clause)
//
// This file is part of CEED:  http://github.com/ceed

use std::path::{Path, PathBuf};

#[test]
fn test_readme_deps() {
    let path = std::env::current_dir().unwrap();
    version_sync::assert_markdown_deps_updated!(if path.ends_with("rust/libceed") {
        "../../README.md"
    } else {
        "README.md"
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
        get_rel_path("README.md").to_str().unwrap(),
        "{name} = \"{version}\""
    );
}
