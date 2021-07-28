# Release Procedures

*These notes are meant for a maintainer to create official releases.*

In preparing a release, create a branch to hold pre-release commits. We ideally want all release mechanics (for all languages) to be in one commit, which will then be tagged. (This will change if/when we stop synchronizing releases across all language bindings.)

## Core C library

Some minor bookkeeping updates are needed when releasing a new version of the core library.

The version number must be updated in

* `include/ceed/ceed.h`
* `ceed.pc.template`
* `Doxyfile`
* `CITATION.cff`

as well as `include/ceed/ceed.h` (`CEED_VERSION_MAJOR`, `CEED_VERSION_MINOR`).

Additionally, the release notes in `doc/sphinx/source/releasenotes.rst` should be updated. Use `git log --first-parent v0.7..` to get a sense of the pull requests that have been merged and thus might warrant emphasizing in the release notes. While doing this, gather a couple sentences for key features to highlight on [GitHub releases](https://github.com/CEED/libCEED/releases). The "Current Main" heading needs to be named for the release.

Use `make doc-latexpdf` to build a PDF users manual and inspect it for missing references or formatting problems (e.g., with images that were converted to PDF). This contains the same content as the website, but will be archived on Zenodo.

### Quality control and good citizenry

1. If making a minor release, check for API and ABI changes that could break [semantic versioning](https://semver.org/). The [ABI compliance checker](https://github.com/lvc/abi-compliance-checker) is a useful tool, as is `nm -D libceed.so` and checking for public symbols (capital letters like `T` and `D` that are not namespaced).

2. Double check testing on any architectures that may not be exercised in continuous integration (e.g., HPC facilities) and with users of libCEED, such as MFEM and PETSc applications. While unsupported changes do not prevent release, it's polite to make a PR to support the new release, and it's good for quality to test before taggin a libCEED release.

3. Update and test all the language bindings (see below) within the branch.

4. Check that `spack install libceed@develop` works prior to tagging. The Spack `libceed/package.py` file should be updated immediately after tagging a release.

### Tagging and releasing on GitHub

0. Confirm all the steps above, including all language bindings.
1. `git commit -am'libCEED 0.8.1'`
More frequently, this is amending the commit message on an in-progress commit, after rebasing if applicable on latest `main`.
2. `git push` updates the PR holding release; opportunity for others to review
3. `git switch main && git merge --ff-only HEAD@{1}` fast-forward merge into `main`
4. `git tag --sign -m'libCEED 0.8.1'`
5. `git push origin main v0.8.1`
6. Draft a [new release on GitHub](https://github.com/CEED/libCEED/releases), using a few sentences gathered from the release notes.
7. Submit a PR to Spack.
8. Publish Julia, Python, and Rust packages.

### Archive Users Manual on Zenodo

Generate the PDF using `make doc-latexpdf`, click "New version" on the [Zenodo
record](https://zenodo.org/record/4302737) and upload. Update author info if applicable (new
authors, or existing authors changing institutions). Make a new PR to update the version
number and DOI in `README.rst` and `doc/bib/references.bib`.

## Julia

libCEED's Julia interface (LibCEED.jl) has two components:

* LibCEED.jl, the user-facing package that contains the Julia interface.
* libCEED_jll, a binary wrapper package ("jll package") that contains prebuilt binaries of the
  libCEED library for various architectures.

When there is a new release of libCEED, both of these components need to be updated. First,
libCEED_jll is updated, and then LibCEED.jl.

### Updating libCEED_jll

The binary wrapper package libCEED_jll is updated by making a pull request against
[Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil), the Julia community build tree. In this
PR, the file `L/libCEED/build_tarballs.jl` should be changed to update version number and change the
hash of the libCEED commit to use to build the binaries, similar to the following diff:
```diff
diff --git a/L/libCEED/build_tarballs.jl b/L/libCEED/build_tarballs.jl
--- a/L/libCEED/build_tarballs.jl
+++ b/L/libCEED/build_tarballs.jl
@@ -3,11 +3,11 @@
 using BinaryBuilder, Pkg

 name = "libCEED"
-version = v"0.7.0"
+version = v"0.8.0"

 # Collection of sources required to complete build
 sources = [
-    GitSource("https://github.com/CEED/libCEED.git", "06988bf74cc6ac18eacafe7930f080803395ba29")
+    GitSource("https://github.com/CEED/libCEED.git", "e8f234590eddcce2220edb1d6e979af7a3c35f82")
 ]
```
After the PR is merged into Yggdrasil, the new version of libCEED_jll will automatically be
registered, and then we can proceed to update LibCEED.jl.

### Updating LibCEED.jl

After the binary wrapper package libCEED_jll has been updated, we are ready to update the main Julia
interface LibCEED.jl. This requires updating the file `julia/LibCEED.jl/Project.toml` in the libCEED
repository. The version number should be incremented, and the dependency on the updated version of
libCEED_jll should be listed:
```diff
diff --git a/julia/LibCEED.jl/Project.toml b/julia/LibCEED.jl/Project.toml
--- a/julia/LibCEED.jl/Project.toml
+++ b/julia/LibCEED.jl/Project.toml
@@ -1,7 +1,7 @@
 name = "LibCEED"
 uuid = "2cd74e05-b976-4426-91fa-5f1011f8952b"
-version = "0.1.0"
+version = "0.1.1"

 [deps]
 CEnum = "fa961155-64e5-5f13-b03f-caf6b980ea82"
@@ -26,4 +26,4 @@ Cassette = "0.3"
 Requires = "1"
 StaticArrays = "0.12"
 UnsafeArrays = "1"
-libCEED_jll = "0.7"
+libCEED_jll = "0.8"
```
Once this change is merged into libCEED's `main` branch, the updated package version can be
registered using the GitHub registrator bot by commenting on the commit:

> @JuliaRegistrator register branch=main subdir=julia/LibCEED.jl

At this point, the bot should create a PR against the [general Julia
registry](https://github.com/JuliaRegistries/General), which should be merged automatically after a
short delay.

### Moving development tests to release tests

LibCEED.jl has both _development_ and _release_ unit tests. The _release_ tests are run both with
the current build of libCEED, and with the most recent release of libCEED_jll. The _development_
tests may use features which were not available in the most recent release, and so they are only run
with the current build of libCEED.

Upon release, the development tests may be moved to the release tests, so that these features will
be tested against the most recent release of libCEED_jll. The release tests are found in the file
`julia/LibCEED.jl/test/runtests.jl` and the development tests are found in
`julia/LibCEED.jl/test/rundevtests.jl`.

## Python

The Python package gets its version from `ceed.pc.template` so there are no file modifications necessary.

1. `make wheel` builds and tests the wheels using Docker. See the [manylinux repo](https://github.com/pypa/manylinux) for source and usage inforamtion. If this succeeds, the completed wheels are in `wheelhouse/libceed-0.8-cp39-cp39-manylinux2010_x86_64.whl`.
2. Manually test one or more of the wheels by creating a virtualenv and using `pip install wheelhouse/libceed-0.8-cp39-cp39-manylinux2010_x86_64.whl`, then `python -c 'import libceed'` or otherwise running tests.
3. Create a `~/.pypirc` with entries for `testpypi` (`https://test.pypi.org/legacy/`) and the real `pypi`.
4. Upload to `testpypi` using
```console
$ twine upload --repository testpypi wheelhouse/libceed-0.8-cp39-cp39-manylinux2010_x86_64.whl
```
5. Test installing on another machine/in a virtualenv:
```console
$ pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple libceed
```
The `--extra-index-url` argument allows dependencies like `cffi` and `numpy` from being fetched from the non-test repository.
6. Do it live:
```console
$ twine upload --repository pypi wheelhouse/libceed-0.8-cp39-cp39-manylinux2010_x86_64.whl
```
Note that this cannot be amended.

## Rust

The Rust crates for libCEED are split into
1. [`libceed-sys`](https://crates.io/crates/libceed-sys), which handles building/finding the `libceed.so` or `libceed.a` library and providing unsafe Rust bindings (one to one with the C interface, using C FFI datatypes)
2. [`libceed`](https://crates.io/crates/libceed) containing the safe and idiomatic Rust bindings.

We currently apply the same version number across both of these crates. There are some tests for version strings matching, but in short, one needs to update the following locations.

```console
$ git grep '0\.8' -- rust/
rust/libceed-sys/Cargo.toml:version = "0.8.0"
rust/libceed-sys/README.md:libceed-sys = "0.8.0"
rust/libceed-sys/build.rs:        .atleast_version("0.8")
rust/libceed/Cargo.toml:version = "0.8.0"
rust/libceed/Cargo.toml:libceed-sys = { version = "0.8", path = "../libceed-sys" }
rust/libceed/README.md:libceed = "0.8.0"
```

After doing this,

1. `cargo package --list` to see that the file list makes sense.
2. `cargo package` to build crates locally
3. `cargo publish` to publish the crates to https://crates.io
