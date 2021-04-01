# Release Procedures

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

> @JuliaRegistrator register branch=main subdir=julia/LibCEED.jl”

At this point, the bot should create against the [general Julia
registry](https://github.com/JuliaRegistries/General), which should be merged automatically after a
short delay.
