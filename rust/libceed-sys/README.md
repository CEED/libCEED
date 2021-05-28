# libceed-sys: unsafe bindings to libCEED

This is the documentation for the low level (unsafe) Rust bindings to the libCEED C
interface. See the [libCEED user manual](https://libceed.readthedocs.io) for usage
information. Note that most Rust users will prefer the higher level (safe) Rust
interface in the [`libceed` crate](https://docs.rs/libceed).

libCEED is a low-level API for for the efficient high-order discretization methods
developed by the ECP co-design Center for Efficient Exascale Discretizations (CEED).
While our focus is on high-order finite elements, the approach is mostly algebraic
and thus applicable to other discretizations in factored form.

## Usage

To use low level libCEED bindings in a Rust package, the following `Cargo.toml`
can be used.
```toml
[dependencies]
libceed-sys = "0.8.0"
```

For a development version of the libCEED Rust bindings, use the following `Cargo.toml`.
```toml
[dependencies]
libceed-sys = { git = "https://github.com/CEED/libCEED", branch = "main" }
```

Supported features:
* `static` (default): link to static libceed.a
* `system`: use libceed from a system directory (otherwise, install from source)

## Development

To develop libCEED, use `cargo build` in the `rust/libceed-sys` directory to
install a local copy and build the bindings.

If you need custom flags for the C project, we recommend using `make -C c-src
configure` to cache arguments in `c-src/config.mk`. If that file exists during
`cargo build` then edits will prompt recompilation of the bindings.

### Shared libraries
If one is developing libCEED C source and testing multiple language bindings at
once, a few seconds can be cut out of the edit/compile/test loop by disabling
the `static` feature and using

```bash
export LD_LIBRARY_PATH=$CEED_DIR/lib
export PKG_CONFIG_PATH=$CEED_DIR/lib/pkgconfig
```

#### Without system
If you disable the `static` feature and are not using a system version from a
standard path/somewhere that can be found by pkg-config, then you'll need to set
`LD_LIBRARY_PATH` to the appropriate target directory for doctests to be able to
find it. This might look like

```bash
export LD_LIBRARY_PATH=$CEED_DIR/target/debug/build/libceed-sys-d1ea22c6e1ad3f23/out/lib
```

where the precise hash value is printed during `cargo build --verbose` or you
can find it with `find target -name libceed.so`. This mode of development is
more fragile than the default (which uses static libraries).

Note that the `LD_LIBRARY_PATH` workarounds will become unnecessary if [this
issue](https://github.com/rust-lang/cargo/issues/1592) is resolved -- it's
currently closed, but the problem still exists.

## License: BSD-2-Clause

## Contributing

The `libceed-sys` crate is developed within the [libCEED
repository](https://github.com/CEED/libCEED). See the [contributing
guidelines](https://libceed.readthedocs.io/en/latest/CONTRIBUTING/) for details.
