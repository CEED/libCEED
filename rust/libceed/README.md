# libceed: efficient, extensible discretization

This crate provides an interface to [libCEED](https://libceed.readthedocs.io),
which is a performance-portable library for extensible element-based
discretization for partial differential equations and related computational
problems. The formulation is algebraic and intended to be lightweight and easy
to incorporate in higher level abstractions. See the [libCEED user
manual](https://libceed.readthedocs.io) for details on [interface
concepts](https://libceed.readthedocs.io/en/latest/libCEEDapi/) and extensive
examples.

![libCEED operator decomposition](https://libceed.readthedocs.io/en/latest/_images/libCEED.png)

## Usage

To call libCEED from a Rust package, the following `Cargo.toml` can be used.
```toml
[dependencies]
libceed = "0.9.0"
```

For a development version of the libCEED Rust bindings, use the following `Cargo.toml`.
```toml
[dependencies]
libceed = { git = "https://github.com/CEED/libCEED", branch = "main" }
```

```rust
extern crate libceed;

fn main() -> libceed::Result<()> {
    let ceed = libceed::Ceed::init("/cpu/self/ref");
    let xc = ceed.vector_from_slice(&[0., 0.5, 1.0])?;
    let xs = xc.view()?;
    assert_eq!(xs[..], [0., 0.5, 1.0]);
    Ok(())
}
```

This crate provides modules for each object, but they are usually created from
the `Ceed` object as with the vector above. The resource string passed to
`Ceed::init` is used to identify the "backend", which includes algorithmic
strategies and hardware such as NVIDIA and AMD GPUs. See the [libCEED
documentation](https://libceed.readthedocs.io/en/latest/gettingstarted/#backends)
for more information on available backends.

## Examples

Examples of libCEED can be found in the [libCEED repository](https://github.com/CEED/libCEED) under the
`examples/rust` directory.

## License: BSD-2-Clause

## Contributing

The `libceed` crate is developed within the [libCEED
repository](https://github.com/CEED/libCEED). See the [contributing
guidelines](https://libceed.readthedocs.io/en/latest/CONTRIBUTING/) for details.
