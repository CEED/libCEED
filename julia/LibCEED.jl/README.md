# LibCEED.jl: Julia Interface for [libCEED](https://github.com/CEED/libCEED)

## Installation

When the LibCEED.jl package is built, it requires the environment variable
`JULIA_LIBCEED_LIB` to be set to the location of the compiled libCEED shared
library.

For example, the package can be installed by:
```julia
% JULIA_LIBCEED_LIB=/path/to/libceed.so julia
julia> # press ] to enter package manager

(@v1.5) pkg> add LibCEED
```
or, equivalently,
```julia
% julia

julia> withenv("JULIA_LIBCEED_LIB" => "/path/to/libceed.so") do
    Pkg.add("LibCEED")
end
```


## Usage

This package provides both a low-level and high-level interface for libCEED.

### Low-Level Interface

The low-level interface (provided in the `LibCEED.C` module) is in one-to-one
correspondence with the C libCEED iterface, and is automatically generated (with
some minor manual modifications) using the Julia package Clang.jl. The script
used to generate bindings is available in `generate_bindings.jl`.

With the low-level interface, the user is responsible for freeing all allocated
memory (calling the appropriate `Ceed*Destroy` functions). This interface is
not type-safe, and calling functions with the wrong arguments can cause libCEED
to crash.

### High-Level Interface

The high-level interface provides a more idiomatic Julia interface to the
libCEED library. Objects allocated using the high-level interface will
automatically be destroyed by the garbage collector, so the user does not need
to manually manage memory.

See the documentation for more information.
