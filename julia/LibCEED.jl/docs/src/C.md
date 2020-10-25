# Low-level C interface

The low-level interface (provided in the `LibCEED.C` module) is in one-to-one
correspondence with the C libCEED iterface, and is automatically generated (with
some minor manual modifications) using the Julia package
[Clang.jl](https://github.com/JuliaInterop/Clang.jl/). The code used to generate
bindings is available in `generate_bindings.jl`.

With the low-level interface, the user is responsible for freeing all allocated
memory (calling the appropriate `Ceed*Destroy` functions). This interface is not
type-safe, and calling functions with the wrong arguments can cause libCEED to
crash.

It is generally recommended for users to use the Julia interface exported from
the `LibCEED` module, unless other specific low-level functionality is required.
