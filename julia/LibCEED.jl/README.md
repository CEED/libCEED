# LibCEED.jl: Julia Interface for [libCEED](https://github.com/CEED/libCEED)

Please see the [LibCEED.jl
documentation](http://ceed.exascaleproject.org/libCEED-julia-docs/dev/) for
usage and API documentation.

## Installation

The LibCEED.jl package can be installed with Julia's package manager by running
`] add LibCEED`. This will automatically install a pre-built binary of the
libCEED library. If you require features of a specific build of libCEED (e.g.
CUDA/GPU support, specific compiler flags, etc.) then you should compile your
own version of the libCEED library, and configure LibCEED.jl to use this binary
as described in the [Configuring LibCEED.jl](#configuring-libceedjl) section.

**Warning:** the pre-built libCEED binaries do not support CUDA backends

The pre-built binaries automatically installed by LibCEED.jl (through the
[libCEED_jll](https://juliahub.com/ui/Packages/libCEED_jll/LB2fn) package) are
not built with CUDA support. If you want to run libCEED on the GPU, you will
have to build libCEED from source and configure LibCEED.jl as described in the
[Configuring LibCEED.jl](#configuring-libceedjl) section.

### Configuring LibCEED.jl

By default, LibCEED.jl will use the pre-built libCEED binaries provided by the
[libCEED_jll](https://juliahub.com/ui/Packages/libCEED_jll/LB2fn) package. If
you wish to use a different libCEED binary (e.g. one built from source),
LibCEED.jl can be configured using the `JULIA_LIBCEED_LIB` environment variable
set to the absolute path of the libCEED dynamic library file (i.e. `libceed.so`,
and _not_ the enclosing directory). For the configuration to take effect,
LibCEED.jl must be **built** with this environment variable, for example:

```julia
% JULIA_LIBCEED_LIB=/path/to/libceed.so julia
julia> # press ] to enter package manager
(env) pkg> build LibCEED
```
or, equivalently,
```julia
julia> withenv("JULIA_LIBCEED_LIB" => "/path/to/libceed.so") do
    Pkg.build("LibCEED")
end
```
