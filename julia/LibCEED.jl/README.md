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
LibCEED.jl can be configured using Julia's _preferences_ mechanism. Note that
this preference will be set for the currently active Julia environment, and can
be different between different environments. The Julia session must be restarted
for changes to take effect.

```julia
julia> using LibCEED
julia> set_libceed_path!("/path/to/libceed.so")
[ Info: Setting the libCEED library path to /path/to/libceed.so.
[ Info: Restart the Julia session for changes to take effect.
```

See [Preferences.jl](https://github.com/JuliaPackaging/Preferences.jl) for more
information.
