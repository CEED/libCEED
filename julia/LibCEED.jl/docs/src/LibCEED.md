# Library configuration

By default, LibCEED.jl uses a "basic version" of the libCEED library that is
bundled as a pre-built binary. In order to access more advanced features (CUDA
support, architecture-specific compiler flags, etc.), users can use LibCEED.jl
with a other versions of the libCEED library (e.g. compiled from source).

This is achieved by by calling [`set_libceed_path!`](@ref) with the path to the
library file. The choice of library file is stored as a per-environment
preference. For changes to take effect, the Julia session must be restarted. The
library currently being used by LibCEED.jl can be queried using
[`get_libceed_path`](@ref).

The version number of the currently loaded libCEED library can also be queried
using [`ceedversion`](@ref).

```@docs
ceedversion
isrelease
set_libceed_path!
use_prebuilt_libceed!
get_libceed_path
```
