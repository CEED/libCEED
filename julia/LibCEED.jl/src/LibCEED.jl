module LibCEED

using StaticArrays, UnsafeArrays, Requires, Preferences
using Base: RefValue

# import low-level C interface
include("C.jl")
using .C

export @interior_qf,
    @witharray,
    @witharray_read,
    Abscissa,
    AbscissaAndWeights,
    Basis,
    BasisCollocated,
    COPY_VALUES,
    Ceed,
    CeedDim,
    CeedInt,
    CeedScalar,
    CeedVector,
    CeedVectorActive,
    CeedVectorNone,
    Context,
    CopyMode,
    EVAL_CURL,
    EVAL_DIV,
    EVAL_GRAD,
    EVAL_INTERP,
    EVAL_NONE,
    EVAL_WEIGHT,
    ElemRestriction,
    ElemRestrictionNone,
    EvalMode,
    GAUSS,
    GAUSS_LOBATTO,
    HEX,
    LINE,
    MEM_DEVICE,
    MEM_HOST,
    MemType,
    NORM_1,
    NORM_2,
    NORM_MAX,
    NOTRANSPOSE,
    NormType,
    OWN_POINTER,
    Operator,
    PRISM,
    PYRAMID,
    QFunction,
    QFunctionNone,
    QUAD,
    QuadMode,
    RequestImmediate,
    RequestOrdered,
    STRIDES_BACKEND,
    TET,
    TRANSPOSE,
    TRIANGLE,
    Topology,
    TransposeMode,
    USE_POINTER,
    UserQFunction,
    add_input!,
    add_output!,
    apply!,
    apply_add!,
    apply,
    assemble,
    assemble_add_diagonal!,
    assemble_diagonal!,
    axpy!,
    ceedversion,
    create_composite_operator,
    create_elem_restriction,
    create_elem_restriction_strided,
    create_evector,
    create_h1_basis,
    create_identity_qfunction,
    create_interior_qfunction,
    create_lvector,
    create_tensor_h1_basis,
    create_tensor_h1_lagrange_basis,
    create_vectors,
    det,
    extract_array,
    extract_context,
    gauss_quadrature,
    get_libceed_path,
    get_preferred_memtype,
    get_scalar_type,
    getcompstride,
    getnumelements,
    getelementsize,
    getlvectorsize,
    getmultiplicity!,
    getmultiplicity,
    getdimension,
    getgrad,
    getgrad1d,
    getinterp,
    getinterp1d,
    getnumcomponents,
    getnumnodes,
    getnumnodes1d,
    getnumqpts,
    getnumqpts1d,
    getqref,
    getqweights,
    getresource,
    gettopology,
    getvoigt!,
    getvoigt,
    iscuda,
    isdeterministic,
    isrelease,
    lobatto_quadrature,
    norm,
    pointwisemult!,
    reciprocal!,
    scale!,
    set_context!,
    set_cufunction!,
    set_data!,
    set_field!,
    set_libceed_path!,
    setarray!,
    setvalue!,
    setvoigt!,
    setvoigt,
    syncarray!,
    takearray!,
    use_prebuilt_libceed!,
    witharray,
    witharray_read

include("Globals.jl")
include("Ceed.jl")
include("CeedVector.jl")
include("Basis.jl")
include("ElemRestriction.jl")
include("Quadrature.jl")
include("Context.jl")
include("UserQFunction.jl")
include("QFunction.jl")
include("Request.jl")
include("Operator.jl")
include("Misc.jl")

const minimum_libceed_version = v"0.10.0"

function __init__()
    if !ceedversion_ge(minimum_libceed_version)
        @warn("""
              Incompatible libCEED version.
              LibCEED.jl requires libCEED version at least $minimum_libceed_version.
              The version of the libCEED library is $(ceedversion())."
              """)
    end
    configure_scalar_type(C.libceed_handle)
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("Cuda.jl")
    set_globals()
end

"""
    ceedversion()

Returns a `VersionNumber` corresponding to the version of the libCEED library currently used.
"""
function ceedversion()
    major = Ref{Cint}()
    minor = Ref{Cint}()
    patch = Ref{Cint}()
    release = Ref{Bool}()
    C.CeedGetVersion(major, minor, patch, release)
    return VersionNumber(major[], minor[], patch[])
end

"""
    isrelease()

Returns true if the libCEED library is a release build, false otherwise.
"""
function isrelease()
    major = Ref{Cint}()
    minor = Ref{Cint}()
    patch = Ref{Cint}()
    release = Ref{Bool}()
    C.CeedGetVersion(major, minor, patch, release)
    return release[]
end

"""
    ceedversion_ge(version::VersionNumber)

Returns true if the libCEED library is at least as current as the specified `version`. Returns
`true` whenever libCEED is not a release build.
"""
ceedversion_ge(version::VersionNumber) = !isrelease() || (ceedversion() >= version)

"""
    set_libceed_path!(path::AbstractString)
    set_libceed_path!(:prebuilt)
    set_libceed_path!(:default)

Sets the path of the libCEED dynamic library. `path` should be the absolute path to the library
file.

`set_libceed_path!(:prebuilt)` indicates to LibCEED.jl to use the prebuilt version of the libCEED
library (bundled with libCEED\\_jll).

`set_libceed_path!(:default)` indicates to LibCEED.jl to use the default library. This usually has
the same effect as `set_libceed_path!(:prebuilt)`, unless a different path has been specified in the
depot-wide preferences or using
[Overrides.toml](https://pkgdocs.julialang.org/dev/artifacts/#Overriding-artifact-locations).

This function sets the library path as a _preference_ associated with the currently active
environment. Changes will take effect after restarting the Julia session. See the [Preferences.jl
documentation](https://github.com/JuliaPackaging/Preferences.jl) for more information.
"""
function set_libceed_path!(path::AbstractString)
    handle = C.dlopen(path)
    set_preferences!(C.libCEED_jll, "libceed_path" => path; force=true)
    @info("""
            Setting the libCEED library path to $path.
            Restart the Julia session for changes to take effect.
            """)
    configure_scalar_type(handle)
    (handle != C.libceed_handle) && C.dlclose(handle)
    return
end

function set_libceed_path!(sym::Symbol)
    if sym == :prebuilt
        set_preferences!(C.libCEED_jll, "libceed_path" => nothing; force=true)
        @info("""
              Using the prebuilt libCEED binary.
              Restart the Julia session for changes to take effect.
              """)
    elseif sym == :default
        delete_preferences!(Preferences.get_uuid(C.libCEED_jll), "libceed_path"; force=true)
        @info("""
              Deleting preference for libCEED library path.
              Restart the Julia session for changes to take effect.
              """)
    else
        error("set_libceed_path(::Symbol) must be called with :prebuilt or :default.")
    end
end

"""
    use_prebuilt_libceed!()

Indicates that the prebuilt version of the libCEED library (bundled with libCEED\\_jll) should be
used.

Equivalent to `set_libceed_path!(:prebuilt)`.
"""
use_prebuilt_libceed!() = set_libceed_path!(:prebuilt)

"""
    get_libceed_path()

Returns the path to the currently used libCEED library. A different libCEED library can be used by
calling [`set_libceed_path!`](@ref) or by using a depot-wide Overrides.toml file.
"""
get_libceed_path() = C.libCEED_jll.libceed_path

"""
    get_scalar_type()

Return the type of `CeedScalar` used by the libCEED library (either `Float32` or `Float64`).
"""
get_scalar_type() = get_scalar_type(LibCEED.C.libceed_handle)

function get_scalar_type(handle::Ptr{Nothing})
    # If CeedGetScalarType is not provided by the libCEED shared library, default for Float64
    sym = LibCEED.C.dlsym(handle, :CeedGetScalarType; throw_error=false)
    if sym === nothing
        return Float64
    else
        scalar_type = Ref{C.CeedScalarType}()
        ccall(sym, Cint, (Ptr{C.CeedScalarType},), scalar_type)
        if scalar_type[] == C.CEED_SCALAR_FP32
            return Float32
        elseif scalar_type[] == C.CEED_SCALAR_FP64
            return Float64
        else
            error("Unknown CeedScalar type $(scalar_type[])")
        end
    end
end

function configure_scalar_type(handle::Ptr{Nothing})
    scalar_type = get_scalar_type(handle)
    if scalar_type != CeedScalar
        @set_preferences!("CeedScalar" => string(scalar_type))
        @warn("""
              libCEED is compiled with $scalar_type but LibCEED.jl is using $CeedScalar.
              Configuring LibCEED.jl to use $scalar_type.
              The Julia session must be restarted for changes to take effect.
              """)
    end
end

end # module
