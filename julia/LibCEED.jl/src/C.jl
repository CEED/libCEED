# Low-level C API for libCEED

module C

using CEnum, Libdl

# Get the path to the libCEED dynamic library, configured during the build step
# of the LibCEED.jl package.
const depsfile = joinpath(@__DIR__, "..", "deps", "deps.jl")
if !isfile(depsfile)
    error("LibCEED.jl not properly installed. Please run Pkg.build(\"LibCEED\")")
end
include(depsfile)

include(joinpath(@__DIR__, "generated", "libceed_common.jl"))
include(joinpath(@__DIR__, "generated", "libceed_api.jl"))

const CEED_STRIDES_BACKEND = Ref{Ptr{CeedInt}}()
const CEED_BASIS_COLLOCATED = Ref{CeedBasis}()
const CEED_VECTOR_ACTIVE = Ref{CeedVector}()
const CEED_VECTOR_NONE = Ref{CeedVector}()
const CEED_ELEMRESTRICTION_NONE = Ref{CeedElemRestriction}()
const CEED_QFUNCTION_NONE = Ref{CeedQFunction}()
const CEED_REQUEST_IMMEDIATE = Ref{CeedRequest}()
const CEED_REQUEST_ORDERED = Ref{CeedRequest}()

function __init__()
    global libceed_handle = dlopen(libceed)
    # some global variables
    CEED_STRIDES_BACKEND[] = cglobal((:CEED_STRIDES_BACKEND, libceed))
    CEED_BASIS_COLLOCATED[] =
        unsafe_load(cglobal((:CEED_BASIS_COLLOCATED, libceed), CeedBasis))
    CEED_VECTOR_ACTIVE[] = unsafe_load(cglobal((:CEED_VECTOR_ACTIVE, libceed), CeedVector))
    CEED_VECTOR_NONE[] = unsafe_load(cglobal((:CEED_VECTOR_NONE, libceed), CeedVector))
    CEED_ELEMRESTRICTION_NONE[] =
        unsafe_load(cglobal((:CEED_ELEMRESTRICTION_NONE, libceed), CeedElemRestriction))
    CEED_QFUNCTION_NONE[] =
        unsafe_load(cglobal((:CEED_QFUNCTION_NONE, libceed), CeedQFunction))
    CEED_REQUEST_IMMEDIATE[] =
        unsafe_load(cglobal((:CEED_REQUEST_IMMEDIATE, libceed), Ptr{CeedRequest}))
    CEED_REQUEST_ORDERED[] =
        unsafe_load(cglobal((:CEED_REQUEST_ORDERED, libceed), Ptr{CeedRequest}))
end

end # module
