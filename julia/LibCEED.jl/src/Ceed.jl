struct CeedError <: Exception
    fname::String
    lineno::Int
    func::String
    ecode::Int
    message::String
end

# COV_EXCL_START
function Base.showerror(io::IO, e::CeedError)
    println(io, "libCEED error code ", e.ecode, " in ", e.func)
    println(io, e.fname, ':', e.lineno, '\n')
    println(io, e.message)
end
# COV_EXCL_STOP

function handle_ceed_error(
    ceed::C.Ceed,
    c_fname::Cstring,
    lineno::Cint,
    c_func::Cstring,
    ecode::Cint,
    c_format::Cstring,
    args::Ptr{Cvoid},
)
    c_message = ccall(
        (:CeedErrorFormat, C.libceed),
        Cstring,
        (C.Ceed, Cstring, Ptr{Cvoid}),
        ceed,
        c_format,
        args,
    )
    fname = unsafe_string(c_fname)
    func = unsafe_string(c_func)
    message = unsafe_string(c_message)
    throw(CeedError(fname, lineno, func, ecode, message))
end

mutable struct Ceed
    ref::RefValue{C.Ceed}
end

"""
    Ceed(spec="/cpu/self")

Wraps a libCEED `Ceed` object, created with the given resource specification string.
"""
function Ceed(spec::AbstractString="/cpu/self")
    obj = Ceed(Ref{C.Ceed}())
    C.CeedInit(spec, obj.ref)
    ehandler = @cfunction(
        handle_ceed_error,
        Cint,
        (C.Ceed, Cstring, Cint, Cstring, Cint, Cstring, Ptr{Cvoid})
    )
    C.CeedSetErrorHandler(obj.ref[], ehandler)
    finalizer(obj) do x
        # ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring), "Finalizing %s.\n", repr(x))
        destroy(x)
    end
    return obj
end
destroy(c::Ceed) = C.CeedDestroy(c.ref) # COV_EXCL_LINE
Base.getindex(c::Ceed) = c.ref[]

Base.show(io::IO, ::MIME"text/plain", c::Ceed) = ceed_show(io, c, C.CeedView)

"""
    getresource(c::Ceed)

Returns the resource string associated with the given [`Ceed`](@ref) object.
"""
function getresource(c::Ceed)
    res = Ref{Cstring}()
    C.CeedGetResource(c[], res)
    unsafe_string(res[])
end

"""
    isdeterministic(c::Ceed)

Returns true if backend of the given [`Ceed`](@ref) object is deterministic, and false
otherwise.
"""
function isdeterministic(c::Ceed)
    isdet = Ref{Bool}()
    C.CeedIsDeterministic(c[], isdet)
    isdet[]
end

"""
    get_preferred_memtype(c::Ceed)

Returns the preferred [`MemType`](@ref) (either `MEM_HOST` or `MEM_DEVICE`) of the given
[`Ceed`](@ref) object.
"""
function get_preferred_memtype(c::Ceed)
    mtype = Ref{MemType}()
    C.CeedGetPreferredMemType(c[], mtype)
    mtype[]
end

"""
    iscuda(c::Ceed)

Returns true if the given [`Ceed`](@ref) object has resource `"/gpu/cuda/*"` and false
otherwise.
"""
function iscuda(c::Ceed)
    res_split = split(getresource(c), "/")
    length(res_split) >= 3 && res_split[3] == "cuda"
end
