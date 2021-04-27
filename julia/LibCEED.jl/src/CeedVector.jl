import LinearAlgebra: norm, axpy!

abstract type AbstractCeedVector end

struct CeedVectorActive <: AbstractCeedVector end
Base.getindex(::CeedVectorActive) = C.CEED_VECTOR_ACTIVE[]

struct CeedVectorNone <: AbstractCeedVector end
Base.getindex(::CeedVectorNone) = C.CEED_VECTOR_NONE[]

mutable struct CeedVector <: AbstractCeedVector
    ref::RefValue{C.CeedVector}
    arr::Union{Nothing,AbstractArray}
    CeedVector(ref::Ref{C.CeedVector}) = new(ref, nothing)
end

"""
    CeedVector(c::Ceed, len::Integer)

Creates a `CeedVector` of given length.
"""
function CeedVector(c::Ceed, len::Integer)
    ref = Ref{C.CeedVector}()
    C.CeedVectorCreate(c[], len, ref)
    obj = CeedVector(ref)
    finalizer(obj) do x
        # ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring), "Finalizing %s.\n", repr(x))
        destroy(x)
    end
    return obj
end
destroy(v::CeedVector) = C.CeedVectorDestroy(v.ref) # COV_EXCL_LINE
Base.getindex(v::CeedVector) = v.ref[]

Base.summary(io::IO, v::CeedVector) = print(io, length(v), "-element CeedVector")
function Base.show(io::IO, ::MIME"text/plain", v::CeedVector)
    summary(io, v)
    println(io, ":")
    witharray_read(v, MEM_HOST) do arr
        Base.print_array(io, arr)
    end
end
Base.show(io::IO, v::CeedVector) = witharray_read(a -> show(io, a), v, MEM_HOST)

function Base.length(::Type{T}, v::CeedVector) where {T}
    len = Ref{C.CeedInt}()
    C.CeedVectorGetLength(v[], len)
    return T(len[])
end

Base.ndims(::CeedVector) = 1
Base.ndims(::Type{CeedVector}) = 1
Base.axes(v::CeedVector) = (Base.OneTo(length(v)),)
Base.size(v::CeedVector) = (length(Int, v),)
Base.length(v::CeedVector) = length(Int, v)

"""
    setvalue!(v::CeedVector, val::Real)

Set the [`CeedVector`](@ref) to a constant value.
"""
setvalue!(v::CeedVector, val::Real) = C.CeedVectorSetValue(v[], val)
"""
    setindex!(v::CeedVector, val::Real)
    v[] = val

Set the [`CeedVector`](@ref) to a constant value, synonymous to [`setvalue!`](@ref).
"""
Base.setindex!(v::CeedVector, val::Real) = setvalue!(v, val)

"""
    norm(v::CeedVector, ntype::NormType)

Return the norm of the given [`CeedVector`](@ref).

The norm type can either be specified as one of `NORM_1`, `NORM_2`, `NORM_MAX`.
"""
function norm(v::CeedVector, ntype::NormType)
    nrm = Ref{CeedScalar}()
    C.CeedVectorNorm(v[], ntype, nrm)
    nrm[]
end

"""
    norm(v::CeedVector, p::Real)

Return the norm of the given [`CeedVector`](@ref), see [`norm(::CeedVector,
::NormType)`](@ref).

`p` can have value 1, 2, or Inf, corresponding to `NORM_1`, `NORM_2`, and `NORM_MAX`,
respectively.
"""
function norm(v::CeedVector, p::Real)
    if p == 1
        ntype = NORM_1
    elseif p == 2
        ntype = NORM_2
    elseif isinf(p)
        ntype = NORM_MAX
    else
        error("norm(v::CeedVector, p): p must be 1, 2, or Inf")
    end
    norm(v, ntype)
end

"""
    reciprocal!(v::CeedVector)

Set `v` to be equal to its elementwise reciprocal.
"""
reciprocal!(v::CeedVector) = C.CeedVectorReciprocal(v[])

"""
    setarray!(v::CeedVector, mtype::MemType, cmode::CopyMode, arr)

Set the array used by a [`CeedVector`](@ref), freeing any previously allocated array if
applicable. The backend may copy values to a different [`MemType`](@ref). See also
[`syncarray!`](@ref) and [`takearray!`](@ref).

!!! warning "Avoid OWN_POINTER CopyMode"
    The [`CopyMode`](@ref) `OWN_POINTER` is not suitable for use with arrays that are
    allocated by Julia, since those cannot be properly freed from libCEED.
"""
function setarray!(v::CeedVector, mtype::MemType, cmode::CopyMode, arr)
    C.CeedVectorSetArray(v[], mtype, cmode, arr)
    if cmode == USE_POINTER
        v.arr = arr
    end
end

"""
    syncarray!(v::CeedVector, mtype::MemType)

Sync the [`CeedVector`](@ref) to a specified [`MemType`](@ref). This function is used to
force synchronization of arrays set with [`setarray!`](@ref). If the requested memtype is
already synchronized, this function results in a no-op.
"""
syncarray!(v::CeedVector, mtype::MemType) = C.CeedVectorSyncArray(v[], mtype)

"""
    takearray!(v::CeedVector, mtype::MemType)

Take ownership of the [`CeedVector`](@ref) array and remove the array from the
[`CeedVector`](@ref). The caller is responsible for managing and freeing the array. The
array is returns as a `Ptr{CeedScalar}`.
"""
function takearray!(v::CeedVector, mtype::MemType)
    ptr = Ref{Ptr{CeedScalar}}()
    C.CeedVectorTakeArray(v[], mtype, ptr)
    v.arr = nothing
    ptr[]
end

# Helper function to parse arguments of @witharray and @witharray_read
function witharray_parse(assignment, args)
    if !Meta.isexpr(assignment, :(=))
        error("@witharray must have first argument of the form v_arr=v") # COV_EXCL_LINE
    end
    arr = assignment.args[1]
    v = assignment.args[2]
    mtype = MEM_HOST
    sz = :((length($(esc(v))),))
    body = args[end]
    for i = 1:length(args)-1
        a = args[i]
        if !Meta.isexpr(a, :(=))
            error("Incorrect call to @witharray or @witharray_read") # COV_EXCL_LINE
        end
        if a.args[1] == :mtype
            mtype = a.args[2]
        elseif a.args[1] == :size
            sz = esc(a.args[2])
        end
    end
    arr, v, sz, mtype, body
end

"""
    @witharray(v_arr=v, [size=(dims...)], [mtype=MEM_HOST], body)

Executes `body`, having extracted the contents of the [`CeedVector`](@ref) `v` as an array
with name `v_arr`. If the [`memory type`](@ref MemType) `mtype` is not provided, `MEM_HOST`
will be used. If the size is not specified, a flat vector will be assumed.

# Examples
Negate the contents of `CeedVector` `v`:
```
@witharray v_arr=v v_arr .*= -1.0
```
"""
macro witharray(assignment, args...)
    arr, v, sz, mtype, body = witharray_parse(assignment, args)
    quote
        arr_ref = Ref{Ptr{C.CeedScalar}}()
        C.CeedVectorGetArray($(esc(v))[], $(esc(mtype)), arr_ref)
        try
            $(esc(arr)) = UnsafeArray(arr_ref[], Int.($sz))
            $(esc(body))
        finally
            C.CeedVectorRestoreArray($(esc(v))[], arr_ref)
        end
    end
end

"""
    @witharray_read(v_arr=v, [size=(dims...)], [mtype=MEM_HOST], body)

Same as [`@witharray`](@ref), but provides read-only access to the data.
"""
macro witharray_read(assignment, args...)
    arr, v, sz, mtype, body = witharray_parse(assignment, args)
    quote
        arr_ref = Ref{Ptr{C.CeedScalar}}()
        C.CeedVectorGetArrayRead($(esc(v))[], $(esc(mtype)), arr_ref)
        try
            $(esc(arr)) = UnsafeArray(arr_ref[], Int.($sz))
            $(esc(body))
        finally
            C.CeedVectorRestoreArrayRead($(esc(v))[], arr_ref)
        end
    end
end

"""
    setindex!(v::CeedVector, v2::AbstractArray)
    v[] = v2

Sets the values of [`CeedVector`](@ref) `v` equal to those of `v2` using broadcasting.
"""
Base.setindex!(v::CeedVector, v2::AbstractArray) = @witharray(a = v, a .= v2)

"""
    CeedVector(c::Ceed, v2::AbstractVector; mtype=MEM_HOST, cmode=COPY_VALUES)

Creates a new [`CeedVector`](@ref) using the contents of the given vector `v2`. By default,
the contents of `v2` will be copied to the new [`CeedVector`](@ref), but this behavior can
be changed by specifying a different `cmode`.
"""
function CeedVector(c::Ceed, v2::AbstractVector; mtype=MEM_HOST, cmode=COPY_VALUES)
    v = CeedVector(c, length(v2))
    setarray!(v, mtype, cmode, v2)
    v
end

"""
    Vector(v::CeedVector)

Create a new `Vector` by copying the contents of `v`.
"""
function Base.Vector(v::CeedVector)
    v2 = Vector{CeedScalar}(undef, length(v))
    @witharray_read(a = v, v2 .= a)
end

"""
    witharray(f, v::CeedVector, mtype=MEM_HOST)

Calls `f` with an array containing the data of the `CeedVector` `v`, using [`memory
type`](@ref MemType) `mtype`.

Because of performance issues involving closures, if `f` is a complex operation, it may be
more efficient to use the macro version `@witharray` (cf. the section on "Performance of
captured variable" in the [Julia
documentation](https://docs.julialang.org/en/v1/manual/performance-tips) and related [GitHub
issue](https://github.com/JuliaLang/julia/issues/15276).

# Examples

Return the sum of a vector:
```
witharray(sum, v)
```
"""
function witharray(f, v::CeedVector, mtype::MemType=MEM_HOST)
    arr_ref = Ref{Ptr{C.CeedScalar}}()
    C.CeedVectorGetArray(v[], mtype, arr_ref)
    arr = UnsafeArray(arr_ref[], (length(v),))
    res = try
        f(arr)
    finally
        C.CeedVectorRestoreArray(v[], arr_ref)
    end
    return res
end

"""
    witharray_read(f, v::CeedVector, mtype::MemType=MEM_HOST)

Same as [`witharray`](@ref), but with read-only access to the data.

# Examples

Display the contents of a vector:
```
witharray_read(display, v)
```
"""
function witharray_read(f, v::CeedVector, mtype::MemType=MEM_HOST)
    arr_ref = Ref{Ptr{C.CeedScalar}}()
    C.CeedVectorGetArrayRead(v[], mtype, arr_ref)
    arr = UnsafeArray(arr_ref[], (length(v),))
    res = try
        f(arr)
    finally
        C.CeedVectorRestoreArrayRead(v[], arr_ref)
    end
    return res
end

"""
    scale!(v::CeedVector, a::Real)

Overwrite `v` with `a*v` for scalar `a`. Returns `v`.
"""
function scale!(v::CeedVector, a::Real)
    C.CeedVectorScale(v[], a)
    return v
end

"""
    axpy!(a::Real, x::CeedVector, y::CeedVector)

Overwrite `y` with `x*a + y`, where `a` is a scalar. Returns `y`.

!!! warning "Different argument order"
    In order to be consistent with `LinearAlgebra.axpy!`, the arguments are passed in order: `a`,
    `x`, `y`. This is different than the order of arguments of the C function `CeedVectorAXPY`.
"""
function axpy!(a::Real, x::CeedVector, y::CeedVector)
    C.CeedVectorAXPY(y[], a, x[])
    return y
end

"""
    pointwisemult!(w::CeedVector, x::CeedVector, y::CeedVector)

Overwrite `w` with `x .* y`. Any subset of x, y, and w may be the same vector. Returns `w`.
"""
function pointwisemult!(w::CeedVector, x::CeedVector, y::CeedVector)
    C.CeedVectorPointwiseMult(w[], x[], y[])
    return w
end
