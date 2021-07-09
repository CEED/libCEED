abstract type AbstractQFunction end

struct QFunctionNone <: AbstractQFunction end
Base.getindex(::QFunctionNone) = C.CEED_QFUNCTION_NONE[]

"""
    QFunction

A libCEED `CeedQFunction` object, typically created using the [`@interior_qf`](@ref) macro.

A `QFunction` can also be created from the "Q-function gallery" using
[`create_interior_qfunction`](@ref). The identity Q-function can be created using
[`create_identity_qfunction`](@ref).
"""
mutable struct QFunction <: AbstractQFunction
    ref::RefValue{C.CeedQFunction}
    user_qf::Union{Nothing,UserQFunction}
    ctx::Union{Nothing,Context}
    function QFunction(ref, user_qf)
        obj = new(ref, user_qf, nothing)
        finalizer(obj) do x
            # ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring), "Finalizing %s.\n", repr(x))
            destroy(x)
        end
        return obj
    end
end
QFunction(ref::Ref{C.CeedQFunction}) = QFunction(ref, nothing)
destroy(qf::QFunction) = C.CeedQFunctionDestroy(qf.ref) # COV_EXCL_LINE
Base.getindex(qf::QFunction) = qf.ref[]
Base.show(io::IO, ::MIME"text/plain", qf::QFunction) =
    ceed_show(io, qf, C.CeedQFunctionView)

function create_interior_qfunction(c::Ceed, f::UserQFunction; vlength=1)
    ref = Ref{C.CeedQFunction}()
    # Use empty string as source location to indicate to libCEED that there is
    # no C source for this Q-function
    C.CeedQFunctionCreateInterior(c[], vlength, f.fptr, "", ref)
    # COV_EXCL_START
    if !isnothing(f.cuf)
        C.CeedQFunctionSetCUDAUserFunction(ref[], f.cuf)
    elseif iscuda(c) && !isdefined(@__MODULE__, CUDA)
        error(
            string(
                "In order to use user Q-functions with a CUDA backend, the CUDA.jl package ",
                "must be loaded",
            ),
        )
    end
    # COV_EXCL_STOP
    QFunction(ref, f)
end

"""
    create_interior_qfunction(ceed::Ceed, name::AbstractString)

Create a [`QFunction`](@ref) from the Q-function gallery, using the provided name.

# Examples

- Build and apply the 3D mass operator
```
build_mass_qf = create_interior_qfunction(c, "Mass3DBuild")
apply_mass_qf = create_interior_qfunction(c, "MassApply")
```
- Build and apply the 3D Poisson operator
```
build_poi_qf = create_interior_qfunction(c, "Poisson3DBuild")
apply_poi_qf = create_interior_qfunction(c, "Poisson3DApply")
```
"""
function create_interior_qfunction(c::Ceed, name::AbstractString)
    ref = Ref{C.CeedQFunction}()
    C.CeedQFunctionCreateInteriorByName(c.ref[], name, ref)
    QFunction(ref)
end

"""
    create_identity_qfunction(c::Ceed, size, inmode::EvalMode, outmode::EvalMode)

Create an identity [`QFunction`](@ref). Inputs are written into outputs in the order given.
This is useful for [`Operators`](@ref Operator) that can be represented with only the action
of a [`ElemRestriction`](@ref) and [`Basis`](@ref), such as restriction and prolongation
operators for p-multigrid. Backends may optimize `CeedOperators` with this Q-function to
avoid the copy of input data to output fields by using the same memory location for both.
"""
function create_identity_qfunction(c::Ceed, size, inmode::EvalMode, outmode::EvalMode)
    ref = Ref{C.CeedQFunction}()
    C.CeedQFunctionCreateIdentity(c[], size, inmode, outmode, ref)
    QFunction(ref)
end

function add_input!(qf::AbstractQFunction, name::AbstractString, size, emode)
    C.CeedQFunctionAddInput(qf[], name, size, emode)
end

function add_output!(qf::AbstractQFunction, name::AbstractString, size, emode)
    C.CeedQFunctionAddOutput(qf[], name, size, emode)
end

"""
    set_context!(qf::QFunction, ctx::Context)

Associate a [`Context`](@ref) object `ctx` with the given Q-function `qf`.
"""
function set_context!(qf::QFunction, ctx)
    # Preserve the context data from the GC by storing a reference
    qf.ctx = ctx
    C.CeedQFunctionSetContext(qf[], ctx[])
end

function get_field_sizes(qf::AbstractQFunction)
    ninputs = Ref{CeedInt}()
    noutputs = Ref{CeedInt}()

    C.CeedQFunctionGetNumArgs(qf[], ninputs, noutputs)

    inputs = Ref{Ptr{C.CeedQFunctionField}}()
    outputs = Ref{Ptr{C.CeedQFunctionField}}()
    C.CeedQFunctionGetFields(qf[], inputs, outputs)

    input_sizes = zeros(CeedInt, ninputs[])
    output_sizes = zeros(CeedInt, noutputs[])

    for i = 1:ninputs[]
        field = unsafe_load(inputs[], i)
        C.CeedQFunctionFieldGetSize(field, pointer(input_sizes, i))
    end

    for i = 1:noutputs[]
        field = unsafe_load(outputs[], i)
        C.CeedQFunctionFieldGetSize(field, pointer(output_sizes, i))
    end

    input_sizes, output_sizes
end

"""
    apply!(qf::QFunction, Q, vin, vout)

Apply the action of a [`QFunction`](@ref) to an array of input vectors, and store the result
in an array of output vectors.
"""
function apply!(qf::QFunction, Q, vin, vout)
    vins = map(x -> x[], vin)
    vouts = map(x -> x[], vout)
    GC.@preserve vin vout begin
        C.CeedQFunctionApply(qf[], Q, pointer(vins), pointer(vouts))
    end
end
