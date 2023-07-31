mutable struct Operator
    ref::RefValue{C.CeedOperator}
    qf::AbstractQFunction
    dqf::AbstractQFunction
    dqfT::AbstractQFunction
    sub_ops::Vector{Operator}
    function Operator(ref, qf, dqf, dqfT)
        obj = new(ref, qf, dqf, dqfT, [])
        finalizer(obj) do x
            # ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring), "Finalizing %s.\n", repr(x))
            destroy(x)
        end
        return obj
    end
end
destroy(op::Operator) = C.CeedOperatorDestroy(op.ref) # COV_EXCL_LINE
Base.getindex(op::Operator) = op.ref[]
Base.show(io::IO, ::MIME"text/plain", op::Operator) = ceed_show(io, op, C.CeedOperatorView)

"""
    Operator(ceed::Ceed; qf, dqf=QFunctionNone(), dqfT=QFunctionNone(), fields)

Creates a libCEED `CeedOperator` object using the given Q-function `qf`, and optionally its
derivative and derivative transpose.

An array of fields must be provided, where each element of the array is a tuple containing
the name of the field (as a string or symbol), the corresponding element restriction, basis,
and vector.

# Examples

Create the operator that builds the Q-data associated with the mass matrix.
```
build_oper = Operator(
    ceed,
    qf=build_qfunc,
    fields=[
        (:J, mesh_restr, mesh_basis, CeedVectorActive()),
        (:w, ElemRestrictionNone(), mesh_basis, CeedVectorNone()),
        (:qdata, sol_restr_i, BasisNone(), CeedVectorActive())
    ]
)
```
"""
function Operator(c::Ceed; qf, dqf=QFunctionNone(), dqfT=QFunctionNone(), fields)
    op = Operator(c, qf, dqf, dqfT)
    for f ∈ fields
        set_field!(op, String(f[1]), f[2], f[3], f[4])
    end
    op
end

function Operator(
    c::Ceed,
    qf::AbstractQFunction,
    dqf::AbstractQFunction,
    dqfT::AbstractQFunction,
)
    ref = Ref{C.CeedOperator}()
    C.CeedOperatorCreate(c[], qf[], dqf[], dqfT[], ref)
    Operator(ref, qf, dqf, dqfT)
end

"""
    create_composite_operator(c::Ceed, ops)

Create an [`Operator`](@ref) whose action represents the sum of the operators in the
collection `ops`.
"""
function create_composite_operator(c::Ceed, ops)
    ref = Ref{C.CeedOperator}()
    C.CeedCompositeOperatorCreate(c[], ref)
    comp_op = Operator(ref, QFunctionNone(), QFunctionNone(), QFunctionNone())
    comp_op.sub_ops = ops
    for op ∈ ops
        C.CeedCompositeOperatorAddSub(comp_op[], op[])
    end
    comp_op
end

function set_field!(
    op::Operator,
    fieldname::AbstractString,
    r::AbstractElemRestriction,
    b::AbstractBasis,
    v::AbstractCeedVector,
)
    C.CeedOperatorSetField(op[], fieldname, r[], b[], v[])
end

"""
    apply!(op::Operator, vin, vout; request=RequestImmediate())

Apply the action of the operator `op` to the input vector `vin`, and store the result in the
output vector `vout`.

For non-blocking application, the user can specify a request object. By default, immediate
(synchronous) completion is requested.
"""
function apply!(
    op::Operator,
    vin::AbstractCeedVector,
    vout::AbstractCeedVector;
    request=RequestImmediate(),
)
    C.CeedOperatorApply(op[], vin[], vout[], request[])
end

"""
    apply_add!(op::Operator, vin, vout; request=RequestImmediate())

Apply the action of the operator `op` to the input vector `vin`, and add the result to the
output vector `vout`.

For non-blocking application, the user can specify a request object. By default, immediate
(synchronous) completion is requested.
"""
function apply_add!(
    op::Operator,
    vin::AbstractCeedVector,
    vout::AbstractCeedVector;
    request=RequestImmediate(),
)
    C.CeedOperatorApplyAdd(op[], vin[], vout[], request[])
end

"""
    assemble_diagonal!(op::Operator, diag::CeedVector; request=RequestImmediate())

Overwrites a [`CeedVector`](@ref) with the diagonal of a linear [`Operator`](@ref).

!!! note "Note:"
    Currently only [`Operator`](@ref)s with a single field are supported.
"""
function assemble_diagonal!(op::Operator, diag::CeedVector; request=RequestImmediate())
    C.CeedOperatorLinearAssembleDiagonal(op[], diag[], request[])
end

"""
    assemble_diagonal!(op::Operator, diag::CeedVector; request=RequestImmediate())

Adds the diagonal of a linear [`Operator`](@ref) to the given [`CeedVector`](@ref).

!!! note "Note:"
    Currently only [`Operator`](@ref)s with a single field are supported.
"""
function assemble_add_diagonal!(op::Operator, diag::CeedVector; request=RequestImmediate())
    C.CeedOperatorLinearAssembleAddDiagonal(op[], diag[], request[])
end
