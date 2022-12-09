abstract type AbstractBasis end

"""
    BasisCollocated()

Returns the singleton object corresponding to libCEED's `CEED_BASIS_COLLOCATED`.
"""
struct BasisCollocated <: AbstractBasis end
Base.getindex(::BasisCollocated) = C.CEED_BASIS_COLLOCATED[]

"""
    Basis

Wraps a `CeedBasis` object, representing a finite element basis. A `Basis` object can be
created using one of:

- [`create_tensor_h1_lagrange_basis`](@ref)
- [`create_tensor_h1_basis`](@ref)
- [`create_h1_basis`](@ref)
"""
mutable struct Basis <: AbstractBasis
    ref::RefValue{C.CeedBasis}
    function Basis(ref)
        obj = new(ref)
        finalizer(obj) do x
            # ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring), "Finalizing %s.\n", repr(x))
            destroy(x)
        end
        return obj
    end
end
destroy(b::Basis) = C.CeedBasisDestroy(b.ref) # COV_EXCL_LINE
Base.getindex(b::Basis) = b.ref[]
Base.show(io::IO, ::MIME"text/plain", b::Basis) = ceed_show(io, b, C.CeedBasisView)

@doc raw"""
    create_tensor_h1_lagrange_basis(ceed, dim, ncomp, p, q, qmode)

Create a tensor-product Lagrange basis.

# Arguments:
- `ceed`:  A [`Ceed`](@ref) object where the [`Basis`](@ref) will be created.
- `dim`:   Topological dimension of element.
- `ncomp`: Number of field components (1 for scalar fields).
- `p`:     Number of Gauss-Lobatto nodes in one dimension.  The polynomial degree of the
           resulting $Q_k$ element is $k=p-1$.
- `q`:     Number of quadrature points in one dimension.
- `qmode`: Distribution of the $q$ quadrature points (affects order of accuracy for the
           quadrature).
"""
function create_tensor_h1_lagrange_basis(c::Ceed, dim, ncomp, p, q, quad_mode::QuadMode)
    ref = Ref{C.CeedBasis}()
    C.CeedBasisCreateTensorH1Lagrange(c[], dim, ncomp, p, q, quad_mode, ref)
    Basis(ref)
end

@doc raw"""
    create_tensor_h1_basis(c::Ceed, dim, ncomp, p, q, interp1d, grad1d, qref1d, qweight1d)

Create a tensor-product basis for $H^1$ discretizations.

# Arguments:
- `ceed`:      A [`Ceed`](@ref) object where the [`Basis`](@ref) will be created.
- `dim`:       Topological dimension.
- `ncomp`:     Number of field components (1 for scalar fields).
- `p`:         Number of nodes in one dimension.
- `q`:         Number of quadrature points in one dimension
- `interp1d`:  Matrix of size `(q, p)` expressing the values of nodal basis functions at
               quadrature points.
- `grad1d`:    Matrix of size `(p, q)` expressing derivatives of nodal basis functions at
               quadrature points.
- `qref1d`:    Array of length `q` holding the locations of quadrature points on the 1D
               reference element $[-1, 1]$.
- `qweight1d`: Array of length `q` holding the quadrature weights on the reference element.
"""
function create_tensor_h1_basis(
    c::Ceed,
    dim,
    ncomp,
    p,
    q,
    interp1d::AbstractArray{CeedScalar},
    grad1d::AbstractArray{CeedScalar},
    qref1d::AbstractArray{CeedScalar},
    qweight1d::AbstractArray{CeedScalar},
)
    @assert size(interp1d) == (q, p)
    @assert size(grad1d) == (q, p)
    @assert length(qref1d) == q
    @assert length(qweight1d) == q

    # Convert from Julia matrices (column-major) to row-major format
    interp1d_rowmajor = collect(interp1d')
    grad1d_rowmajor = collect(grad1d')

    ref = Ref{C.CeedBasis}()
    C.CeedBasisCreateTensorH1(
        c[],
        dim,
        ncomp,
        p,
        q,
        interp1d_rowmajor,
        grad1d_rowmajor,
        qref1d,
        qweight1d,
        ref,
    )
    Basis(ref)
end

@doc raw"""
    create_h1_basis(c::Ceed, topo::Topology, ncomp, nnodes, nqpts, interp, grad, qref, qweight)

Create a non tensor-product basis for H^1 discretizations

# Arguments:
- `ceed`:    A [`Ceed`](@ref) object where the [`Basis`](@ref) will be created.
- `topo`:    [`Topology`](@ref) of element, e.g. hypercube, simplex, etc.
- `ncomp`:   Number of field components (1 for scalar fields).
- `nnodes`:  Total number of nodes.
- `nqpts`:   Total number of quadrature points.
- `interp`:  Matrix of size `(nqpts, nnodes)` expressing the values of nodal basis functions
             at quadrature points.
- `grad`:    Array of size `(dim, nqpts, nnodes)` expressing derivatives of nodal basis
             functions at quadrature points.
- `qref`:    Array of length `nqpts` holding the locations of quadrature points on the
             reference element $[-1, 1]$.
- `qweight`: Array of length `nqpts` holding the quadrature weights on the reference
             element.
"""
function create_h1_basis(
    c::Ceed,
    topo::Topology,
    ncomp,
    nnodes,
    nqpts,
    interp::AbstractArray{CeedScalar},
    grad::AbstractArray{CeedScalar},
    qref::AbstractArray{CeedScalar},
    qweight::AbstractArray{CeedScalar},
)
    dim = getdimension(topo)
    @assert size(interp) == (nqpts, nnodes)
    @assert size(grad) == (dim, nqpts, nnodes)
    @assert length(qref) == nqpts
    @assert length(qweight) == nqpts

    # Convert from Julia matrices and tensors (column-major) to row-major format
    interp_rowmajor = collect(interp')
    grad_rowmajor = permutedims(grad, [3, 2, 1])

    ref = Ref{C.CeedBasis}()
    C.CeedBasisCreateH1(
        c[],
        topo,
        ncomp,
        nnodes,
        nqpts,
        interp_rowmajor,
        grad_rowmajor,
        qref,
        qweight,
        ref,
    )
    Basis(ref)
end

"""
    apply!(b::Basis, nelem, tmode::TransposeMode, emode::EvalMode, u::AbstractCeedVector, v::AbstractCeedVector)

Apply basis evaluation from nodes to quadrature points or vice versa, storing the result in
the [`CeedVector`](@ref) `v`.

`nelem` specifies the number of elements to apply the basis evaluation to; the backend will
specify the ordering in CeedElemRestrictionCreateBlocked()

Set `tmode` to `CEED_NOTRANSPOSE` to evaluate from nodes to quadrature or to
`CEED_TRANSPOSE` to apply the transpose, mapping from quadrature points to nodes.

Set the [`EvalMode`](@ref) `emode` to:
- `CEED_EVAL_NONE` to use values directly,
- `CEED_EVAL_INTERP` to use interpolated values,
- `CEED_EVAL_GRAD` to use gradients,
- `CEED_EVAL_WEIGHT` to use quadrature weights.
"""
function apply!(
    b::Basis,
    nelem,
    tmode::TransposeMode,
    emode::EvalMode,
    u::AbstractCeedVector,
    v::AbstractCeedVector,
)
    C.CeedBasisApply(b[], nelem, tmode, emode, u[], v[])
end

"""
    apply(b::Basis, u::AbstractVector; nelem=1, tmode=NOTRANSPOSE, emode=EVAL_INTERP)

Performs the same function as the above-defined [`apply!`](@ref apply!(b::Basis, nelem,
tmode::TransposeMode, emode::EvalMode, u::AbstractCeedVector, v::AbstractCeedVector)), but
automatically convert from Julia arrays to [`CeedVector`](@ref) for convenience.

The result will be returned in a newly allocated array of the correct size.
"""
function apply(b::Basis, u::AbstractVector; nelem=1, tmode=NOTRANSPOSE, emode=EVAL_INTERP)
    ceed_ref = Ref{C.Ceed}()
    ccall((:CeedBasisGetCeed, C.libceed), Cint, (C.CeedBasis, Ptr{C.Ceed}), b[], ceed_ref)
    c = Ceed(ceed_ref)

    u_vec = CeedVector(c, u)

    len_v = (tmode == TRANSPOSE) ? getnumnodes(b) : getnumqpts(b)
    if emode == EVAL_GRAD
        len_v *= getdimension(b)
    end

    v_vec = CeedVector(c, len_v)

    apply!(b, nelem, tmode, emode, u_vec, v_vec)
    Vector(v_vec)
end

"""
    getdimension(b::Basis)

Return the spatial dimension of the given [`Basis`](@ref).
"""
function getdimension(b::Basis)
    dim = Ref{CeedInt}()
    C.CeedBasisGetDimension(b[], dim)
    dim[]
end

"""
    getdimension(t::Topology)

Return the spatial dimension of the given [`Topology`](@ref).
"""
function getdimension(t::Topology)
    return Int(t) >> 16
end

"""
    gettopology(b::Basis)

Return the [`Topology`](@ref) of the given [`Basis`](@ref).
"""
function gettopology(b::Basis)
    topo = Ref{Topology}()
    C.CeedBasisGetTopology(b[], topo)
    topo[]
end

"""
    getnumcomponents(b::Basis)

Return the number of components of the given [`Basis`](@ref).
"""
function getnumcomponents(b::Basis)
    ncomp = Ref{CeedInt}()
    C.CeedBasisGetNumComponents(b[], ncomp)
    ncomp[]
end

"""
    getnumnodes(b::Basis)

Return the number of nodes of the given [`Basis`](@ref).
"""
function getnumnodes(b::Basis)
    nnodes = Ref{CeedInt}()
    C.CeedBasisGetNumNodes(b[], nnodes)
    nnodes[]
end

"""
    getnumnodes1d(b::Basis)

    Return the number of 1D nodes of the given (tensor-product) [`Basis`](@ref).
"""
function getnumnodes1d(b::Basis)
    nnodes1d = Ref{CeedInt}()
    C.CeedBasisGetNumNodes1D(b[], nnodes1d)
    nnodes1d[]
end

"""
    getnumqpts(b::Basis)

Return the number of quadrature points of the given [`Basis`](@ref).
"""
function getnumqpts(b::Basis)
    nqpts = Ref{CeedInt}()
    C.CeedBasisGetNumQuadraturePoints(b[], nqpts)
    nqpts[]
end

"""
    getnumqpts1d(b::Basis)

Return the number of 1D quadrature points of the given (tensor-product) [`Basis`](@ref).
"""
function getnumqpts1d(b::Basis)
    nqpts1d = Ref{CeedInt}()
    C.CeedBasisGetNumQuadraturePoints1D(b[], nqpts1d)
    nqpts1d[]
end

"""
    getqref(b::Basis)

Get the reference coordinates of quadrature points (in `dim` dimensions) of the given
[`Basis`](@ref).
"""
function getqref(b::Basis)
    istensor = Ref{Bool}()
    C.CeedBasisIsTensor(b[], istensor)
    ref = Ref{Ptr{CeedScalar}}()
    C.CeedBasisGetQRef(b[], ref)
    copy(
        unsafe_wrap(
            Array,
            ref[],
            istensor[] ? getnumqpts1d(b) : (getnumqpts(b)*getdimension(b)),
        ),
    )
end

"""
    getqref(b::Basis)

Get the quadrature weights of quadrature points (in `dim` dimensions) of the given
[`Basis`](@ref).
"""
function getqweights(b::Basis)
    istensor = Ref{Bool}()
    C.CeedBasisIsTensor(b[], istensor)
    ref = Ref{Ptr{CeedScalar}}()
    C.CeedBasisGetQWeights(b[], ref)
    copy(unsafe_wrap(Array, ref[], istensor[] ? getnumqpts1d(b) : getnumqpts(b)))
end

"""
    getinterp(b::Basis)

Get the interpolation matrix of the given [`Basis`](@ref). Returns a matrix of size
`(getnumqpts(b), getnumnodes(b))`.
"""
function getinterp(b::Basis)
    ref = Ref{Ptr{CeedScalar}}()
    C.CeedBasisGetInterp(b[], ref)
    q = getnumqpts(b)
    p = getnumnodes(b)
    collect(unsafe_wrap(Array, ref[], (p, q))')
end

"""
    getinterp1d(b::Basis)

Get the 1D interpolation matrix of the given [`Basis`](@ref). `b` must be a tensor-product
basis, otherwise this function will fail. Returns a matrix of size `(getnumqpts1d(b),
getnumnodes1d(b))`.
"""
function getinterp1d(b::Basis)
    ref = Ref{Ptr{CeedScalar}}()
    C.CeedBasisGetInterp1D(b[], ref)
    q = getnumqpts1d(b)
    p = getnumnodes1d(b)
    collect(unsafe_wrap(Array, ref[], (p, q))')
end

"""
    getgrad(b::Basis)

Get the gradient matrix of the given [`Basis`](@ref). Returns a tensor of size
`(getdimension(b), getnumqpts(b), getnumnodes(b))`.
"""
function getgrad(b::Basis)
    ref = Ref{Ptr{CeedScalar}}()
    C.CeedBasisGetGrad(b[], ref)
    dim = getdimension(b)
    q = getnumqpts(b)
    p = getnumnodes(b)
    permutedims(unsafe_wrap(Array, ref[], (p, q, dim)), [3, 2, 1])
end

"""
    getgrad1d(b::Basis)

Get the 1D derivative matrix of the given [`Basis`](@ref). Returns a matrix of size
`(getnumqpts(b), getnumnodes(b))`.
"""
function getgrad1d(b::Basis)
    ref = Ref{Ptr{CeedScalar}}()
    C.CeedBasisGetGrad1D(b[], ref)
    q = getnumqpts1d(b)
    p = getnumnodes1d(b)
    collect(unsafe_wrap(Array, ref[], (p, q))')
end
