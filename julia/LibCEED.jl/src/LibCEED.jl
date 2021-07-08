module LibCEED

using StaticArrays, UnsafeArrays, Requires
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
    configure_ceed_scalar!,
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
    get_preferred_memtype,
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
    setarray!,
    setvalue!,
    setvoigt!,
    setvoigt,
    syncarray!,
    takearray!,
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

cuda_is_loaded = false

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("Cuda.jl")
    set_globals()
end

end # module
