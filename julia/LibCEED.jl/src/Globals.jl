"""
    CeedScalar

Scalar (floating point) type. Equivalent to `Float64`.
"""
const CeedScalar = C.CeedScalar

"""
    CeedInt

Integer type, used for indexing. Equivalent to `Int32`.
"""
const CeedInt = C.CeedInt

"""
    QuadMode

One of `GAUSS` or `GAUSS_LOBATTO`.
"""
const QuadMode = C.CeedQuadMode
const GAUSS = C.CEED_GAUSS
const GAUSS_LOBATTO = C.CEED_GAUSS_LOBATTO

"""
    MemType

One of `MEM_HOST` or `MEM_DEVICE`.
"""
const MemType = C.CeedMemType
const MEM_HOST = C.CEED_MEM_HOST
const MEM_DEVICE = C.CEED_MEM_DEVICE

"""
    CopyMode

One of `COPY_VALUES`, `USE_POINTER` or `OWN_POINTER`.

`OWN_POINTER` is not typically supported for objects created in Julia, because those must be
destroyed by the garbage collector, and cannot be freed from C.
"""
const CopyMode = C.CeedCopyMode
const COPY_VALUES = C.CEED_COPY_VALUES
const USE_POINTER = C.CEED_USE_POINTER
const OWN_POINTER = C.CEED_OWN_POINTER

"""
    EvalMode

Evaluation mode used in the specification of input and output fields for Q-functions, e.g.
in [`@interior_qf`](@ref).

One of:
- `EVAL_NONE`
- `EVAL_INTERP`
- `EVAL_GRAD`
- `EVAL_DIV`
- `EVAL_CURL`
- `EVAL_WEIGHT`

See the [libCEED
documentation](https://libceed.readthedocs.io/en/latest/api/CeedBasis/?highlight=EVAL_MODE#c.CeedEvalMode)
for further information.
"""
const EvalMode = C.CeedEvalMode
const EVAL_NONE = C.CEED_EVAL_NONE
const EVAL_INTERP = C.CEED_EVAL_INTERP
const EVAL_GRAD = C.CEED_EVAL_GRAD
const EVAL_DIV = C.CEED_EVAL_DIV
const EVAL_CURL = C.CEED_EVAL_CURL
const EVAL_WEIGHT = C.CEED_EVAL_WEIGHT

"""
    NormType

Denotes type of vector norm. One of `NORM_1`, `NORM_2`, or `NORM_MAX`.
"""
const NormType = C.CeedNormType
const NORM_1 = C.CEED_NORM_1
const NORM_2 = C.CEED_NORM_2
const NORM_MAX = C.CEED_NORM_MAX

"""
    TransposeMose

Denotes whether a linear transformation or its transpose should be applied. Either
`NOTRANSPOSE` or `TRANSPOSE`.
"""
const TransposeMode = C.CeedTransposeMode
const NOTRANSPOSE = C.CEED_NOTRANSPOSE
const TRANSPOSE = C.CEED_TRANSPOSE

"""
    Topology

Type of basis shape to create non-tensor H1 element basis. One of `LINE`, `TRIANGLE`,
`QUAD`, `TET`, `PYRAMID`, `PRISM`, or `HEX`.

The dimension can be extracted with bitshift:

    dim = Int(topology) >> 16
"""
const Topology = C.CeedElemTopology
const LINE = C.CEED_LINE
const TRIANGLE = C.CEED_TRIANGLE
const QUAD = C.CEED_QUAD
const TET = C.CEED_TET
const PYRAMID = C.CEED_PYRAMID
const PRISM = C.CEED_PRISM
const HEX = C.CEED_HEX

function set_globals()
    @doc """
        STRIDES_BACKEND

    Indicate that the stride is determined by the backend.
    """
    global STRIDES_BACKEND = C.CEED_STRIDES_BACKEND[]
end
