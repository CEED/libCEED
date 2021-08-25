using CEnum

#! format: off


const CeedInt = Int32

@cenum CeedScalarType::UInt32 begin
    CEED_SCALAR_FP32 = 0
    CEED_SCALAR_FP64 = 1
end

mutable struct Ceed_private end

const Ceed = Ptr{Ceed_private}

mutable struct CeedRequest_private end

const CeedRequest = Ptr{CeedRequest_private}

mutable struct CeedVector_private end

const CeedVector = Ptr{CeedVector_private}

mutable struct CeedElemRestriction_private end

const CeedElemRestriction = Ptr{CeedElemRestriction_private}

mutable struct CeedBasis_private end

const CeedBasis = Ptr{CeedBasis_private}

mutable struct CeedQFunction_private end

const CeedQFunction = Ptr{CeedQFunction_private}

mutable struct CeedQFunctionContext_private end

const CeedQFunctionContext = Ptr{CeedQFunctionContext_private}

mutable struct CeedOperator_private end

const CeedOperator = Ptr{CeedOperator_private}

function CeedRegistryGetList(n, resources, array)
    ccall((:CeedRegistryGetList, libceed), Cint, (Ptr{Csize_t}, Ptr{Ptr{Ptr{Cchar}}}, Ptr{Ptr{CeedInt}}), n, resources, array)
end

function CeedInit(resource, ceed)
    ccall((:CeedInit, libceed), Cint, (Ptr{Cchar}, Ptr{Ceed}), resource, ceed)
end

function CeedReferenceCopy(ceed, ceed_copy)
    ccall((:CeedReferenceCopy, libceed), Cint, (Ceed, Ptr{Ceed}), ceed, ceed_copy)
end

function CeedGetResource(ceed, resource)
    ccall((:CeedGetResource, libceed), Cint, (Ceed, Ptr{Ptr{Cchar}}), ceed, resource)
end

function CeedIsDeterministic(ceed, is_deterministic)
    ccall((:CeedIsDeterministic, libceed), Cint, (Ceed, Ptr{Bool}), ceed, is_deterministic)
end

function CeedView(ceed, stream)
    ccall((:CeedView, libceed), Cint, (Ceed, Ptr{Libc.FILE}), ceed, stream)
end

function CeedDestroy(ceed)
    ccall((:CeedDestroy, libceed), Cint, (Ptr{Ceed},), ceed)
end

# automatic type deduction for variadic arguments may not be what you want, please use with caution
@generated function CeedErrorImpl(arg1, arg2, arg3, arg4, arg5, arg6, va_list...)
        :(@ccall(libceed.CeedErrorImpl(arg1::Ceed, arg2::Ptr{Cchar}, arg3::Cint, arg4::Ptr{Cchar}, arg5::Cint, arg6::Ptr{Cchar}; $(to_c_type_pairs(va_list)...))::Cint))
    end

# typedef int ( * CeedErrorHandler ) ( Ceed , const char * , int , const char * , int , const char * , va_list * )
const CeedErrorHandler = Ptr{Cvoid}

function CeedSetErrorHandler(ceed, eh)
    ccall((:CeedSetErrorHandler, libceed), Cint, (Ceed, CeedErrorHandler), ceed, eh)
end

function CeedGetErrorMessage(arg1, err_msg)
    ccall((:CeedGetErrorMessage, libceed), Cint, (Ceed, Ptr{Ptr{Cchar}}), arg1, err_msg)
end

function CeedResetErrorMessage(arg1, err_msg)
    ccall((:CeedResetErrorMessage, libceed), Cint, (Ceed, Ptr{Ptr{Cchar}}), arg1, err_msg)
end

function CeedGetVersion(major, minor, patch, release)
    ccall((:CeedGetVersion, libceed), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Bool}), major, minor, patch, release)
end

function CeedGetScalarType(scalar_type)
    ccall((:CeedGetScalarType, libceed), Cint, (Ptr{CeedScalarType},), scalar_type)
end

@cenum CeedErrorType::Int32 begin
    CEED_ERROR_SUCCESS = 0
    CEED_ERROR_MINOR = 1
    CEED_ERROR_DIMENSION = 2
    CEED_ERROR_INCOMPLETE = 3
    CEED_ERROR_INCOMPATIBLE = 4
    CEED_ERROR_ACCESS = 5
    CEED_ERROR_MAJOR = -1
    CEED_ERROR_BACKEND = -2
    CEED_ERROR_UNSUPPORTED = -3
end

@cenum CeedMemType::UInt32 begin
    CEED_MEM_HOST = 0
    CEED_MEM_DEVICE = 1
end

function CeedGetPreferredMemType(ceed, type)
    ccall((:CeedGetPreferredMemType, libceed), Cint, (Ceed, Ptr{CeedMemType}), ceed, type)
end

@cenum CeedCopyMode::UInt32 begin
    CEED_COPY_VALUES = 0
    CEED_USE_POINTER = 1
    CEED_OWN_POINTER = 2
end

@cenum CeedNormType::UInt32 begin
    CEED_NORM_1 = 0
    CEED_NORM_2 = 1
    CEED_NORM_MAX = 2
end

function CeedVectorCreate(ceed, len, vec)
    ccall((:CeedVectorCreate, libceed), Cint, (Ceed, CeedInt, Ptr{CeedVector}), ceed, len, vec)
end

function CeedVectorReferenceCopy(vec, vec_copy)
    ccall((:CeedVectorReferenceCopy, libceed), Cint, (CeedVector, Ptr{CeedVector}), vec, vec_copy)
end

function CeedVectorSetArray(vec, mem_type, copy_mode, array)
    ccall((:CeedVectorSetArray, libceed), Cint, (CeedVector, CeedMemType, CeedCopyMode, Ptr{CeedScalar}), vec, mem_type, copy_mode, array)
end

function CeedVectorSetValue(vec, value)
    ccall((:CeedVectorSetValue, libceed), Cint, (CeedVector, CeedScalar), vec, value)
end

function CeedVectorSyncArray(vec, mem_type)
    ccall((:CeedVectorSyncArray, libceed), Cint, (CeedVector, CeedMemType), vec, mem_type)
end

function CeedVectorTakeArray(vec, mem_type, array)
    ccall((:CeedVectorTakeArray, libceed), Cint, (CeedVector, CeedMemType, Ptr{Ptr{CeedScalar}}), vec, mem_type, array)
end

function CeedVectorGetArray(vec, mem_type, array)
    ccall((:CeedVectorGetArray, libceed), Cint, (CeedVector, CeedMemType, Ptr{Ptr{CeedScalar}}), vec, mem_type, array)
end

function CeedVectorGetArrayRead(vec, mem_type, array)
    ccall((:CeedVectorGetArrayRead, libceed), Cint, (CeedVector, CeedMemType, Ptr{Ptr{CeedScalar}}), vec, mem_type, array)
end

function CeedVectorRestoreArray(vec, array)
    ccall((:CeedVectorRestoreArray, libceed), Cint, (CeedVector, Ptr{Ptr{CeedScalar}}), vec, array)
end

function CeedVectorRestoreArrayRead(vec, array)
    ccall((:CeedVectorRestoreArrayRead, libceed), Cint, (CeedVector, Ptr{Ptr{CeedScalar}}), vec, array)
end

function CeedVectorNorm(vec, type, norm)
    ccall((:CeedVectorNorm, libceed), Cint, (CeedVector, CeedNormType, Ptr{CeedScalar}), vec, type, norm)
end

function CeedVectorScale(x, alpha)
    ccall((:CeedVectorScale, libceed), Cint, (CeedVector, CeedScalar), x, alpha)
end

function CeedVectorAXPY(y, alpha, x)
    ccall((:CeedVectorAXPY, libceed), Cint, (CeedVector, CeedScalar, CeedVector), y, alpha, x)
end

function CeedVectorPointwiseMult(w, x, y)
    ccall((:CeedVectorPointwiseMult, libceed), Cint, (CeedVector, CeedVector, CeedVector), w, x, y)
end

function CeedVectorReciprocal(vec)
    ccall((:CeedVectorReciprocal, libceed), Cint, (CeedVector,), vec)
end

function CeedVectorView(vec, fp_fmt, stream)
    ccall((:CeedVectorView, libceed), Cint, (CeedVector, Ptr{Cchar}, Ptr{Libc.FILE}), vec, fp_fmt, stream)
end

function CeedVectorGetLength(vec, length)
    ccall((:CeedVectorGetLength, libceed), Cint, (CeedVector, Ptr{CeedInt}), vec, length)
end

function CeedVectorDestroy(vec)
    ccall((:CeedVectorDestroy, libceed), Cint, (Ptr{CeedVector},), vec)
end

function CeedRequestWait(req)
    ccall((:CeedRequestWait, libceed), Cint, (Ptr{CeedRequest},), req)
end

@cenum CeedTransposeMode::UInt32 begin
    CEED_NOTRANSPOSE = 0
    CEED_TRANSPOSE = 1
end

function CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, rstr)
    ccall((:CeedElemRestrictionCreate, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, CeedMemType, CeedCopyMode, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, rstr)
end

function CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, num_comp, l_size, strides, rstr)
    ccall((:CeedElemRestrictionCreateStrided, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, num_comp, l_size, strides, rstr)
end

function CeedElemRestrictionCreateBlocked(ceed, num_elem, elem_size, blk_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, rstr)
    ccall((:CeedElemRestrictionCreateBlocked, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, CeedMemType, CeedCopyMode, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, blk_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, rstr)
end

function CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, elem_size, blk_size, num_comp, l_size, strides, rstr)
    ccall((:CeedElemRestrictionCreateBlockedStrided, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, blk_size, num_comp, l_size, strides, rstr)
end

function CeedElemRestrictionReferenceCopy(rstr, rstr_copy)
    ccall((:CeedElemRestrictionReferenceCopy, libceed), Cint, (CeedElemRestriction, Ptr{CeedElemRestriction}), rstr, rstr_copy)
end

function CeedElemRestrictionCreateVector(rstr, lvec, evec)
    ccall((:CeedElemRestrictionCreateVector, libceed), Cint, (CeedElemRestriction, Ptr{CeedVector}, Ptr{CeedVector}), rstr, lvec, evec)
end

function CeedElemRestrictionApply(rstr, t_mode, u, ru, request)
    ccall((:CeedElemRestrictionApply, libceed), Cint, (CeedElemRestriction, CeedTransposeMode, CeedVector, CeedVector, Ptr{CeedRequest}), rstr, t_mode, u, ru, request)
end

function CeedElemRestrictionApplyBlock(rstr, block, t_mode, u, ru, request)
    ccall((:CeedElemRestrictionApplyBlock, libceed), Cint, (CeedElemRestriction, CeedInt, CeedTransposeMode, CeedVector, CeedVector, Ptr{CeedRequest}), rstr, block, t_mode, u, ru, request)
end

function CeedElemRestrictionGetCompStride(rstr, comp_stride)
    ccall((:CeedElemRestrictionGetCompStride, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, comp_stride)
end

function CeedElemRestrictionGetNumElements(rstr, num_elem)
    ccall((:CeedElemRestrictionGetNumElements, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, num_elem)
end

function CeedElemRestrictionGetElementSize(rstr, elem_size)
    ccall((:CeedElemRestrictionGetElementSize, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, elem_size)
end

function CeedElemRestrictionGetLVectorSize(rstr, l_size)
    ccall((:CeedElemRestrictionGetLVectorSize, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, l_size)
end

function CeedElemRestrictionGetNumComponents(rstr, num_comp)
    ccall((:CeedElemRestrictionGetNumComponents, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, num_comp)
end

function CeedElemRestrictionGetNumBlocks(rstr, num_blk)
    ccall((:CeedElemRestrictionGetNumBlocks, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, num_blk)
end

function CeedElemRestrictionGetBlockSize(rstr, blk_size)
    ccall((:CeedElemRestrictionGetBlockSize, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, blk_size)
end

function CeedElemRestrictionGetMultiplicity(rstr, mult)
    ccall((:CeedElemRestrictionGetMultiplicity, libceed), Cint, (CeedElemRestriction, CeedVector), rstr, mult)
end

function CeedElemRestrictionView(rstr, stream)
    ccall((:CeedElemRestrictionView, libceed), Cint, (CeedElemRestriction, Ptr{Libc.FILE}), rstr, stream)
end

function CeedElemRestrictionDestroy(rstr)
    ccall((:CeedElemRestrictionDestroy, libceed), Cint, (Ptr{CeedElemRestriction},), rstr)
end

@cenum CeedEvalMode::UInt32 begin
    CEED_EVAL_NONE = 0
    CEED_EVAL_INTERP = 1
    CEED_EVAL_GRAD = 2
    CEED_EVAL_DIV = 4
    CEED_EVAL_CURL = 8
    CEED_EVAL_WEIGHT = 16
end

@cenum CeedQuadMode::UInt32 begin
    CEED_GAUSS = 0
    CEED_GAUSS_LOBATTO = 1
end

@cenum CeedElemTopology::UInt32 begin
    CEED_LINE = 65536
    CEED_TRIANGLE = 131073
    CEED_QUAD = 131074
    CEED_TET = 196611
    CEED_PYRAMID = 196612
    CEED_PRISM = 196613
    CEED_HEX = 196614
end

function CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, P, Q, quad_mode, basis)
    ccall((:CeedBasisCreateTensorH1Lagrange, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedQuadMode, Ptr{CeedBasis}), ceed, dim, num_comp, P, Q, quad_mode, basis)
end

function CeedBasisCreateTensorH1(ceed, dim, num_comp, P_1d, Q_1d, interp_1d, grad_1d, q_ref_1d, q_weight_1d, basis)
    ccall((:CeedBasisCreateTensorH1, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedBasis}), ceed, dim, num_comp, P_1d, Q_1d, interp_1d, grad_1d, q_ref_1d, q_weight_1d, basis)
end

function CeedBasisCreateH1(ceed, topo, num_comp, num_nodes, nqpts, interp, grad, q_ref, q_weights, basis)
    ccall((:CeedBasisCreateH1, libceed), Cint, (Ceed, CeedElemTopology, CeedInt, CeedInt, CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedBasis}), ceed, topo, num_comp, num_nodes, nqpts, interp, grad, q_ref, q_weights, basis)
end

function CeedBasisReferenceCopy(basis, basis_copy)
    ccall((:CeedBasisReferenceCopy, libceed), Cint, (CeedBasis, Ptr{CeedBasis}), basis, basis_copy)
end

function CeedBasisView(basis, stream)
    ccall((:CeedBasisView, libceed), Cint, (CeedBasis, Ptr{Libc.FILE}), basis, stream)
end

function CeedBasisApply(basis, num_elem, t_mode, eval_mode, u, v)
    ccall((:CeedBasisApply, libceed), Cint, (CeedBasis, CeedInt, CeedTransposeMode, CeedEvalMode, CeedVector, CeedVector), basis, num_elem, t_mode, eval_mode, u, v)
end

function CeedBasisGetDimension(basis, dim)
    ccall((:CeedBasisGetDimension, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, dim)
end

function CeedBasisGetTopology(basis, topo)
    ccall((:CeedBasisGetTopology, libceed), Cint, (CeedBasis, Ptr{CeedElemTopology}), basis, topo)
end

function CeedBasisGetNumComponents(basis, num_comp)
    ccall((:CeedBasisGetNumComponents, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, num_comp)
end

function CeedBasisGetNumNodes(basis, P)
    ccall((:CeedBasisGetNumNodes, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, P)
end

function CeedBasisGetNumNodes1D(basis, P_1d)
    ccall((:CeedBasisGetNumNodes1D, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, P_1d)
end

function CeedBasisGetNumQuadraturePoints(basis, Q)
    ccall((:CeedBasisGetNumQuadraturePoints, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, Q)
end

function CeedBasisGetNumQuadraturePoints1D(basis, Q_1d)
    ccall((:CeedBasisGetNumQuadraturePoints1D, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, Q_1d)
end

function CeedBasisGetQRef(basis, q_ref)
    ccall((:CeedBasisGetQRef, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, q_ref)
end

function CeedBasisGetQWeights(basis, q_weights)
    ccall((:CeedBasisGetQWeights, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, q_weights)
end

function CeedBasisGetInterp(basis, interp)
    ccall((:CeedBasisGetInterp, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, interp)
end

function CeedBasisGetInterp1D(basis, interp_1d)
    ccall((:CeedBasisGetInterp1D, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, interp_1d)
end

function CeedBasisGetGrad(basis, grad)
    ccall((:CeedBasisGetGrad, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, grad)
end

function CeedBasisGetGrad1D(basis, grad_1d)
    ccall((:CeedBasisGetGrad1D, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, grad_1d)
end

function CeedBasisDestroy(basis)
    ccall((:CeedBasisDestroy, libceed), Cint, (Ptr{CeedBasis},), basis)
end

function CeedGaussQuadrature(Q, q_ref_1d, q_weight_1d)
    ccall((:CeedGaussQuadrature, libceed), Cint, (CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}), Q, q_ref_1d, q_weight_1d)
end

function CeedLobattoQuadrature(Q, q_ref_1d, q_weight_1d)
    ccall((:CeedLobattoQuadrature, libceed), Cint, (CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}), Q, q_ref_1d, q_weight_1d)
end

function CeedQRFactorization(ceed, mat, tau, m, n)
    ccall((:CeedQRFactorization, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt, CeedInt), ceed, mat, tau, m, n)
end

function CeedSymmetricSchurDecomposition(ceed, mat, lambda, n)
    ccall((:CeedSymmetricSchurDecomposition, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt), ceed, mat, lambda, n)
end

function CeedSimultaneousDiagonalization(ceed, mat_A, mat_B, x, lambda, n)
    ccall((:CeedSimultaneousDiagonalization, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt), ceed, mat_A, mat_B, x, lambda, n)
end

# typedef int ( * CeedQFunctionUser ) ( void * ctx , const CeedInt Q , const CeedScalar * const * in , CeedScalar * const * out )
const CeedQFunctionUser = Ptr{Cvoid}

function CeedQFunctionCreateInterior(ceed, vec_length, f, source, qf)
    ccall((:CeedQFunctionCreateInterior, libceed), Cint, (Ceed, CeedInt, CeedQFunctionUser, Ptr{Cchar}, Ptr{CeedQFunction}), ceed, vec_length, f, source, qf)
end

function CeedQFunctionCreateInteriorByName(ceed, name, qf)
    ccall((:CeedQFunctionCreateInteriorByName, libceed), Cint, (Ceed, Ptr{Cchar}, Ptr{CeedQFunction}), ceed, name, qf)
end

function CeedQFunctionCreateIdentity(ceed, size, in_mode, out_mode, qf)
    ccall((:CeedQFunctionCreateIdentity, libceed), Cint, (Ceed, CeedInt, CeedEvalMode, CeedEvalMode, Ptr{CeedQFunction}), ceed, size, in_mode, out_mode, qf)
end

function CeedQFunctionReferenceCopy(qf, qf_copy)
    ccall((:CeedQFunctionReferenceCopy, libceed), Cint, (CeedQFunction, Ptr{CeedQFunction}), qf, qf_copy)
end

function CeedQFunctionAddInput(qf, field_name, size, eval_mode)
    ccall((:CeedQFunctionAddInput, libceed), Cint, (CeedQFunction, Ptr{Cchar}, CeedInt, CeedEvalMode), qf, field_name, size, eval_mode)
end

function CeedQFunctionAddOutput(qf, field_name, size, eval_mode)
    ccall((:CeedQFunctionAddOutput, libceed), Cint, (CeedQFunction, Ptr{Cchar}, CeedInt, CeedEvalMode), qf, field_name, size, eval_mode)
end

function CeedQFunctionSetContext(qf, ctx)
    ccall((:CeedQFunctionSetContext, libceed), Cint, (CeedQFunction, CeedQFunctionContext), qf, ctx)
end

function CeedQFunctionView(qf, stream)
    ccall((:CeedQFunctionView, libceed), Cint, (CeedQFunction, Ptr{Libc.FILE}), qf, stream)
end

function CeedQFunctionApply(qf, Q, u, v)
    ccall((:CeedQFunctionApply, libceed), Cint, (CeedQFunction, CeedInt, Ptr{CeedVector}, Ptr{CeedVector}), qf, Q, u, v)
end

function CeedQFunctionDestroy(qf)
    ccall((:CeedQFunctionDestroy, libceed), Cint, (Ptr{CeedQFunction},), qf)
end

function CeedQFunctionContextCreate(ceed, ctx)
    ccall((:CeedQFunctionContextCreate, libceed), Cint, (Ceed, Ptr{CeedQFunctionContext}), ceed, ctx)
end

function CeedQFunctionContextReferenceCopy(ctx, ctx_copy)
    ccall((:CeedQFunctionContextReferenceCopy, libceed), Cint, (CeedQFunctionContext, Ptr{CeedQFunctionContext}), ctx, ctx_copy)
end

function CeedQFunctionContextSetData(ctx, mem_type, copy_mode, size, data)
    ccall((:CeedQFunctionContextSetData, libceed), Cint, (CeedQFunctionContext, CeedMemType, CeedCopyMode, Csize_t, Ptr{Cvoid}), ctx, mem_type, copy_mode, size, data)
end

function CeedQFunctionContextTakeData(ctx, mem_type, data)
    ccall((:CeedQFunctionContextTakeData, libceed), Cint, (CeedQFunctionContext, CeedMemType, Ptr{Cvoid}), ctx, mem_type, data)
end

function CeedQFunctionContextGetData(ctx, mem_type, data)
    ccall((:CeedQFunctionContextGetData, libceed), Cint, (CeedQFunctionContext, CeedMemType, Ptr{Cvoid}), ctx, mem_type, data)
end

function CeedQFunctionContextRestoreData(ctx, data)
    ccall((:CeedQFunctionContextRestoreData, libceed), Cint, (CeedQFunctionContext, Ptr{Cvoid}), ctx, data)
end

function CeedQFunctionContextGetContextSize(ctx, ctx_size)
    ccall((:CeedQFunctionContextGetContextSize, libceed), Cint, (CeedQFunctionContext, Ptr{Csize_t}), ctx, ctx_size)
end

function CeedQFunctionContextView(ctx, stream)
    ccall((:CeedQFunctionContextView, libceed), Cint, (CeedQFunctionContext, Ptr{Libc.FILE}), ctx, stream)
end

function CeedQFunctionContextDestroy(ctx)
    ccall((:CeedQFunctionContextDestroy, libceed), Cint, (Ptr{CeedQFunctionContext},), ctx)
end

function CeedOperatorCreate(ceed, qf, dqf, dqfT, op)
    ccall((:CeedOperatorCreate, libceed), Cint, (Ceed, CeedQFunction, CeedQFunction, CeedQFunction, Ptr{CeedOperator}), ceed, qf, dqf, dqfT, op)
end

function CeedCompositeOperatorCreate(ceed, op)
    ccall((:CeedCompositeOperatorCreate, libceed), Cint, (Ceed, Ptr{CeedOperator}), ceed, op)
end

function CeedOperatorReferenceCopy(op, op_copy)
    ccall((:CeedOperatorReferenceCopy, libceed), Cint, (CeedOperator, Ptr{CeedOperator}), op, op_copy)
end

function CeedOperatorSetField(op, field_name, r, b, v)
    ccall((:CeedOperatorSetField, libceed), Cint, (CeedOperator, Ptr{Cchar}, CeedElemRestriction, CeedBasis, CeedVector), op, field_name, r, b, v)
end

function CeedCompositeOperatorAddSub(composite_op, sub_op)
    ccall((:CeedCompositeOperatorAddSub, libceed), Cint, (CeedOperator, CeedOperator), composite_op, sub_op)
end

function CeedOperatorLinearAssembleQFunction(op, assembled, rstr, request)
    ccall((:CeedOperatorLinearAssembleQFunction, libceed), Cint, (CeedOperator, Ptr{CeedVector}, Ptr{CeedElemRestriction}, Ptr{CeedRequest}), op, assembled, rstr, request)
end

function CeedOperatorLinearAssembleDiagonal(op, assembled, request)
    ccall((:CeedOperatorLinearAssembleDiagonal, libceed), Cint, (CeedOperator, CeedVector, Ptr{CeedRequest}), op, assembled, request)
end

function CeedOperatorLinearAssembleAddDiagonal(op, assembled, request)
    ccall((:CeedOperatorLinearAssembleAddDiagonal, libceed), Cint, (CeedOperator, CeedVector, Ptr{CeedRequest}), op, assembled, request)
end

function CeedOperatorLinearAssemblePointBlockDiagonal(op, assembled, request)
    ccall((:CeedOperatorLinearAssemblePointBlockDiagonal, libceed), Cint, (CeedOperator, CeedVector, Ptr{CeedRequest}), op, assembled, request)
end

function CeedOperatorLinearAssembleAddPointBlockDiagonal(op, assembled, request)
    ccall((:CeedOperatorLinearAssembleAddPointBlockDiagonal, libceed), Cint, (CeedOperator, CeedVector, Ptr{CeedRequest}), op, assembled, request)
end

function CeedOperatorLinearAssembleSymbolic(op, num_entries, rows, cols)
    ccall((:CeedOperatorLinearAssembleSymbolic, libceed), Cint, (CeedOperator, Ptr{CeedInt}, Ptr{Ptr{CeedInt}}, Ptr{Ptr{CeedInt}}), op, num_entries, rows, cols)
end

function CeedOperatorLinearAssemble(op, values)
    ccall((:CeedOperatorLinearAssemble, libceed), Cint, (CeedOperator, CeedVector), op, values)
end

function CeedOperatorMultigridLevelCreate(op_fine, p_mult_fine, rstr_coarse, basis_coarse, op_coarse, op_prolong, op_restrict)
    ccall((:CeedOperatorMultigridLevelCreate, libceed), Cint, (CeedOperator, CeedVector, CeedElemRestriction, CeedBasis, Ptr{CeedOperator}, Ptr{CeedOperator}, Ptr{CeedOperator}), op_fine, p_mult_fine, rstr_coarse, basis_coarse, op_coarse, op_prolong, op_restrict)
end

function CeedOperatorMultigridLevelCreateTensorH1(op_fine, p_mult_fine, rstr_coarse, basis_coarse, interp_c_to_f, op_coarse, op_prolong, op_restrict)
    ccall((:CeedOperatorMultigridLevelCreateTensorH1, libceed), Cint, (CeedOperator, CeedVector, CeedElemRestriction, CeedBasis, Ptr{CeedScalar}, Ptr{CeedOperator}, Ptr{CeedOperator}, Ptr{CeedOperator}), op_fine, p_mult_fine, rstr_coarse, basis_coarse, interp_c_to_f, op_coarse, op_prolong, op_restrict)
end

function CeedOperatorMultigridLevelCreateH1(op_fine, p_mult_fine, rstr_coarse, basis_coarse, interp_c_to_f, op_coarse, op_prolong, op_restrict)
    ccall((:CeedOperatorMultigridLevelCreateH1, libceed), Cint, (CeedOperator, CeedVector, CeedElemRestriction, CeedBasis, Ptr{CeedScalar}, Ptr{CeedOperator}, Ptr{CeedOperator}, Ptr{CeedOperator}), op_fine, p_mult_fine, rstr_coarse, basis_coarse, interp_c_to_f, op_coarse, op_prolong, op_restrict)
end

function CeedOperatorCreateFDMElementInverse(op, fdm_inv, request)
    ccall((:CeedOperatorCreateFDMElementInverse, libceed), Cint, (CeedOperator, Ptr{CeedOperator}, Ptr{CeedRequest}), op, fdm_inv, request)
end

function CeedOperatorSetNumQuadraturePoints(op, num_qpts)
    ccall((:CeedOperatorSetNumQuadraturePoints, libceed), Cint, (CeedOperator, CeedInt), op, num_qpts)
end

function CeedOperatorView(op, stream)
    ccall((:CeedOperatorView, libceed), Cint, (CeedOperator, Ptr{Libc.FILE}), op, stream)
end

function CeedOperatorApply(op, in, out, request)
    ccall((:CeedOperatorApply, libceed), Cint, (CeedOperator, CeedVector, CeedVector, Ptr{CeedRequest}), op, in, out, request)
end

function CeedOperatorApplyAdd(op, in, out, request)
    ccall((:CeedOperatorApplyAdd, libceed), Cint, (CeedOperator, CeedVector, CeedVector, Ptr{CeedRequest}), op, in, out, request)
end

function CeedOperatorDestroy(op)
    ccall((:CeedOperatorDestroy, libceed), Cint, (Ptr{CeedOperator},), op)
end

function CeedIntPow(base, power)
    ccall((:CeedIntPow, libceed), CeedInt, (CeedInt, CeedInt), base, power)
end

function CeedIntMin(a, b)
    ccall((:CeedIntMin, libceed), CeedInt, (CeedInt, CeedInt), a, b)
end

function CeedIntMax(a, b)
    ccall((:CeedIntMax, libceed), CeedInt, (CeedInt, CeedInt), a, b)
end

function CeedRegisterAll()
    ccall((:CeedRegisterAll, libceed), Cint, ())
end

function CeedQFunctionRegisterAll()
    ccall((:CeedQFunctionRegisterAll, libceed), Cint, ())
end

function CeedQFunctionSetCUDAUserFunction(qf, f)
    ccall((:CeedQFunctionSetCUDAUserFunction, libceed), Cint, (CeedQFunction, Cint), qf, f)
end

# automatic type deduction for variadic arguments may not be what you want, please use with caution
@generated function CeedDebugImpl(arg1, arg2, va_list...)
        :(@ccall(libceed.CeedDebugImpl(arg1::Ceed, arg2::Ptr{Cchar}; $(to_c_type_pairs(va_list)...))::Cvoid))
    end

# automatic type deduction for variadic arguments may not be what you want, please use with caution
@generated function CeedDebugImpl256(arg1, arg2, arg3, va_list...)
        :(@ccall(libceed.CeedDebugImpl256(arg1::Ceed, arg2::Cuchar, arg3::Ptr{Cchar}; $(to_c_type_pairs(va_list)...))::Cvoid))
    end

mutable struct CeedTensorContract_private end

const CeedTensorContract = Ptr{CeedTensorContract_private}

function CeedMallocArray(n, unit, p)
    ccall((:CeedMallocArray, libceed), Cint, (Csize_t, Csize_t, Ptr{Cvoid}), n, unit, p)
end

function CeedCallocArray(n, unit, p)
    ccall((:CeedCallocArray, libceed), Cint, (Csize_t, Csize_t, Ptr{Cvoid}), n, unit, p)
end

function CeedReallocArray(n, unit, p)
    ccall((:CeedReallocArray, libceed), Cint, (Csize_t, Csize_t, Ptr{Cvoid}), n, unit, p)
end

function CeedFree(p)
    ccall((:CeedFree, libceed), Cint, (Ptr{Cvoid},), p)
end

mutable struct CeedQFunctionField_private end

const CeedQFunctionField = Ptr{CeedQFunctionField_private}

mutable struct CeedOperatorField_private end

const CeedOperatorField = Ptr{CeedOperatorField_private}

function CeedRegister(prefix, init, priority)
    ccall((:CeedRegister, libceed), Cint, (Ptr{Cchar}, Ptr{Cvoid}, Cuint), prefix, init, priority)
end

function CeedIsDebug(ceed, is_debug)
    ccall((:CeedIsDebug, libceed), Cint, (Ceed, Ptr{Bool}), ceed, is_debug)
end

function CeedGetParent(ceed, parent)
    ccall((:CeedGetParent, libceed), Cint, (Ceed, Ptr{Ceed}), ceed, parent)
end

function CeedGetDelegate(ceed, delegate)
    ccall((:CeedGetDelegate, libceed), Cint, (Ceed, Ptr{Ceed}), ceed, delegate)
end

function CeedSetDelegate(ceed, delegate)
    ccall((:CeedSetDelegate, libceed), Cint, (Ceed, Ceed), ceed, delegate)
end

function CeedGetObjectDelegate(ceed, delegate, obj_name)
    ccall((:CeedGetObjectDelegate, libceed), Cint, (Ceed, Ptr{Ceed}, Ptr{Cchar}), ceed, delegate, obj_name)
end

function CeedSetObjectDelegate(ceed, delegate, obj_name)
    ccall((:CeedSetObjectDelegate, libceed), Cint, (Ceed, Ceed, Ptr{Cchar}), ceed, delegate, obj_name)
end

function CeedOperatorCheckReady(op)
    ccall((:CeedOperatorCheckReady, libceed), Cint, (CeedOperator,), op)
end

function CeedOperatorGetActiveBasis(op, active_basis)
    ccall((:CeedOperatorGetActiveBasis, libceed), Cint, (CeedOperator, Ptr{CeedBasis}), op, active_basis)
end

function CeedOperatorGetActiveElemRestriction(op, active_rstr)
    ccall((:CeedOperatorGetActiveElemRestriction, libceed), Cint, (CeedOperator, Ptr{CeedElemRestriction}), op, active_rstr)
end

function CeedGetOperatorFallbackResource(ceed, resource)
    ccall((:CeedGetOperatorFallbackResource, libceed), Cint, (Ceed, Ptr{Ptr{Cchar}}), ceed, resource)
end

function CeedSetOperatorFallbackResource(ceed, resource)
    ccall((:CeedSetOperatorFallbackResource, libceed), Cint, (Ceed, Ptr{Cchar}), ceed, resource)
end

function CeedGetOperatorFallbackParentCeed(ceed, parent)
    ccall((:CeedGetOperatorFallbackParentCeed, libceed), Cint, (Ceed, Ptr{Ceed}), ceed, parent)
end

function CeedSetDeterministic(ceed, is_deterministic)
    ccall((:CeedSetDeterministic, libceed), Cint, (Ceed, Bool), ceed, is_deterministic)
end

function CeedSetBackendFunction(ceed, type, object, func_name, f)
    ccall((:CeedSetBackendFunction, libceed), Cint, (Ceed, Ptr{Cchar}, Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cvoid}), ceed, type, object, func_name, f)
end

function CeedGetData(ceed, data)
    ccall((:CeedGetData, libceed), Cint, (Ceed, Ptr{Cvoid}), ceed, data)
end

function CeedSetData(ceed, data)
    ccall((:CeedSetData, libceed), Cint, (Ceed, Ptr{Cvoid}), ceed, data)
end

function CeedReference(ceed)
    ccall((:CeedReference, libceed), Cint, (Ceed,), ceed)
end

function CeedVectorGetCeed(vec, ceed)
    ccall((:CeedVectorGetCeed, libceed), Cint, (CeedVector, Ptr{Ceed}), vec, ceed)
end

function CeedVectorGetState(vec, state)
    ccall((:CeedVectorGetState, libceed), Cint, (CeedVector, Ptr{UInt64}), vec, state)
end

function CeedVectorAddReference(vec)
    ccall((:CeedVectorAddReference, libceed), Cint, (CeedVector,), vec)
end

function CeedVectorGetData(vec, data)
    ccall((:CeedVectorGetData, libceed), Cint, (CeedVector, Ptr{Cvoid}), vec, data)
end

function CeedVectorSetData(vec, data)
    ccall((:CeedVectorSetData, libceed), Cint, (CeedVector, Ptr{Cvoid}), vec, data)
end

function CeedVectorReference(vec)
    ccall((:CeedVectorReference, libceed), Cint, (CeedVector,), vec)
end

function CeedElemRestrictionGetCeed(rstr, ceed)
    ccall((:CeedElemRestrictionGetCeed, libceed), Cint, (CeedElemRestriction, Ptr{Ceed}), rstr, ceed)
end

function CeedElemRestrictionGetStrides(rstr, strides)
    ccall((:CeedElemRestrictionGetStrides, libceed), Cint, (CeedElemRestriction, Ptr{NTuple{3, CeedInt}}), rstr, strides)
end

function CeedElemRestrictionGetOffsets(rstr, mem_type, offsets)
    ccall((:CeedElemRestrictionGetOffsets, libceed), Cint, (CeedElemRestriction, CeedMemType, Ptr{Ptr{CeedInt}}), rstr, mem_type, offsets)
end

function CeedElemRestrictionRestoreOffsets(rstr, offsets)
    ccall((:CeedElemRestrictionRestoreOffsets, libceed), Cint, (CeedElemRestriction, Ptr{Ptr{CeedInt}}), rstr, offsets)
end

function CeedElemRestrictionIsStrided(rstr, is_strided)
    ccall((:CeedElemRestrictionIsStrided, libceed), Cint, (CeedElemRestriction, Ptr{Bool}), rstr, is_strided)
end

function CeedElemRestrictionHasBackendStrides(rstr, has_backend_strides)
    ccall((:CeedElemRestrictionHasBackendStrides, libceed), Cint, (CeedElemRestriction, Ptr{Bool}), rstr, has_backend_strides)
end

function CeedElemRestrictionGetELayout(rstr, layout)
    ccall((:CeedElemRestrictionGetELayout, libceed), Cint, (CeedElemRestriction, Ptr{NTuple{3, CeedInt}}), rstr, layout)
end

function CeedElemRestrictionSetELayout(rstr, layout)
    ccall((:CeedElemRestrictionSetELayout, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, layout)
end

function CeedElemRestrictionGetData(rstr, data)
    ccall((:CeedElemRestrictionGetData, libceed), Cint, (CeedElemRestriction, Ptr{Cvoid}), rstr, data)
end

function CeedElemRestrictionSetData(rstr, data)
    ccall((:CeedElemRestrictionSetData, libceed), Cint, (CeedElemRestriction, Ptr{Cvoid}), rstr, data)
end

function CeedElemRestrictionReference(rstr)
    ccall((:CeedElemRestrictionReference, libceed), Cint, (CeedElemRestriction,), rstr)
end

function CeedBasisGetCollocatedGrad(basis, colo_grad_1d)
    ccall((:CeedBasisGetCollocatedGrad, libceed), Cint, (CeedBasis, Ptr{CeedScalar}), basis, colo_grad_1d)
end

function CeedHouseholderApplyQ(A, Q, tau, t_mode, m, n, k, row, col)
    ccall((:CeedHouseholderApplyQ, libceed), Cint, (Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedTransposeMode, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt), A, Q, tau, t_mode, m, n, k, row, col)
end

function CeedBasisGetCeed(basis, ceed)
    ccall((:CeedBasisGetCeed, libceed), Cint, (CeedBasis, Ptr{Ceed}), basis, ceed)
end

function CeedBasisIsTensor(basis, is_tensor)
    ccall((:CeedBasisIsTensor, libceed), Cint, (CeedBasis, Ptr{Bool}), basis, is_tensor)
end

function CeedBasisGetData(basis, data)
    ccall((:CeedBasisGetData, libceed), Cint, (CeedBasis, Ptr{Cvoid}), basis, data)
end

function CeedBasisSetData(basis, data)
    ccall((:CeedBasisSetData, libceed), Cint, (CeedBasis, Ptr{Cvoid}), basis, data)
end

function CeedBasisReference(basis)
    ccall((:CeedBasisReference, libceed), Cint, (CeedBasis,), basis)
end

function CeedBasisGetTopologyDimension(topo, dim)
    ccall((:CeedBasisGetTopologyDimension, libceed), Cint, (CeedElemTopology, Ptr{CeedInt}), topo, dim)
end

function CeedBasisGetTensorContract(basis, contract)
    ccall((:CeedBasisGetTensorContract, libceed), Cint, (CeedBasis, Ptr{CeedTensorContract}), basis, contract)
end

function CeedBasisSetTensorContract(basis, contract)
    ccall((:CeedBasisSetTensorContract, libceed), Cint, (CeedBasis, CeedTensorContract), basis, contract)
end

function CeedTensorContractCreate(ceed, basis, contract)
    ccall((:CeedTensorContractCreate, libceed), Cint, (Ceed, CeedBasis, Ptr{CeedTensorContract}), ceed, basis, contract)
end

function CeedTensorContractApply(contract, A, B, C, J, t, t_mode, Add, u, v)
    ccall((:CeedTensorContractApply, libceed), Cint, (CeedTensorContract, CeedInt, CeedInt, CeedInt, CeedInt, Ptr{CeedScalar}, CeedTransposeMode, CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}), contract, A, B, C, J, t, t_mode, Add, u, v)
end

function CeedTensorContractGetCeed(contract, ceed)
    ccall((:CeedTensorContractGetCeed, libceed), Cint, (CeedTensorContract, Ptr{Ceed}), contract, ceed)
end

function CeedTensorContractGetData(contract, data)
    ccall((:CeedTensorContractGetData, libceed), Cint, (CeedTensorContract, Ptr{Cvoid}), contract, data)
end

function CeedTensorContractSetData(contract, data)
    ccall((:CeedTensorContractSetData, libceed), Cint, (CeedTensorContract, Ptr{Cvoid}), contract, data)
end

function CeedTensorContractReference(contract)
    ccall((:CeedTensorContractReference, libceed), Cint, (CeedTensorContract,), contract)
end

function CeedTensorContractDestroy(contract)
    ccall((:CeedTensorContractDestroy, libceed), Cint, (Ptr{CeedTensorContract},), contract)
end

function CeedQFunctionRegister(arg1, arg2, arg3, arg4, init)
    ccall((:CeedQFunctionRegister, libceed), Cint, (Ptr{Cchar}, Ptr{Cchar}, CeedInt, CeedQFunctionUser, Ptr{Cvoid}), arg1, arg2, arg3, arg4, init)
end

function CeedQFunctionSetFortranStatus(qf, status)
    ccall((:CeedQFunctionSetFortranStatus, libceed), Cint, (CeedQFunction, Bool), qf, status)
end

function CeedQFunctionGetCeed(qf, ceed)
    ccall((:CeedQFunctionGetCeed, libceed), Cint, (CeedQFunction, Ptr{Ceed}), qf, ceed)
end

function CeedQFunctionGetVectorLength(qf, vec_length)
    ccall((:CeedQFunctionGetVectorLength, libceed), Cint, (CeedQFunction, Ptr{CeedInt}), qf, vec_length)
end

function CeedQFunctionGetNumArgs(qf, num_input_fields, num_output_fields)
    ccall((:CeedQFunctionGetNumArgs, libceed), Cint, (CeedQFunction, Ptr{CeedInt}, Ptr{CeedInt}), qf, num_input_fields, num_output_fields)
end

function CeedQFunctionGetSourcePath(qf, source)
    ccall((:CeedQFunctionGetSourcePath, libceed), Cint, (CeedQFunction, Ptr{Ptr{Cchar}}), qf, source)
end

function CeedQFunctionGetUserFunction(qf, f)
    ccall((:CeedQFunctionGetUserFunction, libceed), Cint, (CeedQFunction, Ptr{CeedQFunctionUser}), qf, f)
end

function CeedQFunctionGetContext(qf, ctx)
    ccall((:CeedQFunctionGetContext, libceed), Cint, (CeedQFunction, Ptr{CeedQFunctionContext}), qf, ctx)
end

function CeedQFunctionGetInnerContext(qf, ctx)
    ccall((:CeedQFunctionGetInnerContext, libceed), Cint, (CeedQFunction, Ptr{CeedQFunctionContext}), qf, ctx)
end

function CeedQFunctionIsIdentity(qf, is_identity)
    ccall((:CeedQFunctionIsIdentity, libceed), Cint, (CeedQFunction, Ptr{Bool}), qf, is_identity)
end

function CeedQFunctionGetData(qf, data)
    ccall((:CeedQFunctionGetData, libceed), Cint, (CeedQFunction, Ptr{Cvoid}), qf, data)
end

function CeedQFunctionSetData(qf, data)
    ccall((:CeedQFunctionSetData, libceed), Cint, (CeedQFunction, Ptr{Cvoid}), qf, data)
end

function CeedQFunctionReference(qf)
    ccall((:CeedQFunctionReference, libceed), Cint, (CeedQFunction,), qf)
end

function CeedQFunctionGetFields(qf, input_fields, output_fields)
    ccall((:CeedQFunctionGetFields, libceed), Cint, (CeedQFunction, Ptr{Ptr{CeedQFunctionField}}, Ptr{Ptr{CeedQFunctionField}}), qf, input_fields, output_fields)
end

function CeedQFunctionFieldGetName(qf_field, field_name)
    ccall((:CeedQFunctionFieldGetName, libceed), Cint, (CeedQFunctionField, Ptr{Ptr{Cchar}}), qf_field, field_name)
end

function CeedQFunctionFieldGetSize(qf_field, size)
    ccall((:CeedQFunctionFieldGetSize, libceed), Cint, (CeedQFunctionField, Ptr{CeedInt}), qf_field, size)
end

function CeedQFunctionFieldGetEvalMode(qf_field, eval_mode)
    ccall((:CeedQFunctionFieldGetEvalMode, libceed), Cint, (CeedQFunctionField, Ptr{CeedEvalMode}), qf_field, eval_mode)
end

function CeedQFunctionContextGetCeed(cxt, ceed)
    ccall((:CeedQFunctionContextGetCeed, libceed), Cint, (CeedQFunctionContext, Ptr{Ceed}), cxt, ceed)
end

function CeedQFunctionContextGetState(ctx, state)
    ccall((:CeedQFunctionContextGetState, libceed), Cint, (CeedQFunctionContext, Ptr{UInt64}), ctx, state)
end

function CeedQFunctionContextGetBackendData(ctx, data)
    ccall((:CeedQFunctionContextGetBackendData, libceed), Cint, (CeedQFunctionContext, Ptr{Cvoid}), ctx, data)
end

function CeedQFunctionContextSetBackendData(ctx, data)
    ccall((:CeedQFunctionContextSetBackendData, libceed), Cint, (CeedQFunctionContext, Ptr{Cvoid}), ctx, data)
end

function CeedQFunctionContextReference(ctx)
    ccall((:CeedQFunctionContextReference, libceed), Cint, (CeedQFunctionContext,), ctx)
end

function CeedOperatorGetCeed(op, ceed)
    ccall((:CeedOperatorGetCeed, libceed), Cint, (CeedOperator, Ptr{Ceed}), op, ceed)
end

function CeedOperatorGetNumElements(op, num_elem)
    ccall((:CeedOperatorGetNumElements, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, num_elem)
end

function CeedOperatorGetNumQuadraturePoints(op, num_qpts)
    ccall((:CeedOperatorGetNumQuadraturePoints, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, num_qpts)
end

function CeedOperatorGetNumArgs(op, num_args)
    ccall((:CeedOperatorGetNumArgs, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, num_args)
end

function CeedOperatorIsSetupDone(op, is_setup_done)
    ccall((:CeedOperatorIsSetupDone, libceed), Cint, (CeedOperator, Ptr{Bool}), op, is_setup_done)
end

function CeedOperatorGetQFunction(op, qf)
    ccall((:CeedOperatorGetQFunction, libceed), Cint, (CeedOperator, Ptr{CeedQFunction}), op, qf)
end

function CeedOperatorIsComposite(op, is_composite)
    ccall((:CeedOperatorIsComposite, libceed), Cint, (CeedOperator, Ptr{Bool}), op, is_composite)
end

function CeedOperatorGetNumSub(op, num_suboperators)
    ccall((:CeedOperatorGetNumSub, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, num_suboperators)
end

function CeedOperatorGetSubList(op, sub_operators)
    ccall((:CeedOperatorGetSubList, libceed), Cint, (CeedOperator, Ptr{Ptr{CeedOperator}}), op, sub_operators)
end

function CeedOperatorGetData(op, data)
    ccall((:CeedOperatorGetData, libceed), Cint, (CeedOperator, Ptr{Cvoid}), op, data)
end

function CeedOperatorSetData(op, data)
    ccall((:CeedOperatorSetData, libceed), Cint, (CeedOperator, Ptr{Cvoid}), op, data)
end

function CeedOperatorReference(op)
    ccall((:CeedOperatorReference, libceed), Cint, (CeedOperator,), op)
end

function CeedOperatorSetSetupDone(op)
    ccall((:CeedOperatorSetSetupDone, libceed), Cint, (CeedOperator,), op)
end

function CeedOperatorGetFields(op, input_fields, output_fields)
    ccall((:CeedOperatorGetFields, libceed), Cint, (CeedOperator, Ptr{Ptr{CeedOperatorField}}, Ptr{Ptr{CeedOperatorField}}), op, input_fields, output_fields)
end

function CeedOperatorFieldGetElemRestriction(op_field, rstr)
    ccall((:CeedOperatorFieldGetElemRestriction, libceed), Cint, (CeedOperatorField, Ptr{CeedElemRestriction}), op_field, rstr)
end

function CeedOperatorFieldGetBasis(op_field, basis)
    ccall((:CeedOperatorFieldGetBasis, libceed), Cint, (CeedOperatorField, Ptr{CeedBasis}), op_field, basis)
end

function CeedOperatorFieldGetVector(op_field, vec)
    ccall((:CeedOperatorFieldGetVector, libceed), Cint, (CeedOperatorField, Ptr{CeedVector}), op_field, vec)
end

function CeedMatrixMultiply(ceed, mat_A, mat_B, mat_C, m, n, kk)
    ccall((:CeedMatrixMultiply, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt, CeedInt, CeedInt), ceed, mat_A, mat_B, mat_C, m, n, kk)
end

# Skipping MacroDefinition: CEED_EXTERN extern

# Skipping MacroDefinition: CEED_QFUNCTION_HELPER static inline

const CEED_VERSION_MAJOR = 0

const CEED_VERSION_MINOR = 9

const CEED_VERSION_PATCH = 0

const CEED_VERSION_RELEASE = false

# Skipping MacroDefinition: CEED_INTERN CEED_EXTERN __attribute__ ( ( visibility ( "hidden" ) ) )

# Skipping MacroDefinition: CEED_UNUSED __attribute__ ( ( unused ) )

const CEED_MAX_RESOURCE_LEN = 1024

const CEED_MAX_BACKEND_PRIORITY = UINT_MAX

const CEED_COMPOSITE_MAX = 16

# Skipping MacroDefinition: CeedPragmaOptimizeOff _Pragma ( "clang optimize off" )

# Skipping MacroDefinition: CeedPragmaOptimizeOn _Pragma ( "clang optimize on" )

const CEED_DEBUG_COLOR = 0
