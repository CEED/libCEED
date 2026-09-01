using CEnum: CEnum, @cenum

#! format: off


const CeedInt = Int32

const CeedSize = Cptrdiff_t

const CeedInt8 = Int8

@cenum CeedScalarType::UInt32 begin
    CEED_SCALAR_FP32 = 0
    CEED_SCALAR_FP64 = 1
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

@cenum CeedTransposeMode::UInt32 begin
    CEED_NOTRANSPOSE = 0
    CEED_TRANSPOSE = 1
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
    CEED_TOPOLOGY_LINE = 65536
    CEED_TOPOLOGY_TRIANGLE = 131073
    CEED_TOPOLOGY_QUAD = 131074
    CEED_TOPOLOGY_TET = 196611
    CEED_TOPOLOGY_PYRAMID = 196612
    CEED_TOPOLOGY_PRISM = 196613
    CEED_TOPOLOGY_HEX = 196614
end

@cenum CeedContextFieldType::UInt32 begin
    CEED_CONTEXT_FIELD_BYTE = 1
    CEED_CONTEXT_FIELD_SCALAR = 2
    CEED_CONTEXT_FIELD_FLOAT = 3
    CEED_CONTEXT_FIELD_DOUBLE = 4
    CEED_CONTEXT_FIELD_INT8 = 5
    CEED_CONTEXT_FIELD_INT = 6
    CEED_CONTEXT_FIELD_INT32 = 7
    CEED_CONTEXT_FIELD_INT64 = 8
    CEED_CONTEXT_FIELD_SIZE = 9
    CEED_CONTEXT_FIELD_BOOL = 10
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

mutable struct CeedQFunctionField_private end

const CeedQFunctionField = Ptr{CeedQFunctionField_private}

mutable struct CeedQFunction_private end

const CeedQFunction = Ptr{CeedQFunction_private}

mutable struct CeedOperatorField_private end

const CeedOperatorField = Ptr{CeedOperatorField_private}

mutable struct CeedQFunctionContext_private end

const CeedQFunctionContext = Ptr{CeedQFunctionContext_private}

mutable struct CeedContextFieldLabel_private end

const CeedContextFieldLabel = Ptr{CeedContextFieldLabel_private}

mutable struct CeedOperator_private end

const CeedOperator = Ptr{CeedOperator_private}

mutable struct CeedObject_private end

const CeedObject = Ptr{CeedObject_private}

function CeedObjectView(obj, stream)
    ccall((:CeedObjectView, libceed), Cint, (CeedObject, Ptr{Libc.FILE}), obj, stream)
end

function CeedObjectSetNumViewTabs(obj, num_tabs)
    ccall((:CeedObjectSetNumViewTabs, libceed), Cint, (CeedObject, CeedInt), obj, num_tabs)
end

function CeedObjectGetNumViewTabs(obj, num_tabs)
    ccall((:CeedObjectGetNumViewTabs, libceed), Cint, (CeedObject, Ptr{CeedInt}), obj, num_tabs)
end

function CeedObjectGetCeed(obj, ceed)
    ccall((:CeedObjectGetCeed, libceed), Cint, (CeedObject, Ptr{Ceed}), obj, ceed)
end

function CeedObjectReturnCeed(obj)
    ccall((:CeedObjectReturnCeed, libceed), Ceed, (CeedObject,), obj)
end

function CeedObjectDestroy(obj)
    ccall((:CeedObjectDestroy, libceed), Cint, (Ptr{CeedObject},), obj)
end

function CeedRegistryGetList(n, resources, array)
    ccall((:CeedRegistryGetList, libceed), Cint, (Ptr{Csize_t}, Ptr{Ptr{Ptr{Cchar}}}, Ptr{Ptr{CeedInt}}), n, resources, array)
end

function CeedInit(resource, ceed)
    ccall((:CeedInit, libceed), Cint, (Ptr{Cchar}, Ptr{Ceed}), resource, ceed)
end

function CeedSetStream(ceed, handle)
    ccall((:CeedSetStream, libceed), Cint, (Ceed, Ptr{Cvoid}), ceed, handle)
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

function CeedAddJitSourceRoot(ceed, jit_source_root)
    ccall((:CeedAddJitSourceRoot, libceed), Cint, (Ceed, Ptr{Cchar}), ceed, jit_source_root)
end

function CeedAddRustSourceRoot(ceed, rust_source_root)
    ccall((:CeedAddRustSourceRoot, libceed), Cint, (Ceed, Ptr{Cchar}), ceed, rust_source_root)
end

function CeedAddJitDefine(ceed, jit_define)
    ccall((:CeedAddJitDefine, libceed), Cint, (Ceed, Ptr{Cchar}), ceed, jit_define)
end

function CeedSetNumViewTabs(ceed, num_tabs)
    ccall((:CeedSetNumViewTabs, libceed), Cint, (Ceed, CeedInt), ceed, num_tabs)
end

function CeedGetNumViewTabs(ceed, num_tabs)
    ccall((:CeedGetNumViewTabs, libceed), Cint, (Ceed, Ptr{CeedInt}), ceed, num_tabs)
end

function CeedView(ceed, stream)
    ccall((:CeedView, libceed), Cint, (Ceed, Ptr{Libc.FILE}), ceed, stream)
end

function CeedDestroy(ceed)
    ccall((:CeedDestroy, libceed), Cint, (Ptr{Ceed},), ceed)
end

# typedef int ( * CeedErrorHandler ) ( Ceed , const char * , int , const char * , int , const char * , va_list * )
const CeedErrorHandler = Ptr{Cvoid}

function CeedSetErrorHandler(ceed, handler)
    ccall((:CeedSetErrorHandler, libceed), Cint, (Ceed, CeedErrorHandler), ceed, handler)
end

function CeedGetErrorMessage(ceed, err_msg)
    ccall((:CeedGetErrorMessage, libceed), Cint, (Ceed, Ptr{Ptr{Cchar}}), ceed, err_msg)
end

function CeedResetErrorMessage(ceed, err_msg)
    ccall((:CeedResetErrorMessage, libceed), Cint, (Ceed, Ptr{Ptr{Cchar}}), ceed, err_msg)
end

function CeedGetVersion(major, minor, patch, release)
    ccall((:CeedGetVersion, libceed), Cint, (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Bool}), major, minor, patch, release)
end

function CeedGetGitVersion(git_version)
    ccall((:CeedGetGitVersion, libceed), Cint, (Ptr{Ptr{Cchar}},), git_version)
end

function CeedGetBuildConfiguration(build_config)
    ccall((:CeedGetBuildConfiguration, libceed), Cint, (Ptr{Ptr{Cchar}},), build_config)
end

function CeedSetIsClang(ceed, isClang)
    ccall((:CeedSetIsClang, libceed), Cint, (Ceed, Bool), ceed, isClang)
end

function CeedGetIsClang(ceed, isClang)
    ccall((:CeedGetIsClang, libceed), Cint, (Ceed, Ptr{Bool}), ceed, isClang)
end

function CeedGetScalarType(scalar_type)
    ccall((:CeedGetScalarType, libceed), Cint, (Ptr{CeedScalarType},), scalar_type)
end

function CeedGetPreferredMemType(ceed, type)
    ccall((:CeedGetPreferredMemType, libceed), Cint, (Ceed, Ptr{CeedMemType}), ceed, type)
end

function CeedVectorCreate(ceed, len, vec)
    ccall((:CeedVectorCreate, libceed), Cint, (Ceed, CeedSize, Ptr{CeedVector}), ceed, len, vec)
end

function CeedVectorReferenceCopy(vec, vec_copy)
    ccall((:CeedVectorReferenceCopy, libceed), Cint, (CeedVector, Ptr{CeedVector}), vec, vec_copy)
end

function CeedVectorCopy(vec, vec_copy)
    ccall((:CeedVectorCopy, libceed), Cint, (CeedVector, CeedVector), vec, vec_copy)
end

function CeedVectorCopyStrided(vec, start, stop, step, vec_copy)
    ccall((:CeedVectorCopyStrided, libceed), Cint, (CeedVector, CeedSize, CeedSize, CeedSize, CeedVector), vec, start, stop, step, vec_copy)
end

function CeedVectorSetArray(vec, mem_type, copy_mode, array)
    ccall((:CeedVectorSetArray, libceed), Cint, (CeedVector, CeedMemType, CeedCopyMode, Ptr{CeedScalar}), vec, mem_type, copy_mode, array)
end

function CeedVectorSetValue(vec, value)
    ccall((:CeedVectorSetValue, libceed), Cint, (CeedVector, CeedScalar), vec, value)
end

function CeedVectorSetValueStrided(vec, start, stop, step, value)
    ccall((:CeedVectorSetValueStrided, libceed), Cint, (CeedVector, CeedSize, CeedSize, CeedSize, CeedScalar), vec, start, stop, step, value)
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

function CeedVectorGetArrayWrite(vec, mem_type, array)
    ccall((:CeedVectorGetArrayWrite, libceed), Cint, (CeedVector, CeedMemType, Ptr{Ptr{CeedScalar}}), vec, mem_type, array)
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

function CeedVectorFilter(x, threshold)
    ccall((:CeedVectorFilter, libceed), Cint, (CeedVector, CeedScalar), x, threshold)
end

function CeedVectorAXPY(y, alpha, x)
    ccall((:CeedVectorAXPY, libceed), Cint, (CeedVector, CeedScalar, CeedVector), y, alpha, x)
end

function CeedVectorAXPBY(y, alpha, beta, x)
    ccall((:CeedVectorAXPBY, libceed), Cint, (CeedVector, CeedScalar, CeedScalar, CeedVector), y, alpha, beta, x)
end

function CeedVectorPointwiseMult(w, x, y)
    ccall((:CeedVectorPointwiseMult, libceed), Cint, (CeedVector, CeedVector, CeedVector), w, x, y)
end

function CeedVectorReciprocal(vec)
    ccall((:CeedVectorReciprocal, libceed), Cint, (CeedVector,), vec)
end

function CeedVectorSetNumViewTabs(vec, num_tabs)
    ccall((:CeedVectorSetNumViewTabs, libceed), Cint, (CeedVector, CeedInt), vec, num_tabs)
end

function CeedVectorGetNumViewTabs(vec, num_tabs)
    ccall((:CeedVectorGetNumViewTabs, libceed), Cint, (CeedVector, Ptr{CeedInt}), vec, num_tabs)
end

function CeedVectorViewRange(vec, start, stop, step, fp_fmt, stream)
    ccall((:CeedVectorViewRange, libceed), Cint, (CeedVector, CeedSize, CeedSize, CeedInt, Ptr{Cchar}, Ptr{Libc.FILE}), vec, start, stop, step, fp_fmt, stream)
end

function CeedVectorView(vec, fp_fmt, stream)
    ccall((:CeedVectorView, libceed), Cint, (CeedVector, Ptr{Cchar}, Ptr{Libc.FILE}), vec, fp_fmt, stream)
end

function CeedVectorGetCeed(vec, ceed)
    ccall((:CeedVectorGetCeed, libceed), Cint, (CeedVector, Ptr{Ceed}), vec, ceed)
end

function CeedVectorReturnCeed(vec)
    ccall((:CeedVectorReturnCeed, libceed), Ceed, (CeedVector,), vec)
end

function CeedVectorGetLength(vec, length)
    ccall((:CeedVectorGetLength, libceed), Cint, (CeedVector, Ptr{CeedSize}), vec, length)
end

function CeedVectorDestroy(vec)
    ccall((:CeedVectorDestroy, libceed), Cint, (Ptr{CeedVector},), vec)
end

function CeedRequestWait(req)
    ccall((:CeedRequestWait, libceed), Cint, (Ptr{CeedRequest},), req)
end

function CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, rstr)
    ccall((:CeedElemRestrictionCreate, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedSize, CeedMemType, CeedCopyMode, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, rstr)
end

function CeedElemRestrictionCreateOriented(ceed, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, orients, rstr)
    ccall((:CeedElemRestrictionCreateOriented, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedSize, CeedMemType, CeedCopyMode, Ptr{CeedInt}, Ptr{Bool}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, orients, rstr)
end

function CeedElemRestrictionCreateCurlOriented(ceed, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, curl_orients, rstr)
    ccall((:CeedElemRestrictionCreateCurlOriented, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedSize, CeedMemType, CeedCopyMode, Ptr{CeedInt}, Ptr{CeedInt8}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, curl_orients, rstr)
end

function CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, num_comp, l_size, strides, rstr)
    ccall((:CeedElemRestrictionCreateStrided, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedSize, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, num_comp, l_size, strides, rstr)
end

function CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, num_comp, l_size, mem_type, copy_mode, offsets, rstr)
    ccall((:CeedElemRestrictionCreateAtPoints, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedSize, CeedMemType, CeedCopyMode, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, num_elem, num_points, num_comp, l_size, mem_type, copy_mode, offsets, rstr)
end

function CeedElemRestrictionCreateBlocked(ceed, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, rstr)
    ccall((:CeedElemRestrictionCreateBlocked, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, CeedSize, CeedMemType, CeedCopyMode, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, rstr)
end

function CeedElemRestrictionCreateBlockedOriented(ceed, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, orients, rstr)
    ccall((:CeedElemRestrictionCreateBlockedOriented, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, CeedSize, CeedMemType, CeedCopyMode, Ptr{CeedInt}, Ptr{Bool}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, orients, rstr)
end

function CeedElemRestrictionCreateBlockedCurlOriented(ceed, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, curl_orients, rstr)
    ccall((:CeedElemRestrictionCreateBlockedCurlOriented, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, CeedSize, CeedMemType, CeedCopyMode, Ptr{CeedInt}, Ptr{CeedInt8}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, curl_orients, rstr)
end

function CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, elem_size, block_size, num_comp, l_size, strides, rstr)
    ccall((:CeedElemRestrictionCreateBlockedStrided, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedSize, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, num_elem, elem_size, block_size, num_comp, l_size, strides, rstr)
end

function CeedElemRestrictionCreateUnsignedCopy(rstr, rstr_unsigned)
    ccall((:CeedElemRestrictionCreateUnsignedCopy, libceed), Cint, (CeedElemRestriction, Ptr{CeedElemRestriction}), rstr, rstr_unsigned)
end

function CeedElemRestrictionCreateUnorientedCopy(rstr, rstr_unoriented)
    ccall((:CeedElemRestrictionCreateUnorientedCopy, libceed), Cint, (CeedElemRestriction, Ptr{CeedElemRestriction}), rstr, rstr_unoriented)
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

function CeedElemRestrictionApplyAtPointsInElement(rstr, elem, t_mode, u, ru, request)
    ccall((:CeedElemRestrictionApplyAtPointsInElement, libceed), Cint, (CeedElemRestriction, CeedInt, CeedTransposeMode, CeedVector, CeedVector, Ptr{CeedRequest}), rstr, elem, t_mode, u, ru, request)
end

function CeedElemRestrictionApplyBlock(rstr, block, t_mode, u, ru, request)
    ccall((:CeedElemRestrictionApplyBlock, libceed), Cint, (CeedElemRestriction, CeedInt, CeedTransposeMode, CeedVector, CeedVector, Ptr{CeedRequest}), rstr, block, t_mode, u, ru, request)
end

function CeedElemRestrictionGetCeed(rstr, ceed)
    ccall((:CeedElemRestrictionGetCeed, libceed), Cint, (CeedElemRestriction, Ptr{Ceed}), rstr, ceed)
end

function CeedElemRestrictionReturnCeed(rstr)
    ccall((:CeedElemRestrictionReturnCeed, libceed), Ceed, (CeedElemRestriction,), rstr)
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

function CeedElemRestrictionGetNumPoints(rstr, num_points)
    ccall((:CeedElemRestrictionGetNumPoints, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, num_points)
end

function CeedElemRestrictionGetNumPointsInElement(rstr, elem, num_points)
    ccall((:CeedElemRestrictionGetNumPointsInElement, libceed), Cint, (CeedElemRestriction, CeedInt, Ptr{CeedInt}), rstr, elem, num_points)
end

function CeedElemRestrictionGetMaxPointsInElement(rstr, max_points)
    ccall((:CeedElemRestrictionGetMaxPointsInElement, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, max_points)
end

function CeedElemRestrictionGetMinPointsInElement(rstr, min_points)
    ccall((:CeedElemRestrictionGetMinPointsInElement, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, min_points)
end

function CeedElemRestrictionGetMinMaxPointsInElement(rstr, min_points, max_points)
    ccall((:CeedElemRestrictionGetMinMaxPointsInElement, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}, Ptr{CeedInt}), rstr, min_points, max_points)
end

function CeedElemRestrictionGetLVectorSize(rstr, l_size)
    ccall((:CeedElemRestrictionGetLVectorSize, libceed), Cint, (CeedElemRestriction, Ptr{CeedSize}), rstr, l_size)
end

function CeedElemRestrictionGetEVectorSize(rstr, e_size)
    ccall((:CeedElemRestrictionGetEVectorSize, libceed), Cint, (CeedElemRestriction, Ptr{CeedSize}), rstr, e_size)
end

function CeedElemRestrictionGetNumComponents(rstr, num_comp)
    ccall((:CeedElemRestrictionGetNumComponents, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, num_comp)
end

function CeedElemRestrictionGetNumBlocks(rstr, num_block)
    ccall((:CeedElemRestrictionGetNumBlocks, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, num_block)
end

function CeedElemRestrictionGetBlockSize(rstr, block_size)
    ccall((:CeedElemRestrictionGetBlockSize, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, block_size)
end

function CeedElemRestrictionGetMultiplicity(rstr, mult)
    ccall((:CeedElemRestrictionGetMultiplicity, libceed), Cint, (CeedElemRestriction, CeedVector), rstr, mult)
end

function CeedElemRestrictionSetNumViewTabs(rstr, num_tabs)
    ccall((:CeedElemRestrictionSetNumViewTabs, libceed), Cint, (CeedElemRestriction, CeedInt), rstr, num_tabs)
end

function CeedElemRestrictionGetNumViewTabs(rstr, num_tabs)
    ccall((:CeedElemRestrictionGetNumViewTabs, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, num_tabs)
end

function CeedElemRestrictionView(rstr, stream)
    ccall((:CeedElemRestrictionView, libceed), Cint, (CeedElemRestriction, Ptr{Libc.FILE}), rstr, stream)
end

function CeedElemRestrictionDestroy(rstr)
    ccall((:CeedElemRestrictionDestroy, libceed), Cint, (Ptr{CeedElemRestriction},), rstr)
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

function CeedBasisCreateHdiv(ceed, topo, num_comp, num_nodes, nqpts, interp, div, q_ref, q_weights, basis)
    ccall((:CeedBasisCreateHdiv, libceed), Cint, (Ceed, CeedElemTopology, CeedInt, CeedInt, CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedBasis}), ceed, topo, num_comp, num_nodes, nqpts, interp, div, q_ref, q_weights, basis)
end

function CeedBasisCreateHcurl(ceed, topo, num_comp, num_nodes, nqpts, interp, curl, q_ref, q_weights, basis)
    ccall((:CeedBasisCreateHcurl, libceed), Cint, (Ceed, CeedElemTopology, CeedInt, CeedInt, CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedBasis}), ceed, topo, num_comp, num_nodes, nqpts, interp, curl, q_ref, q_weights, basis)
end

function CeedBasisCreateProjection(basis_from, basis_to, basis_project)
    ccall((:CeedBasisCreateProjection, libceed), Cint, (CeedBasis, CeedBasis, Ptr{CeedBasis}), basis_from, basis_to, basis_project)
end

function CeedBasisReferenceCopy(basis, basis_copy)
    ccall((:CeedBasisReferenceCopy, libceed), Cint, (CeedBasis, Ptr{CeedBasis}), basis, basis_copy)
end

function CeedBasisSetNumViewTabs(basis, num_tabs)
    ccall((:CeedBasisSetNumViewTabs, libceed), Cint, (CeedBasis, CeedInt), basis, num_tabs)
end

function CeedBasisGetNumViewTabs(basis, num_tabs)
    ccall((:CeedBasisGetNumViewTabs, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, num_tabs)
end

function CeedBasisView(basis, stream)
    ccall((:CeedBasisView, libceed), Cint, (CeedBasis, Ptr{Libc.FILE}), basis, stream)
end

function CeedBasisApply(basis, num_elem, t_mode, eval_mode, u, v)
    ccall((:CeedBasisApply, libceed), Cint, (CeedBasis, CeedInt, CeedTransposeMode, CeedEvalMode, CeedVector, CeedVector), basis, num_elem, t_mode, eval_mode, u, v)
end

function CeedBasisApplyAdd(basis, num_elem, t_mode, eval_mode, u, v)
    ccall((:CeedBasisApplyAdd, libceed), Cint, (CeedBasis, CeedInt, CeedTransposeMode, CeedEvalMode, CeedVector, CeedVector), basis, num_elem, t_mode, eval_mode, u, v)
end

function CeedBasisApplyAtPoints(basis, num_elem, num_points, t_mode, eval_mode, x_ref, u, v)
    ccall((:CeedBasisApplyAtPoints, libceed), Cint, (CeedBasis, CeedInt, Ptr{CeedInt}, CeedTransposeMode, CeedEvalMode, CeedVector, CeedVector, CeedVector), basis, num_elem, num_points, t_mode, eval_mode, x_ref, u, v)
end

function CeedBasisApplyAddAtPoints(basis, num_elem, num_points, t_mode, eval_mode, x_ref, u, v)
    ccall((:CeedBasisApplyAddAtPoints, libceed), Cint, (CeedBasis, CeedInt, Ptr{CeedInt}, CeedTransposeMode, CeedEvalMode, CeedVector, CeedVector, CeedVector), basis, num_elem, num_points, t_mode, eval_mode, x_ref, u, v)
end

function CeedBasisGetCeed(basis, ceed)
    ccall((:CeedBasisGetCeed, libceed), Cint, (CeedBasis, Ptr{Ceed}), basis, ceed)
end

function CeedBasisReturnCeed(basis)
    ccall((:CeedBasisReturnCeed, libceed), Ceed, (CeedBasis,), basis)
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

function CeedBasisGetDiv(basis, div)
    ccall((:CeedBasisGetDiv, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, div)
end

function CeedBasisGetCurl(basis, curl)
    ccall((:CeedBasisGetCurl, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, curl)
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

function CeedQFunctionGetFields(qf, num_input_fields, input_fields, num_output_fields, output_fields)
    ccall((:CeedQFunctionGetFields, libceed), Cint, (CeedQFunction, Ptr{CeedInt}, Ptr{Ptr{CeedQFunctionField}}, Ptr{CeedInt}, Ptr{Ptr{CeedQFunctionField}}), qf, num_input_fields, input_fields, num_output_fields, output_fields)
end

function CeedQFunctionSetContext(qf, ctx)
    ccall((:CeedQFunctionSetContext, libceed), Cint, (CeedQFunction, CeedQFunctionContext), qf, ctx)
end

function CeedQFunctionSetContextWritable(qf, is_writable)
    ccall((:CeedQFunctionSetContextWritable, libceed), Cint, (CeedQFunction, Bool), qf, is_writable)
end

function CeedQFunctionSetUserFlopsEstimate(qf, flops)
    ccall((:CeedQFunctionSetUserFlopsEstimate, libceed), Cint, (CeedQFunction, CeedSize), qf, flops)
end

function CeedQFunctionSetNumViewTabs(qf, num_tabs)
    ccall((:CeedQFunctionSetNumViewTabs, libceed), Cint, (CeedQFunction, CeedInt), qf, num_tabs)
end

function CeedQFunctionGetNumViewTabs(qf, num_tabs)
    ccall((:CeedQFunctionGetNumViewTabs, libceed), Cint, (CeedQFunction, Ptr{CeedInt}), qf, num_tabs)
end

function CeedQFunctionView(qf, stream)
    ccall((:CeedQFunctionView, libceed), Cint, (CeedQFunction, Ptr{Libc.FILE}), qf, stream)
end

function CeedQFunctionGetCeed(qf, ceed)
    ccall((:CeedQFunctionGetCeed, libceed), Cint, (CeedQFunction, Ptr{Ceed}), qf, ceed)
end

function CeedQFunctionReturnCeed(qf)
    ccall((:CeedQFunctionReturnCeed, libceed), Ceed, (CeedQFunction,), qf)
end

function CeedQFunctionApply(qf, Q, u, v)
    ccall((:CeedQFunctionApply, libceed), Cint, (CeedQFunction, CeedInt, Ptr{CeedVector}, Ptr{CeedVector}), qf, Q, u, v)
end

function CeedQFunctionDestroy(qf)
    ccall((:CeedQFunctionDestroy, libceed), Cint, (Ptr{CeedQFunction},), qf)
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

function CeedQFunctionFieldGetData(qf_field, field_name, size, eval_mode)
    ccall((:CeedQFunctionFieldGetData, libceed), Cint, (CeedQFunctionField, Ptr{Ptr{Cchar}}, Ptr{CeedInt}, Ptr{CeedEvalMode}), qf_field, field_name, size, eval_mode)
end

# typedef int ( * CeedQFunctionContextDataDestroyUser ) ( void * data )
const CeedQFunctionContextDataDestroyUser = Ptr{Cvoid}

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

function CeedQFunctionContextGetDataRead(ctx, mem_type, data)
    ccall((:CeedQFunctionContextGetDataRead, libceed), Cint, (CeedQFunctionContext, CeedMemType, Ptr{Cvoid}), ctx, mem_type, data)
end

function CeedQFunctionContextRestoreData(ctx, data)
    ccall((:CeedQFunctionContextRestoreData, libceed), Cint, (CeedQFunctionContext, Ptr{Cvoid}), ctx, data)
end

function CeedQFunctionContextRestoreDataRead(ctx, data)
    ccall((:CeedQFunctionContextRestoreDataRead, libceed), Cint, (CeedQFunctionContext, Ptr{Cvoid}), ctx, data)
end

function CeedQFunctionContextRegisterCeedByte(ctx, field_name, field_offset, num_values, field_description)
    ccall((:CeedQFunctionContextRegisterCeedByte, libceed), Cint, (CeedQFunctionContext, Ptr{Cchar}, Csize_t, Csize_t, Ptr{Cchar}), ctx, field_name, field_offset, num_values, field_description)
end

function CeedQFunctionContextRegisterCeedScalar(ctx, field_name, field_offset, num_values, field_description)
    ccall((:CeedQFunctionContextRegisterCeedScalar, libceed), Cint, (CeedQFunctionContext, Ptr{Cchar}, Csize_t, Csize_t, Ptr{Cchar}), ctx, field_name, field_offset, num_values, field_description)
end

function CeedQFunctionContextRegisterCeedFloat(ctx, field_name, field_offset, num_values, field_description)
    ccall((:CeedQFunctionContextRegisterCeedFloat, libceed), Cint, (CeedQFunctionContext, Ptr{Cchar}, Csize_t, Csize_t, Ptr{Cchar}), ctx, field_name, field_offset, num_values, field_description)
end

function CeedQFunctionContextRegisterCeedDouble(ctx, field_name, field_offset, num_values, field_description)
    ccall((:CeedQFunctionContextRegisterCeedDouble, libceed), Cint, (CeedQFunctionContext, Ptr{Cchar}, Csize_t, Csize_t, Ptr{Cchar}), ctx, field_name, field_offset, num_values, field_description)
end

function CeedQFunctionContextRegisterCeedInt8(ctx, field_name, field_offset, num_values, field_description)
    ccall((:CeedQFunctionContextRegisterCeedInt8, libceed), Cint, (CeedQFunctionContext, Ptr{Cchar}, Csize_t, Csize_t, Ptr{Cchar}), ctx, field_name, field_offset, num_values, field_description)
end

function CeedQFunctionContextRegisterCeedInt(ctx, field_name, field_offset, num_values, field_description)
    ccall((:CeedQFunctionContextRegisterCeedInt, libceed), Cint, (CeedQFunctionContext, Ptr{Cchar}, Csize_t, Csize_t, Ptr{Cchar}), ctx, field_name, field_offset, num_values, field_description)
end

function CeedQFunctionContextRegisterCeedInt32(ctx, field_name, field_offset, num_values, field_description)
    ccall((:CeedQFunctionContextRegisterCeedInt32, libceed), Cint, (CeedQFunctionContext, Ptr{Cchar}, Csize_t, Csize_t, Ptr{Cchar}), ctx, field_name, field_offset, num_values, field_description)
end

function CeedQFunctionContextRegisterCeedInt64(ctx, field_name, field_offset, num_values, field_description)
    ccall((:CeedQFunctionContextRegisterCeedInt64, libceed), Cint, (CeedQFunctionContext, Ptr{Cchar}, Csize_t, Csize_t, Ptr{Cchar}), ctx, field_name, field_offset, num_values, field_description)
end

function CeedQFunctionContextRegisterCeedSize(ctx, field_name, field_offset, num_values, field_description)
    ccall((:CeedQFunctionContextRegisterCeedSize, libceed), Cint, (CeedQFunctionContext, Ptr{Cchar}, Csize_t, Csize_t, Ptr{Cchar}), ctx, field_name, field_offset, num_values, field_description)
end

function CeedQFunctionContextRegisterCeedBoolean(ctx, field_name, field_offset, num_values, field_description)
    ccall((:CeedQFunctionContextRegisterCeedBoolean, libceed), Cint, (CeedQFunctionContext, Ptr{Cchar}, Csize_t, Csize_t, Ptr{Cchar}), ctx, field_name, field_offset, num_values, field_description)
end

function CeedQFunctionContextGetAllFieldLabels(ctx, field_labels, num_fields)
    ccall((:CeedQFunctionContextGetAllFieldLabels, libceed), Cint, (CeedQFunctionContext, Ptr{Ptr{CeedContextFieldLabel}}, Ptr{CeedInt}), ctx, field_labels, num_fields)
end

function CeedContextFieldLabelGetDescription(label, field_name, field_offset, num_values, field_description, field_type)
    ccall((:CeedContextFieldLabelGetDescription, libceed), Cint, (CeedContextFieldLabel, Ptr{Ptr{Cchar}}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Ptr{Cchar}}, Ptr{CeedContextFieldType}), label, field_name, field_offset, num_values, field_description, field_type)
end

function CeedQFunctionContextGetContextSize(ctx, ctx_size)
    ccall((:CeedQFunctionContextGetContextSize, libceed), Cint, (CeedQFunctionContext, Ptr{Csize_t}), ctx, ctx_size)
end

function CeedQFunctionContextSetNumViewTabs(ctx, num_tabs)
    ccall((:CeedQFunctionContextSetNumViewTabs, libceed), Cint, (CeedQFunctionContext, CeedInt), ctx, num_tabs)
end

function CeedQFunctionContextGetNumViewTabs(ctx, num_tabs)
    ccall((:CeedQFunctionContextGetNumViewTabs, libceed), Cint, (CeedQFunctionContext, Ptr{CeedInt}), ctx, num_tabs)
end

function CeedQFunctionContextView(ctx, stream)
    ccall((:CeedQFunctionContextView, libceed), Cint, (CeedQFunctionContext, Ptr{Libc.FILE}), ctx, stream)
end

function CeedQFunctionContextSetDataDestroy(ctx, f_mem_type, f)
    ccall((:CeedQFunctionContextSetDataDestroy, libceed), Cint, (CeedQFunctionContext, CeedMemType, CeedQFunctionContextDataDestroyUser), ctx, f_mem_type, f)
end

function CeedQFunctionContextDestroy(ctx)
    ccall((:CeedQFunctionContextDestroy, libceed), Cint, (Ptr{CeedQFunctionContext},), ctx)
end

function CeedOperatorCreate(ceed, qf, dqf, dqfT, op)
    ccall((:CeedOperatorCreate, libceed), Cint, (Ceed, CeedQFunction, CeedQFunction, CeedQFunction, Ptr{CeedOperator}), ceed, qf, dqf, dqfT, op)
end

function CeedOperatorCreateAtPoints(ceed, qf, dqf, dqfT, op)
    ccall((:CeedOperatorCreateAtPoints, libceed), Cint, (Ceed, CeedQFunction, CeedQFunction, CeedQFunction, Ptr{CeedOperator}), ceed, qf, dqf, dqfT, op)
end

function CeedOperatorCreateComposite(ceed, op)
    ccall((:CeedOperatorCreateComposite, libceed), Cint, (Ceed, Ptr{CeedOperator}), ceed, op)
end

function CeedOperatorReferenceCopy(op, op_copy)
    ccall((:CeedOperatorReferenceCopy, libceed), Cint, (CeedOperator, Ptr{CeedOperator}), op, op_copy)
end

function CeedOperatorSetField(op, field_name, rstr, basis, vec)
    ccall((:CeedOperatorSetField, libceed), Cint, (CeedOperator, Ptr{Cchar}, CeedElemRestriction, CeedBasis, CeedVector), op, field_name, rstr, basis, vec)
end

function CeedOperatorGetFields(op, num_input_fields, input_fields, num_output_fields, output_fields)
    ccall((:CeedOperatorGetFields, libceed), Cint, (CeedOperator, Ptr{CeedInt}, Ptr{Ptr{CeedOperatorField}}, Ptr{CeedInt}, Ptr{Ptr{CeedOperatorField}}), op, num_input_fields, input_fields, num_output_fields, output_fields)
end

function CeedOperatorAtPointsSetPoints(op, rstr_points, point_coords)
    ccall((:CeedOperatorAtPointsSetPoints, libceed), Cint, (CeedOperator, CeedElemRestriction, CeedVector), op, rstr_points, point_coords)
end

function CeedOperatorAtPointsGetPoints(op, rstr_points, point_coords)
    ccall((:CeedOperatorAtPointsGetPoints, libceed), Cint, (CeedOperator, Ptr{CeedElemRestriction}, Ptr{CeedVector}), op, rstr_points, point_coords)
end

function CeedOperatorIsAtPoints(op, is_at_points)
    ccall((:CeedOperatorIsAtPoints, libceed), Cint, (CeedOperator, Ptr{Bool}), op, is_at_points)
end

function CeedOperatorCompositeAddSub(composite_op, sub_op)
    ccall((:CeedOperatorCompositeAddSub, libceed), Cint, (CeedOperator, CeedOperator), composite_op, sub_op)
end

function CeedOperatorCompositeGetNumSub(op, num_suboperators)
    ccall((:CeedOperatorCompositeGetNumSub, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, num_suboperators)
end

function CeedOperatorCompositeGetSubList(op, sub_operators)
    ccall((:CeedOperatorCompositeGetSubList, libceed), Cint, (CeedOperator, Ptr{Ptr{CeedOperator}}), op, sub_operators)
end

function CeedOperatorCompositeGetSubByName(op, op_name, sub_op)
    ccall((:CeedOperatorCompositeGetSubByName, libceed), Cint, (CeedOperator, Ptr{Cchar}, Ptr{CeedOperator}), op, op_name, sub_op)
end

function CeedOperatorCompositeSetSequential(op, is_sequential)
    ccall((:CeedOperatorCompositeSetSequential, libceed), Cint, (CeedOperator, Bool), op, is_sequential)
end

function CeedOperatorCompositeIsSequential(op, is_sequential)
    ccall((:CeedOperatorCompositeIsSequential, libceed), Cint, (CeedOperator, Ptr{Bool}), op, is_sequential)
end

function CeedOperatorCheckReady(op)
    ccall((:CeedOperatorCheckReady, libceed), Cint, (CeedOperator,), op)
end

function CeedOperatorGetActiveVectorLengths(op, input_size, output_size)
    ccall((:CeedOperatorGetActiveVectorLengths, libceed), Cint, (CeedOperator, Ptr{CeedSize}, Ptr{CeedSize}), op, input_size, output_size)
end

function CeedOperatorSetQFunctionAssemblyReuse(op, reuse_assembly_data)
    ccall((:CeedOperatorSetQFunctionAssemblyReuse, libceed), Cint, (CeedOperator, Bool), op, reuse_assembly_data)
end

function CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(op, needs_data_update)
    ccall((:CeedOperatorSetQFunctionAssemblyDataUpdateNeeded, libceed), Cint, (CeedOperator, Bool), op, needs_data_update)
end

function CeedOperatorLinearAssembleQFunction(op, assembled, rstr, request)
    ccall((:CeedOperatorLinearAssembleQFunction, libceed), Cint, (CeedOperator, Ptr{CeedVector}, Ptr{CeedElemRestriction}, Ptr{CeedRequest}), op, assembled, rstr, request)
end

function CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, assembled, rstr, request)
    ccall((:CeedOperatorLinearAssembleQFunctionBuildOrUpdate, libceed), Cint, (CeedOperator, Ptr{CeedVector}, Ptr{CeedElemRestriction}, Ptr{CeedRequest}), op, assembled, rstr, request)
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

function CeedOperatorLinearAssemblePointBlockDiagonalSymbolic(op, num_entries, rows, cols)
    ccall((:CeedOperatorLinearAssemblePointBlockDiagonalSymbolic, libceed), Cint, (CeedOperator, Ptr{CeedSize}, Ptr{Ptr{CeedInt}}, Ptr{Ptr{CeedInt}}), op, num_entries, rows, cols)
end

function CeedOperatorLinearAssembleSymbolic(op, num_entries, rows, cols)
    ccall((:CeedOperatorLinearAssembleSymbolic, libceed), Cint, (CeedOperator, Ptr{CeedSize}, Ptr{Ptr{CeedInt}}, Ptr{Ptr{CeedInt}}), op, num_entries, rows, cols)
end

function CeedOperatorLinearAssembleGetNumEntries(op, num_entries)
    ccall((:CeedOperatorLinearAssembleGetNumEntries, libceed), Cint, (CeedOperator, Ptr{CeedSize}), op, num_entries)
end

function CeedOperatorLinearAssemble(op, values)
    ccall((:CeedOperatorLinearAssemble, libceed), Cint, (CeedOperator, CeedVector), op, values)
end

function CeedOperatorCompositeGetMultiplicity(op, num_skip_indices, skip_indices, mult)
    ccall((:CeedOperatorCompositeGetMultiplicity, libceed), Cint, (CeedOperator, CeedInt, Ptr{CeedInt}, CeedVector), op, num_skip_indices, skip_indices, mult)
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

function CeedOperatorSetName(op, name)
    ccall((:CeedOperatorSetName, libceed), Cint, (CeedOperator, Ptr{Cchar}), op, name)
end

function CeedOperatorGetName(op, name)
    ccall((:CeedOperatorGetName, libceed), Cint, (CeedOperator, Ptr{Ptr{Cchar}}), op, name)
end

function CeedOperatorSetNumViewTabs(op, num_tabs)
    ccall((:CeedOperatorSetNumViewTabs, libceed), Cint, (CeedOperator, CeedInt), op, num_tabs)
end

function CeedOperatorGetNumViewTabs(op, num_tabs)
    ccall((:CeedOperatorGetNumViewTabs, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, num_tabs)
end

function CeedOperatorView(op, stream)
    ccall((:CeedOperatorView, libceed), Cint, (CeedOperator, Ptr{Libc.FILE}), op, stream)
end

function CeedOperatorViewTerse(op, stream)
    ccall((:CeedOperatorViewTerse, libceed), Cint, (CeedOperator, Ptr{Libc.FILE}), op, stream)
end

function CeedOperatorGetCeed(op, ceed)
    ccall((:CeedOperatorGetCeed, libceed), Cint, (CeedOperator, Ptr{Ceed}), op, ceed)
end

function CeedOperatorReturnCeed(op)
    ccall((:CeedOperatorReturnCeed, libceed), Ceed, (CeedOperator,), op)
end

function CeedOperatorGetNumElements(op, num_elem)
    ccall((:CeedOperatorGetNumElements, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, num_elem)
end

function CeedOperatorGetNumQuadraturePoints(op, num_qpts)
    ccall((:CeedOperatorGetNumQuadraturePoints, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, num_qpts)
end

function CeedOperatorGetFlopsEstimate(op, flops)
    ccall((:CeedOperatorGetFlopsEstimate, libceed), Cint, (CeedOperator, Ptr{CeedSize}), op, flops)
end

function CeedOperatorGetContext(op, ctx)
    ccall((:CeedOperatorGetContext, libceed), Cint, (CeedOperator, Ptr{CeedQFunctionContext}), op, ctx)
end

function CeedOperatorGetContextFieldLabel(op, field_name, field_label)
    ccall((:CeedOperatorGetContextFieldLabel, libceed), Cint, (CeedOperator, Ptr{Cchar}, Ptr{CeedContextFieldLabel}), op, field_name, field_label)
end

function CeedOperatorSetContextCeedByte(op, field_label, values)
    ccall((:CeedOperatorSetContextCeedByte, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Cchar}), op, field_label, values)
end

function CeedOperatorGetContextCeedByteRead(op, field_label, num_values, values)
    ccall((:CeedOperatorGetContextCeedByteRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Cchar}}), op, field_label, num_values, values)
end

function CeedOperatorRestoreContextCeedByteRead(op, field_label, values)
    ccall((:CeedOperatorRestoreContextCeedByteRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Ptr{Cchar}}), op, field_label, values)
end

function CeedOperatorSetContextCeedScalar(op, field_label, values)
    ccall((:CeedOperatorSetContextCeedScalar, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{CeedScalar}), op, field_label, values)
end

function CeedOperatorGetContextCeedScalarRead(op, field_label, num_values, values)
    ccall((:CeedOperatorGetContextCeedScalarRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{CeedScalar}}), op, field_label, num_values, values)
end

function CeedOperatorRestoreContextCeedScalarRead(op, field_label, values)
    ccall((:CeedOperatorRestoreContextCeedScalarRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Ptr{CeedScalar}}), op, field_label, values)
end

function CeedOperatorSetContextCeedFloat(op, field_label, values)
    ccall((:CeedOperatorSetContextCeedFloat, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Cfloat}), op, field_label, values)
end

function CeedOperatorGetContextCeedFloatRead(op, field_label, num_values, values)
    ccall((:CeedOperatorGetContextCeedFloatRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Cfloat}}), op, field_label, num_values, values)
end

function CeedOperatorRestoreContextCeedFloatRead(op, field_label, values)
    ccall((:CeedOperatorRestoreContextCeedFloatRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Ptr{Cfloat}}), op, field_label, values)
end

function CeedOperatorSetContextCeedDouble(op, field_label, values)
    ccall((:CeedOperatorSetContextCeedDouble, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Cdouble}), op, field_label, values)
end

function CeedOperatorGetContextCeedDoubleRead(op, field_label, num_values, values)
    ccall((:CeedOperatorGetContextCeedDoubleRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Cdouble}}), op, field_label, num_values, values)
end

function CeedOperatorRestoreContextCeedDoubleRead(op, field_label, values)
    ccall((:CeedOperatorRestoreContextCeedDoubleRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Ptr{Cdouble}}), op, field_label, values)
end

function CeedOperatorSetContextCeedInt8(op, field_label, values)
    ccall((:CeedOperatorSetContextCeedInt8, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{CeedInt8}), op, field_label, values)
end

function CeedOperatorGetContextCeedInt8Read(op, field_label, num_values, values)
    ccall((:CeedOperatorGetContextCeedInt8Read, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{CeedInt8}}), op, field_label, num_values, values)
end

function CeedOperatorRestoreContextCeedInt8Read(op, field_label, values)
    ccall((:CeedOperatorRestoreContextCeedInt8Read, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Ptr{CeedInt8}}), op, field_label, values)
end

function CeedOperatorSetContextCeedInt(op, field_label, values)
    ccall((:CeedOperatorSetContextCeedInt, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{CeedInt}), op, field_label, values)
end

function CeedOperatorGetContextCeedIntRead(op, field_label, num_values, values)
    ccall((:CeedOperatorGetContextCeedIntRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{CeedInt}}), op, field_label, num_values, values)
end

function CeedOperatorRestoreContextCeedIntRead(op, field_label, values)
    ccall((:CeedOperatorRestoreContextCeedIntRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Ptr{CeedInt}}), op, field_label, values)
end

function CeedOperatorSetContextCeedInt32(op, field_label, values)
    ccall((:CeedOperatorSetContextCeedInt32, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Int32}), op, field_label, values)
end

function CeedOperatorGetContextCeedInt32Read(op, field_label, num_values, values)
    ccall((:CeedOperatorGetContextCeedInt32Read, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Int32}}), op, field_label, num_values, values)
end

function CeedOperatorRestoreContextCeedInt32Read(op, field_label, values)
    ccall((:CeedOperatorRestoreContextCeedInt32Read, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Ptr{Int32}}), op, field_label, values)
end

function CeedOperatorSetContextCeedInt64(op, field_label, values)
    ccall((:CeedOperatorSetContextCeedInt64, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Int64}), op, field_label, values)
end

function CeedOperatorGetContextCeedInt64Read(op, field_label, num_values, values)
    ccall((:CeedOperatorGetContextCeedInt64Read, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Int64}}), op, field_label, num_values, values)
end

function CeedOperatorRestoreContextCeedInt64Read(op, field_label, values)
    ccall((:CeedOperatorRestoreContextCeedInt64Read, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Ptr{Int64}}), op, field_label, values)
end

function CeedOperatorSetContextCeedSize(op, field_label, values)
    ccall((:CeedOperatorSetContextCeedSize, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{CeedSize}), op, field_label, values)
end

function CeedOperatorGetContextCeedSizeRead(op, field_label, num_values, values)
    ccall((:CeedOperatorGetContextCeedSizeRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{CeedSize}}), op, field_label, num_values, values)
end

function CeedOperatorRestoreContextCeedSizeRead(op, field_label, values)
    ccall((:CeedOperatorRestoreContextCeedSizeRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Ptr{CeedSize}}), op, field_label, values)
end

function CeedOperatorSetContextCeedBoolean(op, field_label, values)
    ccall((:CeedOperatorSetContextCeedBoolean, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Bool}), op, field_label, values)
end

function CeedOperatorGetContextCeedBooleanRead(op, field_label, num_values, values)
    ccall((:CeedOperatorGetContextCeedBooleanRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Bool}}), op, field_label, num_values, values)
end

function CeedOperatorRestoreContextCeedBooleanRead(op, field_label, values)
    ccall((:CeedOperatorRestoreContextCeedBooleanRead, libceed), Cint, (CeedOperator, CeedContextFieldLabel, Ptr{Ptr{Bool}}), op, field_label, values)
end

function CeedOperatorApply(op, in, out, request)
    ccall((:CeedOperatorApply, libceed), Cint, (CeedOperator, CeedVector, CeedVector, Ptr{CeedRequest}), op, in, out, request)
end

function CeedOperatorApplyAdd(op, in, out, request)
    ccall((:CeedOperatorApplyAdd, libceed), Cint, (CeedOperator, CeedVector, CeedVector, Ptr{CeedRequest}), op, in, out, request)
end

function CeedOperatorApplyAddActive(op, in, out, request)
    ccall((:CeedOperatorApplyAddActive, libceed), Cint, (CeedOperator, CeedVector, CeedVector, Ptr{CeedRequest}), op, in, out, request)
end

function CeedOperatorAssemblyDataStrip(op)
    ccall((:CeedOperatorAssemblyDataStrip, libceed), Cint, (CeedOperator,), op)
end

function CeedOperatorDestroy(op)
    ccall((:CeedOperatorDestroy, libceed), Cint, (Ptr{CeedOperator},), op)
end

function CeedCompositeOperatorCreate(a, b)
    ccall((:CeedCompositeOperatorCreate, libceed), Cint, (Ceed, Ptr{CeedOperator}), a, b)
end

function CeedCompositeOperatorAddSub(a, b)
    ccall((:CeedCompositeOperatorAddSub, libceed), Cint, (CeedOperator, CeedOperator), a, b)
end

function CeedCompositeOperatorGetNumSub(a, b)
    ccall((:CeedCompositeOperatorGetNumSub, libceed), Cint, (CeedOperator, Ptr{CeedInt}), a, b)
end

function CeedCompositeOperatorGetSubList(a, b)
    ccall((:CeedCompositeOperatorGetSubList, libceed), Cint, (CeedOperator, Ptr{Ptr{CeedOperator}}), a, b)
end

function CeedCompositeOperatorGetSubByName(a, b, c)
    ccall((:CeedCompositeOperatorGetSubByName, libceed), Cint, (CeedOperator, Ptr{Cchar}, Ptr{CeedOperator}), a, b, c)
end

function CeedCompositeOperatorGetMultiplicity(a, b, c, d)
    ccall((:CeedCompositeOperatorGetMultiplicity, libceed), Cint, (CeedOperator, CeedInt, Ptr{CeedInt}, CeedVector), a, b, c, d)
end

function CeedOperatorGetFieldByName(op, field_name, op_field)
    ccall((:CeedOperatorGetFieldByName, libceed), Cint, (CeedOperator, Ptr{Cchar}, Ptr{CeedOperatorField}), op, field_name, op_field)
end

function CeedOperatorFieldGetName(op_field, field_name)
    ccall((:CeedOperatorFieldGetName, libceed), Cint, (CeedOperatorField, Ptr{Ptr{Cchar}}), op_field, field_name)
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

function CeedOperatorFieldGetData(op_field, field_name, rstr, basis, vec)
    ccall((:CeedOperatorFieldGetData, libceed), Cint, (CeedOperatorField, Ptr{Ptr{Cchar}}, Ptr{CeedElemRestriction}, Ptr{CeedBasis}, Ptr{CeedVector}), op_field, field_name, rstr, basis, vec)
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

function CeedDebugFlag(ceed)
    ccall((:CeedDebugFlag, libceed), Bool, (Ceed,), ceed)
end

function CeedDebugFlagEnv()
    ccall((:CeedDebugFlagEnv, libceed), Bool, ())
end

function CeedMallocArray(n, unit, p)
    ccall((:CeedMallocArray, libceed), Cint, (Csize_t, Csize_t, Ptr{Cvoid}), n, unit, p)
end

function CeedCallocArray(n, unit, p)
    ccall((:CeedCallocArray, libceed), Cint, (Csize_t, Csize_t, Ptr{Cvoid}), n, unit, p)
end

function CeedReallocArray(n, unit, p)
    ccall((:CeedReallocArray, libceed), Cint, (Csize_t, Csize_t, Ptr{Cvoid}), n, unit, p)
end

function CeedSetBackendFunctionImpl(ceed, type, object, func_name, f)
    ccall((:CeedSetBackendFunctionImpl, libceed), Cint, (Ceed, Ptr{Cchar}, Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cvoid}), ceed, type, object, func_name, f)
end

@cenum CeedDebugColor::UInt32 begin
    CEED_DEBUG_COLOR_SUCCESS = 108
    CEED_DEBUG_COLOR_WARNING = 208
    CEED_DEBUG_COLOR_ERROR = 196
    CEED_DEBUG_COLOR_NONE = 255
end

mutable struct CeedTensorContract_private end

const CeedTensorContract = Ptr{CeedTensorContract_private}

mutable struct CeedQFunctionAssemblyData_private end

const CeedQFunctionAssemblyData = Ptr{CeedQFunctionAssemblyData_private}

mutable struct CeedOperatorAssemblyData_private end

const CeedOperatorAssemblyData = Ptr{CeedOperatorAssemblyData_private}

function CeedStringAllocCopy(source, copy)
    ccall((:CeedStringAllocCopy, libceed), Cint, (Ptr{Cchar}, Ptr{Ptr{Cchar}}), source, copy)
end

function CeedFree(p)
    ccall((:CeedFree, libceed), Cint, (Ptr{Cvoid},), p)
end

function CeedObjectCreate(ceed, view_function, destroy_function, obj)
    ccall((:CeedObjectCreate, libceed), Cint, (Ceed, Ptr{Cvoid}, Ptr{Cvoid}, CeedObject), ceed, view_function, destroy_function, obj)
end

function CeedObjectReference(obj)
    ccall((:CeedObjectReference, libceed), Cint, (CeedObject,), obj)
end

function CeedObjectDereference(obj)
    ccall((:CeedObjectDereference, libceed), Cint, (CeedObject,), obj)
end

function CeedObjectDestroy_Private(obj)
    ccall((:CeedObjectDestroy_Private, libceed), Cint, (CeedObject,), obj)
end

function CeedSetHostBoolArray(source_array, copy_mode, num_values, target_array_owned, target_array_borrowed, target_array)
    ccall((:CeedSetHostBoolArray, libceed), Cint, (Ptr{Bool}, CeedCopyMode, CeedSize, Ptr{Ptr{Bool}}, Ptr{Ptr{Bool}}, Ptr{Ptr{Bool}}), source_array, copy_mode, num_values, target_array_owned, target_array_borrowed, target_array)
end

function CeedSetHostCeedInt8Array(source_array, copy_mode, num_values, target_array_owned, target_array_borrowed, target_array)
    ccall((:CeedSetHostCeedInt8Array, libceed), Cint, (Ptr{CeedInt8}, CeedCopyMode, CeedSize, Ptr{Ptr{CeedInt8}}, Ptr{Ptr{CeedInt8}}, Ptr{Ptr{CeedInt8}}), source_array, copy_mode, num_values, target_array_owned, target_array_borrowed, target_array)
end

function CeedSetHostCeedIntArray(source_array, copy_mode, num_values, target_array_owned, target_array_borrowed, target_array)
    ccall((:CeedSetHostCeedIntArray, libceed), Cint, (Ptr{CeedInt}, CeedCopyMode, CeedSize, Ptr{Ptr{CeedInt}}, Ptr{Ptr{CeedInt}}, Ptr{Ptr{CeedInt}}), source_array, copy_mode, num_values, target_array_owned, target_array_borrowed, target_array)
end

function CeedSetHostCeedScalarArray(source_array, copy_mode, num_values, target_array_owned, target_array_borrowed, target_array)
    ccall((:CeedSetHostCeedScalarArray, libceed), Cint, (Ptr{CeedScalar}, CeedCopyMode, CeedSize, Ptr{Ptr{CeedScalar}}, Ptr{Ptr{CeedScalar}}, Ptr{Ptr{CeedScalar}}), source_array, copy_mode, num_values, target_array_owned, target_array_borrowed, target_array)
end

function CeedRegister(prefix, init, priority)
    ccall((:CeedRegister, libceed), Cint, (Ptr{Cchar}, Ptr{Cvoid}, Cuint), prefix, init, priority)
end

function CeedRegisterImpl(prefix, init, priority)
    ccall((:CeedRegisterImpl, libceed), Cint, (Ptr{Cchar}, Ptr{Cvoid}, Cuint), prefix, init, priority)
end

function CeedIsDebug(ceed, is_debug)
    ccall((:CeedIsDebug, libceed), Cint, (Ceed, Ptr{Bool}), ceed, is_debug)
end

function CeedGetResourceRoot(ceed, resource, delineator, resource_root)
    ccall((:CeedGetResourceRoot, libceed), Cint, (Ceed, Ptr{Cchar}, Ptr{Cchar}, Ptr{Ptr{Cchar}}), ceed, resource, delineator, resource_root)
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

function CeedGetOperatorFallbackCeed(ceed, fallback_ceed)
    ccall((:CeedGetOperatorFallbackCeed, libceed), Cint, (Ceed, Ptr{Ceed}), ceed, fallback_ceed)
end

function CeedSetOperatorFallbackCeed(ceed, fallback_ceed)
    ccall((:CeedSetOperatorFallbackCeed, libceed), Cint, (Ceed, Ceed), ceed, fallback_ceed)
end

function CeedSetDeterministic(ceed, is_deterministic)
    ccall((:CeedSetDeterministic, libceed), Cint, (Ceed, Bool), ceed, is_deterministic)
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

function CeedGetWorkVector(ceed, len, vec)
    ccall((:CeedGetWorkVector, libceed), Cint, (Ceed, CeedSize, Ptr{CeedVector}), ceed, len, vec)
end

function CeedRestoreWorkVector(ceed, vec)
    ccall((:CeedRestoreWorkVector, libceed), Cint, (Ceed, Ptr{CeedVector}), ceed, vec)
end

function CeedClearWorkVectors(ceed, min_len)
    ccall((:CeedClearWorkVectors, libceed), Cint, (Ceed, CeedSize), ceed, min_len)
end

function CeedGetWorkVectorMemoryUsage(ceed, usage_mb)
    ccall((:CeedGetWorkVectorMemoryUsage, libceed), Cint, (Ceed, Ptr{CeedScalar}), ceed, usage_mb)
end

function CeedGetJitSourceRoots(ceed, num_source_roots, jit_source_roots)
    ccall((:CeedGetJitSourceRoots, libceed), Cint, (Ceed, Ptr{CeedInt}, Ptr{Ptr{Ptr{Cchar}}}), ceed, num_source_roots, jit_source_roots)
end

function CeedGetRustSourceRoots(ceed, num_source_roots, rust_source_roots)
    ccall((:CeedGetRustSourceRoots, libceed), Cint, (Ceed, Ptr{CeedInt}, Ptr{Ptr{Ptr{Cchar}}}), ceed, num_source_roots, rust_source_roots)
end

function CeedRestoreJitSourceRoots(ceed, jit_source_roots)
    ccall((:CeedRestoreJitSourceRoots, libceed), Cint, (Ceed, Ptr{Ptr{Ptr{Cchar}}}), ceed, jit_source_roots)
end

function CeedRestoreRustSourceRoots(ceed, rust_source_roots)
    ccall((:CeedRestoreRustSourceRoots, libceed), Cint, (Ceed, Ptr{Ptr{Ptr{Cchar}}}), ceed, rust_source_roots)
end

function CeedGetJitDefines(ceed, num_defines, jit_defines)
    ccall((:CeedGetJitDefines, libceed), Cint, (Ceed, Ptr{CeedInt}, Ptr{Ptr{Ptr{Cchar}}}), ceed, num_defines, jit_defines)
end

function CeedRestoreJitDefines(ceed, jit_defines)
    ccall((:CeedRestoreJitDefines, libceed), Cint, (Ceed, Ptr{Ptr{Ptr{Cchar}}}), ceed, jit_defines)
end

function CeedVectorHasValidArray(vec, has_valid_array)
    ccall((:CeedVectorHasValidArray, libceed), Cint, (CeedVector, Ptr{Bool}), vec, has_valid_array)
end

function CeedVectorHasBorrowedArrayOfType(vec, mem_type, has_borrowed_array_of_type)
    ccall((:CeedVectorHasBorrowedArrayOfType, libceed), Cint, (CeedVector, CeedMemType, Ptr{Bool}), vec, mem_type, has_borrowed_array_of_type)
end

function CeedVectorGetState(vec, state)
    ccall((:CeedVectorGetState, libceed), Cint, (CeedVector, Ptr{UInt64}), vec, state)
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

@cenum CeedRestrictionType::UInt32 begin
    CEED_RESTRICTION_STANDARD = 1
    CEED_RESTRICTION_ORIENTED = 2
    CEED_RESTRICTION_CURL_ORIENTED = 3
    CEED_RESTRICTION_STRIDED = 4
    CEED_RESTRICTION_POINTS = 5
end

function CeedElemRestrictionGetType(rstr, rstr_type)
    ccall((:CeedElemRestrictionGetType, libceed), Cint, (CeedElemRestriction, Ptr{CeedRestrictionType}), rstr, rstr_type)
end

function CeedElemRestrictionIsStrided(rstr, is_strided)
    ccall((:CeedElemRestrictionIsStrided, libceed), Cint, (CeedElemRestriction, Ptr{Bool}), rstr, is_strided)
end

function CeedElemRestrictionIsAtPoints(rstr, is_points)
    ccall((:CeedElemRestrictionIsAtPoints, libceed), Cint, (CeedElemRestriction, Ptr{Bool}), rstr, is_points)
end

function CeedElemRestrictionAtPointsAreCompatible(rstr_a, rstr_b, are_compatible)
    ccall((:CeedElemRestrictionAtPointsAreCompatible, libceed), Cint, (CeedElemRestriction, CeedElemRestriction, Ptr{Bool}), rstr_a, rstr_b, are_compatible)
end

function CeedElemRestrictionGetStrides(rstr, strides)
    ccall((:CeedElemRestrictionGetStrides, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, strides)
end

function CeedElemRestrictionHasBackendStrides(rstr, has_backend_strides)
    ccall((:CeedElemRestrictionHasBackendStrides, libceed), Cint, (CeedElemRestriction, Ptr{Bool}), rstr, has_backend_strides)
end

function CeedElemRestrictionGetOffsets(rstr, mem_type, offsets)
    ccall((:CeedElemRestrictionGetOffsets, libceed), Cint, (CeedElemRestriction, CeedMemType, Ptr{Ptr{CeedInt}}), rstr, mem_type, offsets)
end

function CeedElemRestrictionRestoreOffsets(rstr, offsets)
    ccall((:CeedElemRestrictionRestoreOffsets, libceed), Cint, (CeedElemRestriction, Ptr{Ptr{CeedInt}}), rstr, offsets)
end

function CeedElemRestrictionGetOrientations(rstr, mem_type, orients)
    ccall((:CeedElemRestrictionGetOrientations, libceed), Cint, (CeedElemRestriction, CeedMemType, Ptr{Ptr{Bool}}), rstr, mem_type, orients)
end

function CeedElemRestrictionRestoreOrientations(rstr, orients)
    ccall((:CeedElemRestrictionRestoreOrientations, libceed), Cint, (CeedElemRestriction, Ptr{Ptr{Bool}}), rstr, orients)
end

function CeedElemRestrictionGetCurlOrientations(rstr, mem_type, curl_orients)
    ccall((:CeedElemRestrictionGetCurlOrientations, libceed), Cint, (CeedElemRestriction, CeedMemType, Ptr{Ptr{CeedInt8}}), rstr, mem_type, curl_orients)
end

function CeedElemRestrictionRestoreCurlOrientations(rstr, curl_orients)
    ccall((:CeedElemRestrictionRestoreCurlOrientations, libceed), Cint, (CeedElemRestriction, Ptr{Ptr{CeedInt8}}), rstr, curl_orients)
end

function CeedElemRestrictionGetLLayout(rstr, layout)
    ccall((:CeedElemRestrictionGetLLayout, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, layout)
end

function CeedElemRestrictionSetLLayout(rstr, layout)
    ccall((:CeedElemRestrictionSetLLayout, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, layout)
end

function CeedElemRestrictionGetELayout(rstr, layout)
    ccall((:CeedElemRestrictionGetELayout, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, layout)
end

function CeedElemRestrictionSetELayout(rstr, layout)
    ccall((:CeedElemRestrictionSetELayout, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, layout)
end

function CeedElemRestrictionGetAtPointsElementOffset(rstr, elem, elem_offset)
    ccall((:CeedElemRestrictionGetAtPointsElementOffset, libceed), Cint, (CeedElemRestriction, CeedInt, Ptr{CeedSize}), rstr, elem, elem_offset)
end

function CeedElemRestrictionSetAtPointsEVectorSize(rstr, e_size)
    ccall((:CeedElemRestrictionSetAtPointsEVectorSize, libceed), Cint, (CeedElemRestriction, CeedSize), rstr, e_size)
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

function CeedElemRestrictionGetFlopsEstimate(rstr, t_mode, flops)
    ccall((:CeedElemRestrictionGetFlopsEstimate, libceed), Cint, (CeedElemRestriction, CeedTransposeMode, Ptr{CeedSize}), rstr, t_mode, flops)
end

@cenum CeedFESpace::UInt32 begin
    CEED_FE_SPACE_H1 = 1
    CEED_FE_SPACE_HDIV = 2
    CEED_FE_SPACE_HCURL = 3
end

function CeedBasisGetCollocatedGrad(basis, colo_grad_1d)
    ccall((:CeedBasisGetCollocatedGrad, libceed), Cint, (CeedBasis, Ptr{CeedScalar}), basis, colo_grad_1d)
end

function CeedBasisGetChebyshevInterp1D(basis, chebyshev_interp_1d)
    ccall((:CeedBasisGetChebyshevInterp1D, libceed), Cint, (CeedBasis, Ptr{CeedScalar}), basis, chebyshev_interp_1d)
end

function CeedBasisIsTensor(basis, is_tensor)
    ccall((:CeedBasisIsTensor, libceed), Cint, (CeedBasis, Ptr{Bool}), basis, is_tensor)
end

function CeedBasisIsCollocated(basis, is_collocated)
    ccall((:CeedBasisIsCollocated, libceed), Cint, (CeedBasis, Ptr{Bool}), basis, is_collocated)
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

function CeedBasisGetNumQuadratureComponents(basis, eval_mode, q_comp)
    ccall((:CeedBasisGetNumQuadratureComponents, libceed), Cint, (CeedBasis, CeedEvalMode, Ptr{CeedInt}), basis, eval_mode, q_comp)
end

function CeedBasisGetFlopsEstimate(basis, t_mode, eval_mode, is_at_points, num_points, flops)
    ccall((:CeedBasisGetFlopsEstimate, libceed), Cint, (CeedBasis, CeedTransposeMode, CeedEvalMode, Bool, CeedInt, Ptr{CeedSize}), basis, t_mode, eval_mode, is_at_points, num_points, flops)
end

function CeedBasisGetFESpace(basis, fe_space)
    ccall((:CeedBasisGetFESpace, libceed), Cint, (CeedBasis, Ptr{CeedFESpace}), basis, fe_space)
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

function CeedBasisCreateH1Fallback(ceed, topo, num_comp, num_nodes, nqpts, interp, grad, q_ref, q_weights, basis)
    ccall((:CeedBasisCreateH1Fallback, libceed), Cint, (Ceed, CeedElemTopology, CeedInt, CeedInt, CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedBasis), ceed, topo, num_comp, num_nodes, nqpts, interp, grad, q_ref, q_weights, basis)
end

function CeedTensorContractCreate(ceed, contract)
    ccall((:CeedTensorContractCreate, libceed), Cint, (Ceed, Ptr{CeedTensorContract}), ceed, contract)
end

function CeedTensorContractApply(contract, A, B, C, J, t, t_mode, Add, u, v)
    ccall((:CeedTensorContractApply, libceed), Cint, (CeedTensorContract, CeedInt, CeedInt, CeedInt, CeedInt, Ptr{CeedScalar}, CeedTransposeMode, CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}), contract, A, B, C, J, t, t_mode, Add, u, v)
end

function CeedTensorContractStridedApply(contract, A, B, C, D, J, t, t_mode, add, u, v)
    ccall((:CeedTensorContractStridedApply, libceed), Cint, (CeedTensorContract, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, Ptr{CeedScalar}, CeedTransposeMode, CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}), contract, A, B, C, D, J, t, t_mode, add, u, v)
end

function CeedTensorContractGetCeed(contract, ceed)
    ccall((:CeedTensorContractGetCeed, libceed), Cint, (CeedTensorContract, Ptr{Ceed}), contract, ceed)
end

function CeedTensorContractReturnCeed(contract)
    ccall((:CeedTensorContractReturnCeed, libceed), Ceed, (CeedTensorContract,), contract)
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

function CeedTensorContractReferenceCopy(tensor, tensor_copy)
    ccall((:CeedTensorContractReferenceCopy, libceed), Cint, (CeedTensorContract, Ptr{CeedTensorContract}), tensor, tensor_copy)
end

function CeedTensorContractDestroy(contract)
    ccall((:CeedTensorContractDestroy, libceed), Cint, (Ptr{CeedTensorContract},), contract)
end

function CeedQFunctionRegister(name, source, vec_length, f, init)
    ccall((:CeedQFunctionRegister, libceed), Cint, (Ptr{Cchar}, Ptr{Cchar}, CeedInt, CeedQFunctionUser, Ptr{Cvoid}), name, source, vec_length, f, init)
end

function CeedQFunctionSetFortranStatus(qf, status)
    ccall((:CeedQFunctionSetFortranStatus, libceed), Cint, (CeedQFunction, Bool), qf, status)
end

function CeedQFunctionGetVectorLength(qf, vec_length)
    ccall((:CeedQFunctionGetVectorLength, libceed), Cint, (CeedQFunction, Ptr{CeedInt}), qf, vec_length)
end

function CeedQFunctionGetNumArgs(qf, num_input_fields, num_output_fields)
    ccall((:CeedQFunctionGetNumArgs, libceed), Cint, (CeedQFunction, Ptr{CeedInt}, Ptr{CeedInt}), qf, num_input_fields, num_output_fields)
end

function CeedQFunctionGetKernelName(qf, kernel_name)
    ccall((:CeedQFunctionGetKernelName, libceed), Cint, (CeedQFunction, Ptr{Ptr{Cchar}}), qf, kernel_name)
end

function CeedQFunctionGetName(qf, name)
    ccall((:CeedQFunctionGetName, libceed), Cint, (CeedQFunction, Ptr{Ptr{Cchar}}), qf, name)
end

function CeedQFunctionGetSourcePath(qf, source_path)
    ccall((:CeedQFunctionGetSourcePath, libceed), Cint, (CeedQFunction, Ptr{Ptr{Cchar}}), qf, source_path)
end

function CeedQFunctionGetUserFunction(qf, f)
    ccall((:CeedQFunctionGetUserFunction, libceed), Cint, (CeedQFunction, Ptr{CeedQFunctionUser}), qf, f)
end

function CeedQFunctionGetContext(qf, ctx)
    ccall((:CeedQFunctionGetContext, libceed), Cint, (CeedQFunction, Ptr{CeedQFunctionContext}), qf, ctx)
end

function CeedQFunctionGetContextData(qf, mem_type, data)
    ccall((:CeedQFunctionGetContextData, libceed), Cint, (CeedQFunction, CeedMemType, Ptr{Cvoid}), qf, mem_type, data)
end

function CeedQFunctionRestoreContextData(qf, data)
    ccall((:CeedQFunctionRestoreContextData, libceed), Cint, (CeedQFunction, Ptr{Cvoid}), qf, data)
end

function CeedQFunctionGetInnerContext(qf, ctx)
    ccall((:CeedQFunctionGetInnerContext, libceed), Cint, (CeedQFunction, Ptr{CeedQFunctionContext}), qf, ctx)
end

function CeedQFunctionGetInnerContextData(qf, mem_type, data)
    ccall((:CeedQFunctionGetInnerContextData, libceed), Cint, (CeedQFunction, CeedMemType, Ptr{Cvoid}), qf, mem_type, data)
end

function CeedQFunctionRestoreInnerContextData(qf, data)
    ccall((:CeedQFunctionRestoreInnerContextData, libceed), Cint, (CeedQFunction, Ptr{Cvoid}), qf, data)
end

function CeedQFunctionIsIdentity(qf, is_identity)
    ccall((:CeedQFunctionIsIdentity, libceed), Cint, (CeedQFunction, Ptr{Bool}), qf, is_identity)
end

function CeedQFunctionIsContextWritable(qf, is_writable)
    ccall((:CeedQFunctionIsContextWritable, libceed), Cint, (CeedQFunction, Ptr{Bool}), qf, is_writable)
end

function CeedQFunctionGetData(qf, data)
    ccall((:CeedQFunctionGetData, libceed), Cint, (CeedQFunction, Ptr{Cvoid}), qf, data)
end

function CeedQFunctionSetData(qf, data)
    ccall((:CeedQFunctionSetData, libceed), Cint, (CeedQFunction, Ptr{Cvoid}), qf, data)
end

function CeedQFunctionIsImmutable(qf, is_immutable)
    ccall((:CeedQFunctionIsImmutable, libceed), Cint, (CeedQFunction, Ptr{Bool}), qf, is_immutable)
end

function CeedQFunctionSetImmutable(qf)
    ccall((:CeedQFunctionSetImmutable, libceed), Cint, (CeedQFunction,), qf)
end

function CeedQFunctionReference(qf)
    ccall((:CeedQFunctionReference, libceed), Cint, (CeedQFunction,), qf)
end

function CeedQFunctionGetFlopsEstimate(qf, flops)
    ccall((:CeedQFunctionGetFlopsEstimate, libceed), Cint, (CeedQFunction, Ptr{CeedSize}), qf, flops)
end

function CeedQFunctionContextGetCeed(ctx, ceed)
    ccall((:CeedQFunctionContextGetCeed, libceed), Cint, (CeedQFunctionContext, Ptr{Ceed}), ctx, ceed)
end

function CeedQFunctionContextReturnCeed(ctx)
    ccall((:CeedQFunctionContextReturnCeed, libceed), Ceed, (CeedQFunctionContext,), ctx)
end

function CeedQFunctionContextHasValidData(ctx, has_valid_data)
    ccall((:CeedQFunctionContextHasValidData, libceed), Cint, (CeedQFunctionContext, Ptr{Bool}), ctx, has_valid_data)
end

function CeedQFunctionContextHasBorrowedDataOfType(ctx, mem_type, has_borrowed_data_of_type)
    ccall((:CeedQFunctionContextHasBorrowedDataOfType, libceed), Cint, (CeedQFunctionContext, CeedMemType, Ptr{Bool}), ctx, mem_type, has_borrowed_data_of_type)
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

function CeedQFunctionContextGetFieldLabel(ctx, field_name, field_label)
    ccall((:CeedQFunctionContextGetFieldLabel, libceed), Cint, (CeedQFunctionContext, Ptr{Cchar}, Ptr{CeedContextFieldLabel}), ctx, field_name, field_label)
end

function CeedQFunctionContextSetGeneric(ctx, field_label, field_type, value)
    ccall((:CeedQFunctionContextSetGeneric, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, CeedContextFieldType, Ptr{Cvoid}), ctx, field_label, field_type, value)
end

function CeedQFunctionContextGetGenericRead(ctx, field_label, field_type, num_values, value)
    ccall((:CeedQFunctionContextGetGenericRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, CeedContextFieldType, Ptr{Csize_t}, Ptr{Cvoid}), ctx, field_label, field_type, num_values, value)
end

function CeedQFunctionContextRestoreGenericRead(ctx, field_label, field_type, value)
    ccall((:CeedQFunctionContextRestoreGenericRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, CeedContextFieldType, Ptr{Cvoid}), ctx, field_label, field_type, value)
end

function CeedQFunctionContextSetCeedByte(ctx, field_label, values)
    ccall((:CeedQFunctionContextSetCeedByte, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Cchar}), ctx, field_label, values)
end

function CeedQFunctionContextGetCeedByteRead(ctx, field_label, num_values, values)
    ccall((:CeedQFunctionContextGetCeedByteRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Cchar}}), ctx, field_label, num_values, values)
end

function CeedQFunctionContextRestoreCeedByteRead(ctx, field_label, values)
    ccall((:CeedQFunctionContextRestoreCeedByteRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Ptr{Cchar}}), ctx, field_label, values)
end

function CeedQFunctionContextSetCeedScalar(ctx, field_label, values)
    ccall((:CeedQFunctionContextSetCeedScalar, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{CeedScalar}), ctx, field_label, values)
end

function CeedQFunctionContextGetCeedScalarRead(ctx, field_label, num_values, values)
    ccall((:CeedQFunctionContextGetCeedScalarRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{CeedScalar}}), ctx, field_label, num_values, values)
end

function CeedQFunctionContextRestoreCeedScalarRead(ctx, field_label, values)
    ccall((:CeedQFunctionContextRestoreCeedScalarRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Ptr{CeedScalar}}), ctx, field_label, values)
end

function CeedQFunctionContextSetCeedFloat(ctx, field_label, values)
    ccall((:CeedQFunctionContextSetCeedFloat, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Cfloat}), ctx, field_label, values)
end

function CeedQFunctionContextGetCeedFloatRead(ctx, field_label, num_values, values)
    ccall((:CeedQFunctionContextGetCeedFloatRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Cfloat}}), ctx, field_label, num_values, values)
end

function CeedQFunctionContextRestoreCeedFloatRead(ctx, field_label, values)
    ccall((:CeedQFunctionContextRestoreCeedFloatRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Ptr{Cfloat}}), ctx, field_label, values)
end

function CeedQFunctionContextSetCeedDouble(ctx, field_label, values)
    ccall((:CeedQFunctionContextSetCeedDouble, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Cdouble}), ctx, field_label, values)
end

function CeedQFunctionContextGetCeedDoubleRead(ctx, field_label, num_values, values)
    ccall((:CeedQFunctionContextGetCeedDoubleRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Cdouble}}), ctx, field_label, num_values, values)
end

function CeedQFunctionContextRestoreCeedDoubleRead(ctx, field_label, values)
    ccall((:CeedQFunctionContextRestoreCeedDoubleRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Ptr{Cdouble}}), ctx, field_label, values)
end

function CeedQFunctionContextSetCeedInt8(ctx, field_label, values)
    ccall((:CeedQFunctionContextSetCeedInt8, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{CeedInt8}), ctx, field_label, values)
end

function CeedQFunctionContextGetCeedInt8Read(ctx, field_label, num_values, values)
    ccall((:CeedQFunctionContextGetCeedInt8Read, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{CeedInt8}}), ctx, field_label, num_values, values)
end

function CeedQFunctionContextRestoreCeedInt8Read(ctx, field_label, values)
    ccall((:CeedQFunctionContextRestoreCeedInt8Read, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Ptr{CeedInt8}}), ctx, field_label, values)
end

function CeedQFunctionContextSetCeedInt(ctx, field_label, values)
    ccall((:CeedQFunctionContextSetCeedInt, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{CeedInt}), ctx, field_label, values)
end

function CeedQFunctionContextGetCeedIntRead(ctx, field_label, num_values, values)
    ccall((:CeedQFunctionContextGetCeedIntRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{CeedInt}}), ctx, field_label, num_values, values)
end

function CeedQFunctionContextRestoreCeedIntRead(ctx, field_label, values)
    ccall((:CeedQFunctionContextRestoreCeedIntRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Ptr{CeedInt}}), ctx, field_label, values)
end

function CeedQFunctionContextSetCeedInt32(ctx, field_label, values)
    ccall((:CeedQFunctionContextSetCeedInt32, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Int32}), ctx, field_label, values)
end

function CeedQFunctionContextGetCeedInt32Read(ctx, field_label, num_values, values)
    ccall((:CeedQFunctionContextGetCeedInt32Read, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Int32}}), ctx, field_label, num_values, values)
end

function CeedQFunctionContextRestoreCeedInt32Read(ctx, field_label, values)
    ccall((:CeedQFunctionContextRestoreCeedInt32Read, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Ptr{Int32}}), ctx, field_label, values)
end

function CeedQFunctionContextSetCeedInt64(ctx, field_label, values)
    ccall((:CeedQFunctionContextSetCeedInt64, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Int64}), ctx, field_label, values)
end

function CeedQFunctionContextGetCeedInt64Read(ctx, field_label, num_values, values)
    ccall((:CeedQFunctionContextGetCeedInt64Read, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Int64}}), ctx, field_label, num_values, values)
end

function CeedQFunctionContextRestoreCeedInt64Read(ctx, field_label, values)
    ccall((:CeedQFunctionContextRestoreCeedInt64Read, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Ptr{Int64}}), ctx, field_label, values)
end

function CeedQFunctionContextSetCeedSize(ctx, field_label, values)
    ccall((:CeedQFunctionContextSetCeedSize, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{CeedSize}), ctx, field_label, values)
end

function CeedQFunctionContextGetCeedSizeRead(ctx, field_label, num_values, values)
    ccall((:CeedQFunctionContextGetCeedSizeRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{CeedSize}}), ctx, field_label, num_values, values)
end

function CeedQFunctionContextRestoreCeedSizeRead(ctx, field_label, values)
    ccall((:CeedQFunctionContextRestoreCeedSizeRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Ptr{CeedSize}}), ctx, field_label, values)
end

function CeedQFunctionContextSetCeedBoolean(ctx, field_label, values)
    ccall((:CeedQFunctionContextSetCeedBoolean, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Bool}), ctx, field_label, values)
end

function CeedQFunctionContextGetCeedBooleanRead(ctx, field_label, num_values, values)
    ccall((:CeedQFunctionContextGetCeedBooleanRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Csize_t}, Ptr{Ptr{Bool}}), ctx, field_label, num_values, values)
end

function CeedQFunctionContextRestoreCeedBooleanRead(ctx, field_label, values)
    ccall((:CeedQFunctionContextRestoreCeedBooleanRead, libceed), Cint, (CeedQFunctionContext, CeedContextFieldLabel, Ptr{Ptr{Bool}}), ctx, field_label, values)
end

function CeedQFunctionContextGetDataDestroy(ctx, f_mem_type, f)
    ccall((:CeedQFunctionContextGetDataDestroy, libceed), Cint, (CeedQFunctionContext, Ptr{CeedMemType}, Ptr{CeedQFunctionContextDataDestroyUser}), ctx, f_mem_type, f)
end

function CeedQFunctionContextReference(ctx)
    ccall((:CeedQFunctionContextReference, libceed), Cint, (CeedQFunctionContext,), ctx)
end

function CeedOperatorGetBasisPointer(basis, eval_mode, identity, basis_ptr)
    ccall((:CeedOperatorGetBasisPointer, libceed), Cint, (CeedBasis, CeedEvalMode, Ptr{CeedScalar}, Ptr{Ptr{CeedScalar}}), basis, eval_mode, identity, basis_ptr)
end

function CeedOperatorCreateActivePointBlockRestriction(rstr, pointblock_rstr)
    ccall((:CeedOperatorCreateActivePointBlockRestriction, libceed), Cint, (CeedElemRestriction, Ptr{CeedElemRestriction}), rstr, pointblock_rstr)
end

function CeedOperatorGetQFunctionAssemblyData(op, data)
    ccall((:CeedOperatorGetQFunctionAssemblyData, libceed), Cint, (CeedOperator, Ptr{CeedQFunctionAssemblyData}), op, data)
end

function CeedQFunctionAssemblyDataCreate(ceed, data)
    ccall((:CeedQFunctionAssemblyDataCreate, libceed), Cint, (Ceed, Ptr{CeedQFunctionAssemblyData}), ceed, data)
end

function CeedQFunctionAssemblyDataReference(data)
    ccall((:CeedQFunctionAssemblyDataReference, libceed), Cint, (CeedQFunctionAssemblyData,), data)
end

function CeedQFunctionAssemblyDataSetReuse(data, reuse_assembly_data)
    ccall((:CeedQFunctionAssemblyDataSetReuse, libceed), Cint, (CeedQFunctionAssemblyData, Bool), data, reuse_assembly_data)
end

function CeedQFunctionAssemblyDataSetUpdateNeeded(data, needs_data_update)
    ccall((:CeedQFunctionAssemblyDataSetUpdateNeeded, libceed), Cint, (CeedQFunctionAssemblyData, Bool), data, needs_data_update)
end

function CeedQFunctionAssemblyDataIsUpdateNeeded(data, is_update_needed)
    ccall((:CeedQFunctionAssemblyDataIsUpdateNeeded, libceed), Cint, (CeedQFunctionAssemblyData, Ptr{Bool}), data, is_update_needed)
end

function CeedQFunctionAssemblyDataReferenceCopy(data, data_copy)
    ccall((:CeedQFunctionAssemblyDataReferenceCopy, libceed), Cint, (CeedQFunctionAssemblyData, Ptr{CeedQFunctionAssemblyData}), data, data_copy)
end

function CeedQFunctionAssemblyDataIsSetup(data, is_setup)
    ccall((:CeedQFunctionAssemblyDataIsSetup, libceed), Cint, (CeedQFunctionAssemblyData, Ptr{Bool}), data, is_setup)
end

function CeedQFunctionAssemblyDataSetBlockAssembling(data, is_block_assembling)
    ccall((:CeedQFunctionAssemblyDataSetBlockAssembling, libceed), Cint, (CeedQFunctionAssemblyData, Bool), data, is_block_assembling)
end

function CeedQFunctionAssemblyDataSetObjects(data, vec, rstr)
    ccall((:CeedQFunctionAssemblyDataSetObjects, libceed), Cint, (CeedQFunctionAssemblyData, CeedVector, CeedElemRestriction), data, vec, rstr)
end

function CeedQFunctionAssemblyDataGetObjects(data, vec, rstr)
    ccall((:CeedQFunctionAssemblyDataGetObjects, libceed), Cint, (CeedQFunctionAssemblyData, Ptr{CeedVector}, Ptr{CeedElemRestriction}), data, vec, rstr)
end

function CeedQFunctionAssemblyDataDestroy(data)
    ccall((:CeedQFunctionAssemblyDataDestroy, libceed), Cint, (Ptr{CeedQFunctionAssemblyData},), data)
end

function CeedOperatorGetOperatorAssemblyData(op, data)
    ccall((:CeedOperatorGetOperatorAssemblyData, libceed), Cint, (CeedOperator, Ptr{CeedOperatorAssemblyData}), op, data)
end

function CeedOperatorAssemblyDataCreate(ceed, op, data)
    ccall((:CeedOperatorAssemblyDataCreate, libceed), Cint, (Ceed, CeedOperator, Ptr{CeedOperatorAssemblyData}), ceed, op, data)
end

function CeedOperatorAssemblyDataGetEvalModes(data, num_active_bases_in, num_eval_modes_in, eval_modes_in, eval_mode_offsets_in, num_active_bases_out, num_eval_modes_out, eval_modes_out, eval_mode_offsets_out, num_output_components)
    ccall((:CeedOperatorAssemblyDataGetEvalModes, libceed), Cint, (CeedOperatorAssemblyData, Ptr{CeedInt}, Ptr{Ptr{CeedInt}}, Ptr{Ptr{Ptr{CeedEvalMode}}}, Ptr{Ptr{Ptr{CeedSize}}}, Ptr{CeedInt}, Ptr{Ptr{CeedInt}}, Ptr{Ptr{Ptr{CeedEvalMode}}}, Ptr{Ptr{Ptr{CeedSize}}}, Ptr{CeedSize}), data, num_active_bases_in, num_eval_modes_in, eval_modes_in, eval_mode_offsets_in, num_active_bases_out, num_eval_modes_out, eval_modes_out, eval_mode_offsets_out, num_output_components)
end

function CeedOperatorAssemblyDataGetBases(data, num_active_bases_in, active_bases_in, assembled_bases_in, num_active_bases_out, active_bases_out, assembled_bases_out)
    ccall((:CeedOperatorAssemblyDataGetBases, libceed), Cint, (CeedOperatorAssemblyData, Ptr{CeedInt}, Ptr{Ptr{CeedBasis}}, Ptr{Ptr{Ptr{CeedScalar}}}, Ptr{CeedInt}, Ptr{Ptr{CeedBasis}}, Ptr{Ptr{Ptr{CeedScalar}}}), data, num_active_bases_in, active_bases_in, assembled_bases_in, num_active_bases_out, active_bases_out, assembled_bases_out)
end

function CeedOperatorAssemblyDataGetElemRestrictions(data, num_active_elem_rstrs_in, active_elem_rstrs_in, num_active_elem_rstrs_out, active_elem_rstrs_out)
    ccall((:CeedOperatorAssemblyDataGetElemRestrictions, libceed), Cint, (CeedOperatorAssemblyData, Ptr{CeedInt}, Ptr{Ptr{CeedElemRestriction}}, Ptr{CeedInt}, Ptr{Ptr{CeedElemRestriction}}), data, num_active_elem_rstrs_in, active_elem_rstrs_in, num_active_elem_rstrs_out, active_elem_rstrs_out)
end

function CeedOperatorAssemblyDataDestroy(data)
    ccall((:CeedOperatorAssemblyDataDestroy, libceed), Cint, (Ptr{CeedOperatorAssemblyData},), data)
end

function CeedOperatorGetActiveBasis(op, active_basis)
    ccall((:CeedOperatorGetActiveBasis, libceed), Cint, (CeedOperator, Ptr{CeedBasis}), op, active_basis)
end

function CeedOperatorGetActiveBases(op, active_input_basis, active_output_basis)
    ccall((:CeedOperatorGetActiveBases, libceed), Cint, (CeedOperator, Ptr{CeedBasis}, Ptr{CeedBasis}), op, active_input_basis, active_output_basis)
end

function CeedOperatorGetActiveElemRestriction(op, active_rstr)
    ccall((:CeedOperatorGetActiveElemRestriction, libceed), Cint, (CeedOperator, Ptr{CeedElemRestriction}), op, active_rstr)
end

function CeedOperatorGetActiveElemRestrictions(op, active_input_rstr, active_output_rstr)
    ccall((:CeedOperatorGetActiveElemRestrictions, libceed), Cint, (CeedOperator, Ptr{CeedElemRestriction}, Ptr{CeedElemRestriction}), op, active_input_rstr, active_output_rstr)
end

function CeedOperatorGetNumArgs(op, num_args)
    ccall((:CeedOperatorGetNumArgs, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, num_args)
end

function CeedOperatorHasTensorBases(op, has_tensor_bases)
    ccall((:CeedOperatorHasTensorBases, libceed), Cint, (CeedOperator, Ptr{Bool}), op, has_tensor_bases)
end

function CeedOperatorIsImmutable(op, is_immutable)
    ccall((:CeedOperatorIsImmutable, libceed), Cint, (CeedOperator, Ptr{Bool}), op, is_immutable)
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

function CeedOperatorGetData(op, data)
    ccall((:CeedOperatorGetData, libceed), Cint, (CeedOperator, Ptr{Cvoid}), op, data)
end

function CeedOperatorSetData(op, data)
    ccall((:CeedOperatorSetData, libceed), Cint, (CeedOperator, Ptr{Cvoid}), op, data)
end

function CeedOperatorReference(op)
    ccall((:CeedOperatorReference, libceed), Cint, (CeedOperator,), op)
end

function CeedOperatorGetFallback(op, op_fallback)
    ccall((:CeedOperatorGetFallback, libceed), Cint, (CeedOperator, Ptr{CeedOperator}), op, op_fallback)
end

function CeedOperatorGetFallbackParent(op, parent)
    ccall((:CeedOperatorGetFallbackParent, libceed), Cint, (CeedOperator, Ptr{CeedOperator}), op, parent)
end

function CeedOperatorGetFallbackParentCeed(op, parent)
    ccall((:CeedOperatorGetFallbackParentCeed, libceed), Cint, (CeedOperator, Ptr{Ceed}), op, parent)
end

function CeedOperatorLinearAssembleQFunctionBuildOrUpdateFallback(op, build_objects, assembled, rstr, request)
    ccall((:CeedOperatorLinearAssembleQFunctionBuildOrUpdateFallback, libceed), Cint, (CeedOperator, Bool, Ptr{CeedVector}, Ptr{CeedElemRestriction}, Ptr{CeedRequest}), op, build_objects, assembled, rstr, request)
end

function CeedOperatorAssembleSingle(op, offset, values)
    ccall((:CeedOperatorAssembleSingle, libceed), Cint, (CeedOperator, CeedSize, CeedVector), op, offset, values)
end

function CeedOperatorSetSetupDone(op)
    ccall((:CeedOperatorSetSetupDone, libceed), Cint, (CeedOperator,), op)
end

function CeedMatrixMatrixMultiply(ceed, mat_A, mat_B, mat_C, m, n, kk)
    ccall((:CeedMatrixMatrixMultiply, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt, CeedInt, CeedInt), ceed, mat_A, mat_B, mat_C, m, n, kk)
end

function CeedQRFactorization(ceed, mat, tau, m, n)
    ccall((:CeedQRFactorization, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt, CeedInt), ceed, mat, tau, m, n)
end

function CeedHouseholderApplyQ(mat_A, mat_Q, tau, t_mode, m, n, k, row, col)
    ccall((:CeedHouseholderApplyQ, libceed), Cint, (Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedTransposeMode, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt), mat_A, mat_Q, tau, t_mode, m, n, k, row, col)
end

function CeedMatrixPseudoinverse(ceed, mat, m, n, mat_pinv)
    ccall((:CeedMatrixPseudoinverse, libceed), Cint, (Ceed, Ptr{CeedScalar}, CeedInt, CeedInt, Ptr{CeedScalar}), ceed, mat, m, n, mat_pinv)
end

function CeedSymmetricSchurDecomposition(ceed, mat, lambda, n)
    ccall((:CeedSymmetricSchurDecomposition, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt), ceed, mat, lambda, n)
end

function CeedSimultaneousDiagonalization(ceed, mat_A, mat_B, x, lambda, n)
    ccall((:CeedSimultaneousDiagonalization, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt), ceed, mat_A, mat_B, x, lambda, n)
end

# Skipping MacroDefinition: CEED_EXTERN extern CEED_VISIBILITY ( default )

# Skipping MacroDefinition: CEED_QFUNCTION_HELPER_ATTR CEED_QFUNCTION_ATTR __attribute__ ( ( always_inline ) )

# Skipping MacroDefinition: CEED_QFUNCTION_HELPER CEED_QFUNCTION_HELPER_ATTR static inline

# Skipping MacroDefinition: CeedPragmaSIMD _Pragma ( "clang loop vectorize(enable)" )

const CeedInt_FMT = "d"

const CeedSize_FMT = "td"

const CeedInt8_FMT = "d"

const CEED_VERSION_MAJOR = 0

const CEED_VERSION_MINOR = 12

const CEED_VERSION_PATCH = 0

const CEED_VERSION_RELEASE = false

# Skipping MacroDefinition: DEPRECATED ( msg ) __attribute__ ( ( deprecated ( msg ) ) )

# Skipping MacroDefinition: CEED_INTERN extern CEED_VISIBILITY ( hidden )

# Skipping MacroDefinition: CEED_UNUSED __attribute__ ( ( unused ) )

const CEED_MAX_RESOURCE_LEN = 1024

const CEED_MAX_BACKEND_PRIORITY = UINT_MAX

const CEED_COMPOSITE_MAX = 16

const CEED_FIELD_MAX = 16

# Skipping MacroDefinition: CeedPragmaOptimizeOff _Pragma ( "clang optimize off" )

# Skipping MacroDefinition: CeedPragmaOptimizeOn _Pragma ( "clang optimize on" )

const CEED_TAB_WIDTH = 2
