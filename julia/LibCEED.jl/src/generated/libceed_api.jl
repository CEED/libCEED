# Julia wrapper for header: ceed.h
# Automatically generated using Clang.jl
#! format: off

function CeedInit(resource, ceed)
    ccall((:CeedInit, libceed), Cint, (Cstring, Ptr{Ceed}), resource, ceed)
end

function CeedGetResource(ceed, resource)
    ccall((:CeedGetResource, libceed), Cint, (Ceed, Ptr{Cstring}), ceed, resource)
end

function CeedIsDeterministic(ceed, isDeterministic)
    ccall((:CeedIsDeterministic, libceed), Cint, (Ceed, Ptr{Bool}), ceed, isDeterministic)
end

function CeedView(ceed, stream)
    ccall((:CeedView, libceed), Cint, (Ceed, Ptr{FILE}), ceed, stream)
end

function CeedDestroy(ceed)
    ccall((:CeedDestroy, libceed), Cint, (Ptr{Ceed},), ceed)
end

function CeedErrorReturn(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:CeedErrorReturn, libceed), Cint, (Ceed, Cstring, Cint, Cstring, Cint, Cstring, Ptr{Cvoid}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function CeedErrorStore(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:CeedErrorStore, libceed), Cint, (Ceed, Cstring, Cint, Cstring, Cint, Cstring, Ptr{Cvoid}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function CeedErrorAbort(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:CeedErrorAbort, libceed), Cint, (Ceed, Cstring, Cint, Cstring, Cint, Cstring, Ptr{Cvoid}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function CeedErrorExit(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:CeedErrorExit, libceed), Cint, (Ceed, Cstring, Cint, Cstring, Cint, Cstring, Ptr{Cvoid}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function CeedSetErrorHandler(ceed, eh)
    ccall((:CeedSetErrorHandler, libceed), Cint, (Ceed, Ptr{Cvoid}), ceed, eh)
end

function CeedGetErrorMessage(arg1, errmsg)
    ccall((:CeedGetErrorMessage, libceed), Cint, (Ceed, Ptr{Cstring}), arg1, errmsg)
end

function CeedResetErrorMessage(arg1, errmsg)
    ccall((:CeedResetErrorMessage, libceed), Cint, (Ceed, Ptr{Cstring}), arg1, errmsg)
end

function CeedGetPreferredMemType(ceed, type)
    ccall((:CeedGetPreferredMemType, libceed), Cint, (Ceed, Ptr{CeedMemType}), ceed, type)
end

function CeedVectorCreate(ceed, len, vec)
    ccall((:CeedVectorCreate, libceed), Cint, (Ceed, CeedInt, Ptr{CeedVector}), ceed, len, vec)
end

function CeedVectorSetArray(vec, mtype, cmode, array)
    ccall((:CeedVectorSetArray, libceed), Cint, (CeedVector, CeedMemType, CeedCopyMode, Ptr{CeedScalar}), vec, mtype, cmode, array)
end

function CeedVectorSetValue(vec, value)
    ccall((:CeedVectorSetValue, libceed), Cint, (CeedVector, CeedScalar), vec, value)
end

function CeedVectorSyncArray(vec, mtype)
    ccall((:CeedVectorSyncArray, libceed), Cint, (CeedVector, CeedMemType), vec, mtype)
end

function CeedVectorTakeArray(vec, mtype, array)
    ccall((:CeedVectorTakeArray, libceed), Cint, (CeedVector, CeedMemType, Ptr{Ptr{CeedScalar}}), vec, mtype, array)
end

function CeedVectorGetArray(vec, mtype, array)
    ccall((:CeedVectorGetArray, libceed), Cint, (CeedVector, CeedMemType, Ptr{Ptr{CeedScalar}}), vec, mtype, array)
end

function CeedVectorGetArrayRead(vec, mtype, array)
    ccall((:CeedVectorGetArrayRead, libceed), Cint, (CeedVector, CeedMemType, Ptr{Ptr{CeedScalar}}), vec, mtype, array)
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

function CeedVectorReciprocal(vec)
    ccall((:CeedVectorReciprocal, libceed), Cint, (CeedVector,), vec)
end

function CeedVectorView(vec, fpfmt, stream)
    ccall((:CeedVectorView, libceed), Cint, (CeedVector, Cstring, Ptr{FILE}), vec, fpfmt, stream)
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

function CeedElemRestrictionCreate(ceed, nelem, elemsize, ncomp, compstride, lsize, mtype, cmode, offsets, rstr)
    ccall((:CeedElemRestrictionCreate, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, CeedMemType, CeedCopyMode, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, nelem, elemsize, ncomp, compstride, lsize, mtype, cmode, offsets, rstr)
end

function CeedElemRestrictionCreateStrided(ceed, nelem, elemsize, ncomp, lsize, strides, rstr)
    ccall((:CeedElemRestrictionCreateStrided, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, nelem, elemsize, ncomp, lsize, strides, rstr)
end

function CeedElemRestrictionCreateBlocked(ceed, nelem, elemsize, blksize, ncomp, compstride, lsize, mtype, cmode, offsets, rstr)
    ccall((:CeedElemRestrictionCreateBlocked, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, CeedMemType, CeedCopyMode, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, nelem, elemsize, blksize, ncomp, compstride, lsize, mtype, cmode, offsets, rstr)
end

function CeedElemRestrictionCreateBlockedStrided(ceed, nelem, elemsize, blksize, ncomp, lsize, strides, rstr)
    ccall((:CeedElemRestrictionCreateBlockedStrided, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, Ptr{CeedInt}, Ptr{CeedElemRestriction}), ceed, nelem, elemsize, blksize, ncomp, lsize, strides, rstr)
end

function CeedElemRestrictionCreateVector(rstr, lvec, evec)
    ccall((:CeedElemRestrictionCreateVector, libceed), Cint, (CeedElemRestriction, Ptr{CeedVector}, Ptr{CeedVector}), rstr, lvec, evec)
end

function CeedElemRestrictionApply(rstr, tmode, u, ru, request)
    ccall((:CeedElemRestrictionApply, libceed), Cint, (CeedElemRestriction, CeedTransposeMode, CeedVector, CeedVector, Ptr{CeedRequest}), rstr, tmode, u, ru, request)
end

function CeedElemRestrictionApplyBlock(rstr, block, tmode, u, ru, request)
    ccall((:CeedElemRestrictionApplyBlock, libceed), Cint, (CeedElemRestriction, CeedInt, CeedTransposeMode, CeedVector, CeedVector, Ptr{CeedRequest}), rstr, block, tmode, u, ru, request)
end

function CeedElemRestrictionGetCompStride(rstr, compstride)
    ccall((:CeedElemRestrictionGetCompStride, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, compstride)
end

function CeedElemRestrictionGetNumElements(rstr, numelem)
    ccall((:CeedElemRestrictionGetNumElements, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, numelem)
end

function CeedElemRestrictionGetElementSize(rstr, elemsize)
    ccall((:CeedElemRestrictionGetElementSize, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, elemsize)
end

function CeedElemRestrictionGetLVectorSize(rstr, lsize)
    ccall((:CeedElemRestrictionGetLVectorSize, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, lsize)
end

function CeedElemRestrictionGetNumComponents(rstr, numcomp)
    ccall((:CeedElemRestrictionGetNumComponents, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, numcomp)
end

function CeedElemRestrictionGetNumBlocks(rstr, numblk)
    ccall((:CeedElemRestrictionGetNumBlocks, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, numblk)
end

function CeedElemRestrictionGetBlockSize(rstr, blksize)
    ccall((:CeedElemRestrictionGetBlockSize, libceed), Cint, (CeedElemRestriction, Ptr{CeedInt}), rstr, blksize)
end

function CeedElemRestrictionGetMultiplicity(rstr, mult)
    ccall((:CeedElemRestrictionGetMultiplicity, libceed), Cint, (CeedElemRestriction, CeedVector), rstr, mult)
end

function CeedElemRestrictionView(rstr, stream)
    ccall((:CeedElemRestrictionView, libceed), Cint, (CeedElemRestriction, Ptr{FILE}), rstr, stream)
end

function CeedElemRestrictionDestroy(rstr)
    ccall((:CeedElemRestrictionDestroy, libceed), Cint, (Ptr{CeedElemRestriction},), rstr)
end

function CeedBasisCreateTensorH1Lagrange(ceed, dim, ncomp, P, Q, qmode, basis)
    ccall((:CeedBasisCreateTensorH1Lagrange, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, CeedQuadMode, Ptr{CeedBasis}), ceed, dim, ncomp, P, Q, qmode, basis)
end

function CeedBasisCreateTensorH1(ceed, dim, ncomp, P1d, Q1d, interp1d, grad1d, qref1d, qweight1d, basis)
    ccall((:CeedBasisCreateTensorH1, libceed), Cint, (Ceed, CeedInt, CeedInt, CeedInt, CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedBasis}), ceed, dim, ncomp, P1d, Q1d, interp1d, grad1d, qref1d, qweight1d, basis)
end

function CeedBasisCreateH1(ceed, topo, ncomp, nnodes, nqpts, interp, grad, qref, qweight, basis)
    ccall((:CeedBasisCreateH1, libceed), Cint, (Ceed, CeedElemTopology, CeedInt, CeedInt, CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedBasis}), ceed, topo, ncomp, nnodes, nqpts, interp, grad, qref, qweight, basis)
end

function CeedBasisView(basis, stream)
    ccall((:CeedBasisView, libceed), Cint, (CeedBasis, Ptr{FILE}), basis, stream)
end

function CeedBasisApply(basis, nelem, tmode, emode, u, v)
    ccall((:CeedBasisApply, libceed), Cint, (CeedBasis, CeedInt, CeedTransposeMode, CeedEvalMode, CeedVector, CeedVector), basis, nelem, tmode, emode, u, v)
end

function CeedBasisGetDimension(basis, dim)
    ccall((:CeedBasisGetDimension, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, dim)
end

function CeedBasisGetTopology(basis, topo)
    ccall((:CeedBasisGetTopology, libceed), Cint, (CeedBasis, Ptr{CeedElemTopology}), basis, topo)
end

function CeedBasisGetNumComponents(basis, numcomp)
    ccall((:CeedBasisGetNumComponents, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, numcomp)
end

function CeedBasisGetNumNodes(basis, P)
    ccall((:CeedBasisGetNumNodes, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, P)
end

function CeedBasisGetNumNodes1D(basis, P1d)
    ccall((:CeedBasisGetNumNodes1D, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, P1d)
end

function CeedBasisGetNumQuadraturePoints(basis, Q)
    ccall((:CeedBasisGetNumQuadraturePoints, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, Q)
end

function CeedBasisGetNumQuadraturePoints1D(basis, Q1d)
    ccall((:CeedBasisGetNumQuadraturePoints1D, libceed), Cint, (CeedBasis, Ptr{CeedInt}), basis, Q1d)
end

function CeedBasisGetQRef(basis, qref)
    ccall((:CeedBasisGetQRef, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, qref)
end

function CeedBasisGetQWeights(basis, qweight)
    ccall((:CeedBasisGetQWeights, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, qweight)
end

function CeedBasisGetInterp(basis, interp)
    ccall((:CeedBasisGetInterp, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, interp)
end

function CeedBasisGetInterp1D(basis, interp1d)
    ccall((:CeedBasisGetInterp1D, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, interp1d)
end

function CeedBasisGetGrad(basis, grad)
    ccall((:CeedBasisGetGrad, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, grad)
end

function CeedBasisGetGrad1D(basis, grad1d)
    ccall((:CeedBasisGetGrad1D, libceed), Cint, (CeedBasis, Ptr{Ptr{CeedScalar}}), basis, grad1d)
end

function CeedBasisDestroy(basis)
    ccall((:CeedBasisDestroy, libceed), Cint, (Ptr{CeedBasis},), basis)
end

function CeedGaussQuadrature(Q, qref1d, qweight1d)
    ccall((:CeedGaussQuadrature, libceed), Cint, (CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}), Q, qref1d, qweight1d)
end

function CeedLobattoQuadrature(Q, qref1d, qweight1d)
    ccall((:CeedLobattoQuadrature, libceed), Cint, (CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}), Q, qref1d, qweight1d)
end

function CeedQRFactorization(ceed, mat, tau, m, n)
    ccall((:CeedQRFactorization, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt, CeedInt), ceed, mat, tau, m, n)
end

function CeedSymmetricSchurDecomposition(ceed, mat, lambda, n)
    ccall((:CeedSymmetricSchurDecomposition, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt), ceed, mat, lambda, n)
end

function CeedSimultaneousDiagonalization(ceed, matA, matB, x, lambda, n)
    ccall((:CeedSimultaneousDiagonalization, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt), ceed, matA, matB, x, lambda, n)
end

function CeedQFunctionCreateInterior(ceed, vlength, f, source, qf)
    ccall((:CeedQFunctionCreateInterior, libceed), Cint, (Ceed, CeedInt, CeedQFunctionUser, Cstring, Ptr{CeedQFunction}), ceed, vlength, f, source, qf)
end

function CeedQFunctionCreateInteriorByName(ceed, name, qf)
    ccall((:CeedQFunctionCreateInteriorByName, libceed), Cint, (Ceed, Cstring, Ptr{CeedQFunction}), ceed, name, qf)
end

function CeedQFunctionCreateIdentity(ceed, size, inmode, outmode, qf)
    ccall((:CeedQFunctionCreateIdentity, libceed), Cint, (Ceed, CeedInt, CeedEvalMode, CeedEvalMode, Ptr{CeedQFunction}), ceed, size, inmode, outmode, qf)
end

function CeedQFunctionAddInput(qf, fieldname, size, emode)
    ccall((:CeedQFunctionAddInput, libceed), Cint, (CeedQFunction, Cstring, CeedInt, CeedEvalMode), qf, fieldname, size, emode)
end

function CeedQFunctionAddOutput(qf, fieldname, size, emode)
    ccall((:CeedQFunctionAddOutput, libceed), Cint, (CeedQFunction, Cstring, CeedInt, CeedEvalMode), qf, fieldname, size, emode)
end

function CeedQFunctionSetContext(qf, ctx)
    ccall((:CeedQFunctionSetContext, libceed), Cint, (CeedQFunction, CeedQFunctionContext), qf, ctx)
end

function CeedQFunctionView(qf, stream)
    ccall((:CeedQFunctionView, libceed), Cint, (CeedQFunction, Ptr{FILE}), qf, stream)
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

function CeedQFunctionContextSetData(ctx, mtype, cmode, size, data)
    ccall((:CeedQFunctionContextSetData, libceed), Cint, (CeedQFunctionContext, CeedMemType, CeedCopyMode, Csize_t, Ptr{Cvoid}), ctx, mtype, cmode, size, data)
end

function CeedQFunctionContextGetData(ctx, mtype, data)
    ccall((:CeedQFunctionContextGetData, libceed), Cint, (CeedQFunctionContext, CeedMemType, Ptr{Cvoid}), ctx, mtype, data)
end

function CeedQFunctionContextRestoreData(ctx, data)
    ccall((:CeedQFunctionContextRestoreData, libceed), Cint, (CeedQFunctionContext, Ptr{Cvoid}), ctx, data)
end

function CeedQFunctionContextView(ctx, stream)
    ccall((:CeedQFunctionContextView, libceed), Cint, (CeedQFunctionContext, Ptr{FILE}), ctx, stream)
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

function CeedOperatorSetField(op, fieldname, r, b, v)
    ccall((:CeedOperatorSetField, libceed), Cint, (CeedOperator, Cstring, CeedElemRestriction, CeedBasis, CeedVector), op, fieldname, r, b, v)
end

function CeedCompositeOperatorAddSub(compositeop, subop)
    ccall((:CeedCompositeOperatorAddSub, libceed), Cint, (CeedOperator, CeedOperator), compositeop, subop)
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

function CeedOperatorMultigridLevelCreate(opFine, PMultFine, rstrCoarse, basisCoarse, opCoarse, opProlong, opRestrict)
    ccall((:CeedOperatorMultigridLevelCreate, libceed), Cint, (CeedOperator, CeedVector, CeedElemRestriction, CeedBasis, Ptr{CeedOperator}, Ptr{CeedOperator}, Ptr{CeedOperator}), opFine, PMultFine, rstrCoarse, basisCoarse, opCoarse, opProlong, opRestrict)
end

function CeedOperatorMultigridLevelCreateTensorH1(opFine, PMultFine, rstrCoarse, basisCoarse, interpCtoF, opCoarse, opProlong, opRestrict)
    ccall((:CeedOperatorMultigridLevelCreateTensorH1, libceed), Cint, (CeedOperator, CeedVector, CeedElemRestriction, CeedBasis, Ptr{CeedScalar}, Ptr{CeedOperator}, Ptr{CeedOperator}, Ptr{CeedOperator}), opFine, PMultFine, rstrCoarse, basisCoarse, interpCtoF, opCoarse, opProlong, opRestrict)
end

function CeedOperatorMultigridLevelCreateH1(opFine, PMultFine, rstrCoarse, basisCoarse, interpCtoF, opCoarse, opProlong, opRestrict)
    ccall((:CeedOperatorMultigridLevelCreateH1, libceed), Cint, (CeedOperator, CeedVector, CeedElemRestriction, CeedBasis, Ptr{CeedScalar}, Ptr{CeedOperator}, Ptr{CeedOperator}, Ptr{CeedOperator}), opFine, PMultFine, rstrCoarse, basisCoarse, interpCtoF, opCoarse, opProlong, opRestrict)
end

function CeedOperatorCreateFDMElementInverse(op, fdminv, request)
    ccall((:CeedOperatorCreateFDMElementInverse, libceed), Cint, (CeedOperator, Ptr{CeedOperator}, Ptr{CeedRequest}), op, fdminv, request)
end

function CeedOperatorView(op, stream)
    ccall((:CeedOperatorView, libceed), Cint, (CeedOperator, Ptr{FILE}), op, stream)
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
# Julia wrapper for header: ceed-cuda.h
# Automatically generated using Clang.jl


function CeedQFunctionSetCUDAUserFunction(qf, f)
    ccall((:CeedQFunctionSetCUDAUserFunction, libceed), Cint, (CeedQFunction, Cint), qf, f)
end
# Julia wrapper for header: ceed-backend.h
# Automatically generated using Clang.jl


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

function CeedRegister(prefix, init, priority)
    ccall((:CeedRegister, libceed), Cint, (Cstring, Ptr{Cvoid}, UInt32), prefix, init, priority)
end

function CeedIsDebug(ceed, isDebug)
    ccall((:CeedIsDebug, libceed), Cint, (Ceed, Ptr{Bool}), ceed, isDebug)
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

function CeedGetObjectDelegate(ceed, delegate, objname)
    ccall((:CeedGetObjectDelegate, libceed), Cint, (Ceed, Ptr{Ceed}, Cstring), ceed, delegate, objname)
end

function CeedSetObjectDelegate(ceed, delegate, objname)
    ccall((:CeedSetObjectDelegate, libceed), Cint, (Ceed, Ceed, Cstring), ceed, delegate, objname)
end

function CeedGetOperatorFallbackResource(ceed, resource)
    ccall((:CeedGetOperatorFallbackResource, libceed), Cint, (Ceed, Ptr{Cstring}), ceed, resource)
end

function CeedSetOperatorFallbackResource(ceed, resource)
    ccall((:CeedSetOperatorFallbackResource, libceed), Cint, (Ceed, Cstring), ceed, resource)
end

function CeedGetOperatorFallbackParentCeed(ceed, parent)
    ccall((:CeedGetOperatorFallbackParentCeed, libceed), Cint, (Ceed, Ptr{Ceed}), ceed, parent)
end

function CeedSetDeterministic(ceed, isDeterministic)
    ccall((:CeedSetDeterministic, libceed), Cint, (Ceed, Bool), ceed, isDeterministic)
end

function CeedSetBackendFunction(ceed, type, object, fname, f)
    ccall((:CeedSetBackendFunction, libceed), Cint, (Ceed, Cstring, Ptr{Cvoid}, Cstring, Ptr{Cvoid}), ceed, type, object, fname, f)
end

function CeedGetData(ceed, data)
    ccall((:CeedGetData, libceed), Cint, (Ceed, Ptr{Cvoid}), ceed, data)
end

function CeedSetData(ceed, data)
    ccall((:CeedSetData, libceed), Cint, (Ceed, Ptr{Cvoid}), ceed, data)
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

function CeedElemRestrictionGetCeed(rstr, ceed)
    ccall((:CeedElemRestrictionGetCeed, libceed), Cint, (CeedElemRestriction, Ptr{Ceed}), rstr, ceed)
end

function CeedElemRestrictionGetStrides(rstr, strides)
    ccall((:CeedElemRestrictionGetStrides, libceed), Cint, (CeedElemRestriction, Ptr{NTuple{3, CeedInt}}), rstr, strides)
end

function CeedElemRestrictionGetOffsets(rstr, mtype, offsets)
    ccall((:CeedElemRestrictionGetOffsets, libceed), Cint, (CeedElemRestriction, CeedMemType, Ptr{Ptr{CeedInt}}), rstr, mtype, offsets)
end

function CeedElemRestrictionRestoreOffsets(rstr, offsets)
    ccall((:CeedElemRestrictionRestoreOffsets, libceed), Cint, (CeedElemRestriction, Ptr{Ptr{CeedInt}}), rstr, offsets)
end

function CeedElemRestrictionIsStrided(rstr, isstrided)
    ccall((:CeedElemRestrictionIsStrided, libceed), Cint, (CeedElemRestriction, Ptr{Bool}), rstr, isstrided)
end

function CeedElemRestrictionHasBackendStrides(rstr, hasbackendstrides)
    ccall((:CeedElemRestrictionHasBackendStrides, libceed), Cint, (CeedElemRestriction, Ptr{Bool}), rstr, hasbackendstrides)
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

function CeedBasisGetCollocatedGrad(basis, colograd1d)
    ccall((:CeedBasisGetCollocatedGrad, libceed), Cint, (CeedBasis, Ptr{CeedScalar}), basis, colograd1d)
end

function CeedHouseholderApplyQ(A, Q, tau, tmode, m, n, k, row, col)
    ccall((:CeedHouseholderApplyQ, libceed), Cint, (Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedTransposeMode, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt), A, Q, tau, tmode, m, n, k, row, col)
end

function CeedBasisGetCeed(basis, ceed)
    ccall((:CeedBasisGetCeed, libceed), Cint, (CeedBasis, Ptr{Ceed}), basis, ceed)
end

function CeedBasisIsTensor(basis, istensor)
    ccall((:CeedBasisIsTensor, libceed), Cint, (CeedBasis, Ptr{Bool}), basis, istensor)
end

function CeedBasisGetData(basis, data)
    ccall((:CeedBasisGetData, libceed), Cint, (CeedBasis, Ptr{Cvoid}), basis, data)
end

function CeedBasisSetData(basis, data)
    ccall((:CeedBasisSetData, libceed), Cint, (CeedBasis, Ptr{Cvoid}), basis, data)
end

function CeedBasisGetTopologyDimension(topo, dim)
    ccall((:CeedBasisGetTopologyDimension, libceed), Cint, (CeedElemTopology, Ptr{CeedInt}), topo, dim)
end

function CeedBasisGetTensorContract(basis, contract)
    ccall((:CeedBasisGetTensorContract, libceed), Cint, (CeedBasis, Ptr{CeedTensorContract}), basis, contract)
end

function CeedBasisSetTensorContract(basis, contract)
    ccall((:CeedBasisSetTensorContract, libceed), Cint, (CeedBasis, Ptr{CeedTensorContract}), basis, contract)
end

function CeedTensorContractCreate(ceed, basis, contract)
    ccall((:CeedTensorContractCreate, libceed), Cint, (Ceed, CeedBasis, Ptr{CeedTensorContract}), ceed, basis, contract)
end

function CeedTensorContractApply(contract, A, B, C, J, t, tmode, Add, u, v)
    ccall((:CeedTensorContractApply, libceed), Cint, (CeedTensorContract, CeedInt, CeedInt, CeedInt, CeedInt, Ptr{CeedScalar}, CeedTransposeMode, CeedInt, Ptr{CeedScalar}, Ptr{CeedScalar}), contract, A, B, C, J, t, tmode, Add, u, v)
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

function CeedTensorContractDestroy(contract)
    ccall((:CeedTensorContractDestroy, libceed), Cint, (Ptr{CeedTensorContract},), contract)
end

function CeedQFunctionRegister(arg1, arg2, arg3, arg4, init)
    ccall((:CeedQFunctionRegister, libceed), Cint, (Cstring, Cstring, CeedInt, CeedQFunctionUser, Ptr{Cvoid}), arg1, arg2, arg3, arg4, init)
end

function CeedQFunctionSetFortranStatus(qf, status)
    ccall((:CeedQFunctionSetFortranStatus, libceed), Cint, (CeedQFunction, Bool), qf, status)
end

function CeedQFunctionGetCeed(qf, ceed)
    ccall((:CeedQFunctionGetCeed, libceed), Cint, (CeedQFunction, Ptr{Ceed}), qf, ceed)
end

function CeedQFunctionGetVectorLength(qf, vlength)
    ccall((:CeedQFunctionGetVectorLength, libceed), Cint, (CeedQFunction, Ptr{CeedInt}), qf, vlength)
end

function CeedQFunctionGetNumArgs(qf, numinputfields, numoutputfields)
    ccall((:CeedQFunctionGetNumArgs, libceed), Cint, (CeedQFunction, Ptr{CeedInt}, Ptr{CeedInt}), qf, numinputfields, numoutputfields)
end

function CeedQFunctionGetSourcePath(qf, source)
    ccall((:CeedQFunctionGetSourcePath, libceed), Cint, (CeedQFunction, Ptr{Cstring}), qf, source)
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

function CeedQFunctionIsIdentity(qf, isidentity)
    ccall((:CeedQFunctionIsIdentity, libceed), Cint, (CeedQFunction, Ptr{Bool}), qf, isidentity)
end

function CeedQFunctionGetData(qf, data)
    ccall((:CeedQFunctionGetData, libceed), Cint, (CeedQFunction, Ptr{Cvoid}), qf, data)
end

function CeedQFunctionSetData(qf, data)
    ccall((:CeedQFunctionSetData, libceed), Cint, (CeedQFunction, Ptr{Cvoid}), qf, data)
end

function CeedQFunctionGetFields(qf, inputfields, outputfields)
    ccall((:CeedQFunctionGetFields, libceed), Cint, (CeedQFunction, Ptr{Ptr{CeedQFunctionField}}, Ptr{Ptr{CeedQFunctionField}}), qf, inputfields, outputfields)
end

function CeedQFunctionFieldGetName(qffield, fieldname)
    ccall((:CeedQFunctionFieldGetName, libceed), Cint, (CeedQFunctionField, Ptr{Cstring}), qffield, fieldname)
end

function CeedQFunctionFieldGetSize(qffield, size)
    ccall((:CeedQFunctionFieldGetSize, libceed), Cint, (CeedQFunctionField, Ptr{CeedInt}), qffield, size)
end

function CeedQFunctionFieldGetEvalMode(qffield, emode)
    ccall((:CeedQFunctionFieldGetEvalMode, libceed), Cint, (CeedQFunctionField, Ptr{CeedEvalMode}), qffield, emode)
end

function CeedQFunctionContextGetCeed(cxt, ceed)
    ccall((:CeedQFunctionContextGetCeed, libceed), Cint, (CeedQFunctionContext, Ptr{Ceed}), cxt, ceed)
end

function CeedQFunctionContextGetState(ctx, state)
    ccall((:CeedQFunctionContextGetState, libceed), Cint, (CeedQFunctionContext, Ptr{UInt64}), ctx, state)
end

function CeedQFunctionContextGetContextSize(ctx, ctxsize)
    ccall((:CeedQFunctionContextGetContextSize, libceed), Cint, (CeedQFunctionContext, Ptr{Csize_t}), ctx, ctxsize)
end

function CeedQFunctionContextGetBackendData(ctx, data)
    ccall((:CeedQFunctionContextGetBackendData, libceed), Cint, (CeedQFunctionContext, Ptr{Cvoid}), ctx, data)
end

function CeedQFunctionContextSetBackendData(ctx, data)
    ccall((:CeedQFunctionContextSetBackendData, libceed), Cint, (CeedQFunctionContext, Ptr{Cvoid}), ctx, data)
end

function CeedOperatorGetCeed(op, ceed)
    ccall((:CeedOperatorGetCeed, libceed), Cint, (CeedOperator, Ptr{Ceed}), op, ceed)
end

function CeedOperatorGetNumElements(op, numelem)
    ccall((:CeedOperatorGetNumElements, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, numelem)
end

function CeedOperatorGetNumQuadraturePoints(op, numqpts)
    ccall((:CeedOperatorGetNumQuadraturePoints, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, numqpts)
end

function CeedOperatorGetNumArgs(op, numargs)
    ccall((:CeedOperatorGetNumArgs, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, numargs)
end

function CeedOperatorIsSetupDone(op, issetupdone)
    ccall((:CeedOperatorIsSetupDone, libceed), Cint, (CeedOperator, Ptr{Bool}), op, issetupdone)
end

function CeedOperatorGetQFunction(op, qf)
    ccall((:CeedOperatorGetQFunction, libceed), Cint, (CeedOperator, Ptr{CeedQFunction}), op, qf)
end

function CeedOperatorIsComposite(op, iscomposite)
    ccall((:CeedOperatorIsComposite, libceed), Cint, (CeedOperator, Ptr{Bool}), op, iscomposite)
end

function CeedOperatorGetNumSub(op, numsub)
    ccall((:CeedOperatorGetNumSub, libceed), Cint, (CeedOperator, Ptr{CeedInt}), op, numsub)
end

function CeedOperatorGetSubList(op, suboperators)
    ccall((:CeedOperatorGetSubList, libceed), Cint, (CeedOperator, Ptr{Ptr{CeedOperator}}), op, suboperators)
end

function CeedOperatorGetData(op, data)
    ccall((:CeedOperatorGetData, libceed), Cint, (CeedOperator, Ptr{Cvoid}), op, data)
end

function CeedOperatorSetData(op, data)
    ccall((:CeedOperatorSetData, libceed), Cint, (CeedOperator, Ptr{Cvoid}), op, data)
end

function CeedOperatorSetSetupDone(op)
    ccall((:CeedOperatorSetSetupDone, libceed), Cint, (CeedOperator,), op)
end

function CeedOperatorGetFields(op, inputfields, outputfields)
    ccall((:CeedOperatorGetFields, libceed), Cint, (CeedOperator, Ptr{Ptr{CeedOperatorField}}, Ptr{Ptr{CeedOperatorField}}), op, inputfields, outputfields)
end

function CeedOperatorFieldGetElemRestriction(opfield, rstr)
    ccall((:CeedOperatorFieldGetElemRestriction, libceed), Cint, (CeedOperatorField, Ptr{CeedElemRestriction}), opfield, rstr)
end

function CeedOperatorFieldGetBasis(opfield, basis)
    ccall((:CeedOperatorFieldGetBasis, libceed), Cint, (CeedOperatorField, Ptr{CeedBasis}), opfield, basis)
end

function CeedOperatorFieldGetVector(opfield, vec)
    ccall((:CeedOperatorFieldGetVector, libceed), Cint, (CeedOperatorField, Ptr{CeedVector}), opfield, vec)
end

function CeedMatrixMultiply(ceed, matA, matB, matC, m, n, kk)
    ccall((:CeedMatrixMultiply, libceed), Cint, (Ceed, Ptr{CeedScalar}, Ptr{CeedScalar}, Ptr{CeedScalar}, CeedInt, CeedInt, CeedInt), ceed, matA, matB, matC, m, n, kk)
end
