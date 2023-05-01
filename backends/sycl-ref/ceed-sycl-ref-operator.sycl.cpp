// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <cassert>
#include <string>
#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref.hpp"

class CeedOperatorSyclLinearDiagonal;
class CeedOperatorSyclLinearAssemble;
class CeedOperatorSyclLinearAssembleFallback;

//------------------------------------------------------------------------------
//  Get Basis Emode Pointer
//------------------------------------------------------------------------------
void CeedOperatorGetBasisPointer_Sycl(const CeedScalar **basisptr, CeedEvalMode emode, const CeedScalar *identity, const CeedScalar *interp,
                                      const CeedScalar *grad) {
  switch (emode) {
    case CEED_EVAL_NONE:
      *basisptr = identity;
      break;
    case CEED_EVAL_INTERP:
      *basisptr = interp;
      break;
    case CEED_EVAL_GRAD:
      *basisptr = grad;
      break;
    case CEED_EVAL_WEIGHT:
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      break;  // Caught by QF Assembly
  }
}

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Sycl(CeedOperator op) {
  CeedOperator_Sycl *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  Ceed_Sycl *sycl_data;
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  // Apply data
  for (CeedInt i = 0; i < impl->numein + impl->numeout; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->evecs[i]));
  }
  CeedCallBackend(CeedFree(&impl->evecs));

  for (CeedInt i = 0; i < impl->numein; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->qvecsin[i]));
  }
  CeedCallBackend(CeedFree(&impl->qvecsin));

  for (CeedInt i = 0; i < impl->numeout; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->qvecsout[i]));
  }
  CeedCallBackend(CeedFree(&impl->qvecsout));

  // QFunction assembly data
  for (CeedInt i = 0; i < impl->qfnumactivein; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->qfactivein[i]));
  }
  CeedCallBackend(CeedFree(&impl->qfactivein));

  // Diag data
  if (impl->diag) {
    CeedCallBackend(CeedFree(&impl->diag->h_emodein));
    CeedCallBackend(CeedFree(&impl->diag->h_emodeout));

    CeedCallSycl(ceed, sycl_data->sycl_queue.wait_and_throw());
    CeedCallSycl(ceed, sycl::free(impl->diag->d_emodein, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_emodeout, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_identity, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_interpin, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_interpout, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_gradin, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_gradout, sycl_data->sycl_context));
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->diag->pbdiagrstr));

    CeedCallBackend(CeedVectorDestroy(&impl->diag->elemdiag));
    CeedCallBackend(CeedVectorDestroy(&impl->diag->pbelemdiag));
  }
  CeedCallBackend(CeedFree(&impl->diag));

  if (impl->asmb) {
    CeedCallSycl(ceed, sycl_data->sycl_queue.wait_and_throw());
    CeedCallSycl(ceed, sycl::free(impl->asmb->d_B_in, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->asmb->d_B_out, sycl_data->sycl_context));
  }
  CeedCallBackend(CeedFree(&impl->asmb));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup infields or outfields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Sycl(CeedQFunction qf, CeedOperator op, bool isinput, CeedVector *evecs, CeedVector *qvecs, CeedInt starte,
                                        CeedInt numfields, CeedInt Q, CeedInt numelements) {
  CeedInt  dim, size;
  CeedSize q_size;
  Ceed     ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedBasis           basis;
  CeedElemRestriction Erestrict;
  CeedOperatorField  *opfields;
  CeedQFunctionField *qffields;
  CeedVector          fieldvec;
  bool                strided;
  bool                skiprestrict;

  if (isinput) {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, &opfields, NULL, NULL));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qffields, NULL, NULL));
  } else {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, NULL, NULL, &opfields));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qffields));
  }

  // Loop over fields
  for (CeedInt i = 0; i < numfields; i++) {
    CeedEvalMode emode;
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qffields[i], &emode));

    strided      = false;
    skiprestrict = false;
    if (emode != CEED_EVAL_WEIGHT) {
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(opfields[i], &Erestrict));

      // Check whether this field can skip the element restriction:
      // must be passive input, with emode NONE, and have a strided restriction with CEED_STRIDES_BACKEND.

      // First, check whether the field is input or output:
      if (isinput) {
        // Check for passive input:
        CeedCallBackend(CeedOperatorFieldGetVector(opfields[i], &fieldvec));
        if (fieldvec != CEED_VECTOR_ACTIVE) {
          // Check emode
          if (emode == CEED_EVAL_NONE) {
            // Check for strided restriction
            CeedCallBackend(CeedElemRestrictionIsStrided(Erestrict, &strided));
            if (strided) {
              // Check if vector is already in preferred backend ordering
              CeedCallBackend(CeedElemRestrictionHasBackendStrides(Erestrict, &skiprestrict));
            }
          }
        }
      }
      if (skiprestrict) {
        // We do not need an E-Vector, but will use the input field vector's data directly in the operator application
        evecs[i + starte] = NULL;
      } else {
        CeedCallBackend(CeedElemRestrictionCreateVector(Erestrict, NULL, &evecs[i + starte]));
      }
    }

    switch (emode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedQFunctionFieldGetSize(qffields[i], &size));
        q_size = (CeedSize)numelements * Q * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &qvecs[i]));
        break;
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedQFunctionFieldGetSize(qffields[i], &size));
        q_size = (CeedSize)numelements * Q * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &qvecs[i]));
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(opfields[i], &basis));
        CeedCallBackend(CeedQFunctionFieldGetSize(qffields[i], &size));
        CeedCallBackend(CeedBasisGetDimension(basis, &dim));
        q_size = (CeedSize)numelements * Q * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &qvecs[i]));
        break;
      case CEED_EVAL_WEIGHT:  // Only on input fields
        CeedCallBackend(CeedOperatorFieldGetBasis(opfields[i], &basis));
        q_size = (CeedSize)numelements * Q;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &qvecs[i]));
        CeedCallBackend(CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, NULL, qvecs[i]));
        break;
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// CeedOperator needs to connect all the named fields (be they active or
// passive) to the named inputs and outputs of its CeedQFunction.
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Sycl(CeedOperator op) {
  bool setupdone;
  CeedCallBackend(CeedOperatorIsSetupDone(op, &setupdone));
  if (setupdone) return CEED_ERROR_SUCCESS;

  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Sycl *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedQFunction qf;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedInt Q, numelements, numinputfields, numoutputfields;
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &numelements));
  CeedOperatorField *opinputfields, *opoutputfields;
  CeedCallBackend(CeedOperatorGetFields(op, &numinputfields, &opinputfields, &numoutputfields, &opoutputfields));
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qfinputfields, NULL, &qfoutputfields));

  // Allocate
  CeedCallBackend(CeedCalloc(numinputfields + numoutputfields, &impl->evecs));

  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->qvecsin));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->qvecsout));

  impl->numein  = numinputfields;
  impl->numeout = numoutputfields;

  // Set up infield and outfield evecs and qvecs
  // Infields
  CeedCallBackend(CeedOperatorSetupFields_Sycl(qf, op, true, impl->evecs, impl->qvecsin, 0, numinputfields, Q, numelements));

  // Outfields
  CeedCallBackend(CeedOperatorSetupFields_Sycl(qf, op, false, impl->evecs, impl->qvecsout, numinputfields, numoutputfields, Q, numelements));

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Sycl(CeedInt numinputfields, CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
                                               CeedVector invec, const bool skipactive, CeedScalar *edata[2 * CEED_FIELD_MAX],
                                               CeedOperator_Sycl *impl, CeedRequest *request) {
  CeedEvalMode        emode;
  CeedVector          vec;
  CeedElemRestriction Erestrict;

  for (CeedInt i = 0; i < numinputfields; i++) {
    // Get input vector
    CeedCallBackend(CeedOperatorFieldGetVector(opinputfields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      if (skipactive) continue;
      else vec = invec;
    }

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode));
    if (emode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(opinputfields[i], &vec));
      // Get input element restriction
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict));
      if (vec == CEED_VECTOR_ACTIVE) vec = invec;
      // Restrict, if necessary
      if (!impl->evecs[i]) {
        // No restriction for this field; read data directly from vec.
        CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, (const CeedScalar **)&edata[i]));
      } else {
        CeedCallBackend(CeedElemRestrictionApply(Erestrict, CEED_NOTRANSPOSE, vec, impl->evecs[i], request));
        // Get evec
        CeedCallBackend(CeedVectorGetArrayRead(impl->evecs[i], CEED_MEM_DEVICE, (const CeedScalar **)&edata[i]));
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Sycl(CeedInt numelements, CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
                                              CeedInt numinputfields, const bool skipactive, CeedScalar *edata[2 * CEED_FIELD_MAX],
                                              CeedOperator_Sycl *impl) {
  CeedInt             elemsize, size;
  CeedElemRestriction Erestrict;
  CeedEvalMode        emode;
  CeedBasis           basis;

  for (CeedInt i = 0; i < numinputfields; i++) {
    // Skip active input
    if (skipactive) {
      CeedVector vec;
      CeedCallBackend(CeedOperatorFieldGetVector(opinputfields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }
    // Get elemsize, emode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict));
    CeedCallBackend(CeedElemRestrictionGetElementSize(Erestrict, &elemsize));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qfinputfields[i], &size));
    // Basis action
    switch (emode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_DEVICE, CEED_USE_POINTER, edata[i]));
        break;
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedOperatorFieldGetBasis(opinputfields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, impl->evecs[i], impl->qvecsin[i]));
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(opinputfields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, impl->evecs[i], impl->qvecsin[i]));
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Sycl(CeedInt numinputfields, CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
                                                 const bool skipactive, CeedScalar *edata[2 * CEED_FIELD_MAX], CeedOperator_Sycl *impl) {
  CeedEvalMode emode;
  CeedVector   vec;

  for (CeedInt i = 0; i < numinputfields; i++) {
    // Skip active input
    if (skipactive) {
      CeedCallBackend(CeedOperatorFieldGetVector(opinputfields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode));
    if (emode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      if (!impl->evecs[i]) {          // This was a skiprestrict case
        CeedCallBackend(CeedOperatorFieldGetVector(opinputfields[i], &vec));
        CeedCallBackend(CeedVectorRestoreArrayRead(vec, (const CeedScalar **)&edata[i]));
      } else {
        CeedCallBackend(CeedVectorRestoreArrayRead(impl->evecs[i], (const CeedScalar **)&edata[i]));
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Sycl(CeedOperator op, CeedVector invec, CeedVector outvec, CeedRequest *request) {
  CeedOperator_Sycl *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedQFunction qf;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedInt Q, numelements, elemsize, numinputfields, numoutputfields, size;
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &numelements));
  CeedOperatorField *opinputfields, *opoutputfields;
  CeedCallBackend(CeedOperatorGetFields(op, &numinputfields, &opinputfields, &numoutputfields, &opoutputfields));
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qfinputfields, NULL, &qfoutputfields));
  CeedEvalMode        emode;
  CeedVector          vec;
  CeedBasis           basis;
  CeedElemRestriction Erestrict;
  CeedScalar         *edata[2 * CEED_FIELD_MAX] = {0};

  // Setup
  CeedCallBackend(CeedOperatorSetup_Sycl(op));

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Sycl(numinputfields, qfinputfields, opinputfields, invec, false, edata, impl, request));

  // Input basis apply if needed
  CeedCallBackend(CeedOperatorInputBasis_Sycl(numelements, qfinputfields, opinputfields, numinputfields, false, edata, impl));

  // Output pointers, as necessary
  for (CeedInt i = 0; i < numoutputfields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode));
    if (emode == CEED_EVAL_NONE) {
      // Set the output Q-Vector to use the E-Vector data directly
      CeedCallBackend(CeedVectorGetArrayWrite(impl->evecs[i + impl->numein], CEED_MEM_DEVICE, &edata[i + numinputfields]));
      CeedCallBackend(CeedVectorSetArray(impl->qvecsout[i], CEED_MEM_DEVICE, CEED_USE_POINTER, edata[i + numinputfields]));
    }
  }

  // Q function
  CeedCallBackend(CeedQFunctionApply(qf, numelements * Q, impl->qvecsin, impl->qvecsout));

  // Output basis apply if needed
  for (CeedInt i = 0; i < numoutputfields; i++) {
    // Get elemsize, emode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict));
    CeedCallBackend(CeedElemRestrictionGetElementSize(Erestrict, &elemsize));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qfoutputfields[i], &size));
    // Basis action
    switch (emode) {
      case CEED_EVAL_NONE:
        break;
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedOperatorFieldGetBasis(opoutputfields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, numelements, CEED_TRANSPOSE, CEED_EVAL_INTERP, impl->qvecsout[i], impl->evecs[i + impl->numein]));
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(opoutputfields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, numelements, CEED_TRANSPOSE, CEED_EVAL_GRAD, impl->qvecsout[i], impl->evecs[i + impl->numein]));
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT:
        Ceed ceed;
        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        break;  // Should not occur
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
                // LCOV_EXCL_STOP
    }
  }

  // Output restriction
  for (CeedInt i = 0; i < numoutputfields; i++) {
    // Restore evec
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode));
    if (emode == CEED_EVAL_NONE) {
      CeedCallBackend(CeedVectorRestoreArray(impl->evecs[i + impl->numein], &edata[i + numinputfields]));
    }
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(opoutputfields[i], &vec));
    // Restrict
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict));
    // Active
    if (vec == CEED_VECTOR_ACTIVE) vec = outvec;

    CeedCallBackend(CeedElemRestrictionApply(Erestrict, CEED_TRANSPOSE, impl->evecs[i + impl->numein], vec, request));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Sycl(numinputfields, qfinputfields, opinputfields, false, edata, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for assembling linear QFunction
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Sycl(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                               CeedRequest *request) {
  CeedOperator_Sycl *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedQFunction qf;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedInt  Q, numelements, numinputfields, numoutputfields, size;
  CeedSize q_size;
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &numelements));
  CeedOperatorField *opinputfields, *opoutputfields;
  CeedCallBackend(CeedOperatorGetFields(op, &numinputfields, &opinputfields, &numoutputfields, &opoutputfields));
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qfinputfields, NULL, &qfoutputfields));
  CeedVector  vec;
  CeedInt     numactivein = impl->qfnumactivein, numactiveout = impl->qfnumactiveout;
  CeedVector *activein = impl->qfactivein;
  CeedScalar *a, *tmp;
  Ceed        ceed, ceedparent;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetOperatorFallbackParentCeed(ceed, &ceedparent));
  ceedparent = ceedparent ? ceedparent : ceed;
  CeedScalar *edata[2 * CEED_FIELD_MAX];

  // Setup
  CeedCallBackend(CeedOperatorSetup_Sycl(op));

  // Check for identity
  bool identityqf;
  CeedCallBackend(CeedQFunctionIsIdentity(qf, &identityqf));
  if (identityqf) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Assembling identity QFunctions not supported");
    // LCOV_EXCL_STOP
  }

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Sycl(numinputfields, qfinputfields, opinputfields, NULL, true, edata, impl, request));

  // Count number of active input fields
  if (!numactivein) {
    for (CeedInt i = 0; i < numinputfields; i++) {
      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(opinputfields[i], &vec));
      // Check if active input
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qfinputfields[i], &size));
        CeedCallBackend(CeedVectorSetValue(impl->qvecsin[i], 0.0));
        CeedCallBackend(CeedVectorGetArray(impl->qvecsin[i], CEED_MEM_DEVICE, &tmp));
        CeedCallBackend(CeedRealloc(numactivein + size, &activein));
        for (CeedInt field = 0; field < size; field++) {
          q_size = (CeedSize)Q * numelements;
          CeedCallBackend(CeedVectorCreate(ceed, q_size, &activein[numactivein + field]));
          CeedCallBackend(CeedVectorSetArray(activein[numactivein + field], CEED_MEM_DEVICE, CEED_USE_POINTER, &tmp[field * Q * numelements]));
        }
        numactivein += size;
        CeedCallBackend(CeedVectorRestoreArray(impl->qvecsin[i], &tmp));
      }
    }
    impl->qfnumactivein = numactivein;
    impl->qfactivein    = activein;
  }

  // Count number of active output fields
  if (!numactiveout) {
    for (CeedInt i = 0; i < numoutputfields; i++) {
      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(opoutputfields[i], &vec));
      // Check if active output
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qfoutputfields[i], &size));
        numactiveout += size;
      }
    }
    impl->qfnumactiveout = numactiveout;
  }

  // Check sizes
  if (!numactivein || !numactiveout) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");
    // LCOV_EXCL_STOP
  }

  // Build objects if needed
  if (build_objects) {
    // Create output restriction
    CeedInt strides[3] = {1, numelements * Q, Q}; /* *NOPAD* */
    CeedCallBackend(CeedElemRestrictionCreateStrided(ceedparent, numelements, Q, numactivein * numactiveout,
                                                     numactivein * numactiveout * numelements * Q, strides, rstr));
    // Create assembled vector
    CeedSize l_size = (CeedSize)numelements * Q * numactivein * numactiveout;
    CeedCallBackend(CeedVectorCreate(ceedparent, l_size, assembled));
  }
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));
  CeedCallBackend(CeedVectorGetArray(*assembled, CEED_MEM_DEVICE, &a));

  // Input basis apply
  CeedCallBackend(CeedOperatorInputBasis_Sycl(numelements, qfinputfields, opinputfields, numinputfields, true, edata, impl));

  // Assemble QFunction
  for (CeedInt in = 0; in < numactivein; in++) {
    // Set Inputs
    CeedCallBackend(CeedVectorSetValue(activein[in], 1.0));
    if (numactivein > 1) {
      CeedCallBackend(CeedVectorSetValue(activein[(in + numactivein - 1) % numactivein], 0.0));
    }
    // Set Outputs
    for (CeedInt out = 0; out < numoutputfields; out++) {
      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(opoutputfields[out], &vec));
      // Check if active output
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedVectorSetArray(impl->qvecsout[out], CEED_MEM_DEVICE, CEED_USE_POINTER, a));
        CeedCallBackend(CeedQFunctionFieldGetSize(qfoutputfields[out], &size));
        a += size * Q * numelements;  // Advance the pointer by the size of the output
      }
    }
    // Apply QFunction
    CeedCallBackend(CeedQFunctionApply(qf, Q * numelements, impl->qvecsin, impl->qvecsout));
  }

  // Un-set output Qvecs to prevent accidental overwrite of Assembled
  for (CeedInt out = 0; out < numoutputfields; out++) {
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(opoutputfields[out], &vec));
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedVectorTakeArray(impl->qvecsout[out], CEED_MEM_DEVICE, NULL));
    }
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Sycl(numinputfields, qfinputfields, opinputfields, true, edata, impl));

  // Restore output
  CeedCallBackend(CeedVectorRestoreArray(*assembled, &a));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Sycl(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Sycl(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Sycl(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Sycl(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Create point block restriction
//------------------------------------------------------------------------------
static int CreatePBRestriction(CeedElemRestriction rstr, CeedElemRestriction *pbRstr) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  const CeedInt *offsets;
  CeedCallBackend(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));

  // Expand offsets
  CeedInt  nelem, ncomp, elemsize, compstride, *pbOffsets;
  CeedSize l_size;
  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &nelem));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &ncomp));
  CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elemsize));
  CeedCallBackend(CeedElemRestrictionGetCompStride(rstr, &compstride));
  CeedCallBackend(CeedElemRestrictionGetLVectorSize(rstr, &l_size));
  CeedInt shift = ncomp;
  if (compstride != 1) shift *= ncomp;
  CeedCallBackend(CeedCalloc(nelem * elemsize, &pbOffsets));
  for (CeedInt i = 0; i < nelem * elemsize; i++) {
    pbOffsets[i] = offsets[i] * shift;
  }

  // Create new restriction
  CeedCallBackend(
      CeedElemRestrictionCreate(ceed, nelem, elemsize, ncomp * ncomp, 1, l_size * ncomp, CEED_MEM_HOST, CEED_OWN_POINTER, pbOffsets, pbRstr));

  // Cleanup
  CeedCallBackend(CeedElemRestrictionRestoreOffsets(rstr, &offsets));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble diagonal setup
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalSetup_Sycl(CeedOperator op, const bool pointBlock) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedQFunction qf;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedInt numinputfields, numoutputfields;
  CeedCallBackend(CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields));

  // Determine active input basis
  CeedOperatorField  *opfields;
  CeedQFunctionField *qffields;
  CeedCallBackend(CeedOperatorGetFields(op, NULL, &opfields, NULL, NULL));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qffields, NULL, NULL));
  CeedInt             numemodein = 0, ncomp = 0, dim = 1;
  CeedEvalMode       *emodein = NULL;
  CeedBasis           basisin = NULL;
  CeedElemRestriction rstrin  = NULL;
  for (CeedInt i = 0; i < numinputfields; i++) {
    CeedVector vec;
    CeedCallBackend(CeedOperatorFieldGetVector(opfields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedElemRestriction rstr;
      CeedCallBackend(CeedOperatorFieldGetBasis(opfields[i], &basisin));
      CeedCallBackend(CeedBasisGetNumComponents(basisin, &ncomp));
      CeedCallBackend(CeedBasisGetDimension(basisin, &dim));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(opfields[i], &rstr));
      if (rstrin && rstrin != rstr) {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement multi-field non-composite operator diagonal assembly");
        // LCOV_EXCL_STOP
      }
      rstrin = rstr;
      CeedEvalMode emode;
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qffields[i], &emode));
      switch (emode) {
        case CEED_EVAL_NONE:
        case CEED_EVAL_INTERP:
          CeedCallBackend(CeedRealloc(numemodein + 1, &emodein));
          emodein[numemodein] = emode;
          numemodein += 1;
          break;
        case CEED_EVAL_GRAD:
          CeedCallBackend(CeedRealloc(numemodein + dim, &emodein));
          for (CeedInt d = 0; d < dim; d++) emodein[numemodein + d] = emode;
          numemodein += dim;
          break;
        case CEED_EVAL_WEIGHT:
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // Caught by QF Assembly
      }
    }
  }

  // Determine active output basis
  CeedCallBackend(CeedOperatorGetFields(op, NULL, NULL, NULL, &opfields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qffields));
  CeedInt             numemodeout = 0;
  CeedEvalMode       *emodeout    = NULL;
  CeedBasis           basisout    = NULL;
  CeedElemRestriction rstrout     = NULL;
  for (CeedInt i = 0; i < numoutputfields; i++) {
    CeedVector vec;
    CeedCallBackend(CeedOperatorFieldGetVector(opfields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedElemRestriction rstr;
      CeedCallBackend(CeedOperatorFieldGetBasis(opfields[i], &basisout));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(opfields[i], &rstr));
      if (rstrout && rstrout != rstr) {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement multi-field non-composite operator diagonal assembly");
        // LCOV_EXCL_STOP
      }
      rstrout = rstr;
      CeedEvalMode emode;
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qffields[i], &emode));
      switch (emode) {
        case CEED_EVAL_NONE:
        case CEED_EVAL_INTERP:
          CeedCallBackend(CeedRealloc(numemodeout + 1, &emodeout));
          emodeout[numemodeout] = emode;
          numemodeout += 1;
          break;
        case CEED_EVAL_GRAD:
          CeedCallBackend(CeedRealloc(numemodeout + dim, &emodeout));
          for (CeedInt d = 0; d < dim; d++) emodeout[numemodeout + d] = emode;
          numemodeout += dim;
          break;
        case CEED_EVAL_WEIGHT:
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // Caught by QF Assembly
      }
    }
  }

  // Operator data struct
  CeedOperator_Sycl *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  Ceed_Sycl *sycl_data;
  CeedCallBackend(CeedGetData(ceed, &sycl_data));
  CeedCallBackend(CeedCalloc(1, &impl->diag));
  CeedOperatorDiag_Sycl *diag = impl->diag;
  diag->basisin               = basisin;
  diag->basisout              = basisout;
  diag->h_emodein             = emodein;
  diag->h_emodeout            = emodeout;
  diag->numemodein            = numemodein;
  diag->numemodeout           = numemodeout;

  // Kernel parameters
  CeedInt nnodes, nqpts;
  CeedCallBackend(CeedBasisGetNumNodes(basisin, &nnodes));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basisin, &nqpts));
  diag->nnodes = nnodes;
  diag->nqpts  = nqpts;
  diag->ncomp  = ncomp;

  // Basis matrices
  const CeedInt     iLen = nqpts * nnodes;
  const CeedInt     gLen = nqpts * nnodes * dim;
  const CeedScalar *interpin, *interpout, *gradin, *gradout;

  // CEED_EVAL_NONE
  CeedScalar *identity = NULL;
  bool        evalNone = false;
  for (CeedInt i = 0; i < numemodein; i++) evalNone = evalNone || (emodein[i] == CEED_EVAL_NONE);
  for (CeedInt i = 0; i < numemodeout; i++) evalNone = evalNone || (emodeout[i] == CEED_EVAL_NONE);

  std::vector<sycl::event> copy_events;
  if (evalNone) {
    CeedCallBackend(CeedCalloc(nqpts * nnodes, &identity));
    for (CeedInt i = 0; i < (nnodes < nqpts ? nnodes : nqpts); i++) identity[i * nnodes + i] = 1.0;
    CeedCallSycl(ceed, diag->d_identity = sycl::malloc_device<CeedScalar>(iLen, sycl_data->sycl_device, sycl_data->sycl_context));
    sycl::event identity_copy = sycl_data->sycl_queue.copy<CeedScalar>(identity, diag->d_identity, iLen);
    copy_events.push_back(identity_copy);
  }

  // CEED_EVAL_INTERP
  CeedCallBackend(CeedBasisGetInterp(basisin, &interpin));
  CeedCallSycl(ceed, diag->d_interpin = sycl::malloc_device<CeedScalar>(iLen, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event interpin_copy = sycl_data->sycl_queue.copy<CeedScalar>(interpin, diag->d_interpin, iLen);
  copy_events.push_back(interpin_copy);

  CeedCallBackend(CeedBasisGetInterp(basisout, &interpout));
  CeedCallSycl(ceed, diag->d_interpout = sycl::malloc_device<CeedScalar>(iLen, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event interpout_copy = sycl_data->sycl_queue.copy<CeedScalar>(interpout, diag->d_interpout, iLen);
  copy_events.push_back(interpout_copy);

  // CEED_EVAL_GRAD
  CeedCallBackend(CeedBasisGetGrad(basisin, &gradin));
  CeedCallSycl(ceed, diag->d_gradin = sycl::malloc_device<CeedScalar>(gLen, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event gradin_copy = sycl_data->sycl_queue.copy<CeedScalar>(gradin, diag->d_gradin, gLen);
  copy_events.push_back(gradin_copy);

  CeedCallBackend(CeedBasisGetGrad(basisout, &gradout));
  CeedCallSycl(ceed, diag->d_gradout = sycl::malloc_device<CeedScalar>(gLen, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event gradout_copy = sycl_data->sycl_queue.copy<CeedScalar>(gradout, diag->d_gradout, gLen);
  copy_events.push_back(gradout_copy);

  // Arrays of emodes
  CeedCallSycl(ceed, diag->d_emodein = sycl::malloc_device<CeedEvalMode>(numemodein, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event emodein_copy = sycl_data->sycl_queue.copy<CeedEvalMode>(emodein, diag->d_emodein, numemodein);
  copy_events.push_back(emodein_copy);

  CeedCallSycl(ceed, diag->d_emodeout = sycl::malloc_device<CeedEvalMode>(numemodeout, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event emodeout_copy = sycl_data->sycl_queue.copy<CeedEvalMode>(emodeout, diag->d_emodeout, numemodeout);
  copy_events.push_back(emodeout_copy);

  // Restriction
  diag->diagrstr = rstrout;

  // Wait for all copies to complete and handle exceptions
  CeedCallSycl(ceed, sycl::event::wait_and_throw(copy_events));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
//  Kernel for diagonal assembly
//------------------------------------------------------------------------------
static int CeedOperatorLinearDiagonal_Sycl(sycl::queue &sycl_queue, const bool pointBlock, const CeedInt nelem, const CeedOperatorDiag_Sycl *diag,
                                           const CeedScalar *assembledqfarray, CeedScalar *elemdiagarray) {
  const CeedInt nnodes      = diag->nnodes;
  const CeedInt nqpts       = diag->nqpts;
  const CeedInt ncomp       = diag->ncomp;
  const CeedInt numemodein  = diag->numemodein;
  const CeedInt numemodeout = diag->numemodeout;

  const CeedScalar   *identity  = diag->d_identity;
  const CeedScalar   *interpin  = diag->d_interpin;
  const CeedScalar   *gradin    = diag->d_gradin;
  const CeedScalar   *interpout = diag->d_interpout;
  const CeedScalar   *gradout   = diag->d_gradout;
  const CeedEvalMode *emodein   = diag->d_emodein;
  const CeedEvalMode *emodeout  = diag->d_emodeout;

  sycl::range<1> kernel_range(nelem * nnodes);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedOperatorSyclLinearDiagonal>(kernel_range, {e}, [=](sycl::id<1> idx) {
    const CeedInt tid = idx % nnodes;
    const CeedInt e   = idx / nnodes;

    // Compute the diagonal of B^T D B
    // Each element
    CeedInt dout = -1;
    // Each basis eval mode pair
    for (CeedInt eout = 0; eout < numemodeout; eout++) {
      const CeedScalar *bt = NULL;
      if (emodeout[eout] == CEED_EVAL_GRAD) ++dout;
      CeedOperatorGetBasisPointer_Sycl(&bt, emodeout[eout], identity, interpout, &gradout[dout * nqpts * nnodes]);
      CeedInt din = -1;
      for (CeedInt ein = 0; ein < numemodein; ein++) {
        const CeedScalar *b = NULL;
        if (emodein[ein] == CEED_EVAL_GRAD) ++din;
        CeedOperatorGetBasisPointer_Sycl(&b, emodein[ein], identity, interpin, &gradin[din * nqpts * nnodes]);
        // Each component
        for (CeedInt compOut = 0; compOut < ncomp; compOut++) {
          // Each qpoint/node pair
          if (pointBlock) {
            // Point Block Diagonal
            for (CeedInt compIn = 0; compIn < ncomp; compIn++) {
              CeedScalar evalue = 0.0;
              for (CeedInt q = 0; q < nqpts; q++) {
                const CeedScalar qfvalue =
                    assembledqfarray[((((ein * ncomp + compIn) * numemodeout + eout) * ncomp + compOut) * nelem + e) * nqpts + q];
                evalue += bt[q * nnodes + tid] * qfvalue * b[q * nnodes + tid];
              }
              elemdiagarray[((compOut * ncomp + compIn) * nelem + e) * nnodes + tid] += evalue;
            }
          } else {
            // Diagonal Only
            CeedScalar evalue = 0.0;
            for (CeedInt q = 0; q < nqpts; q++) {
              const CeedScalar qfvalue =
                  assembledqfarray[((((ein * ncomp + compOut) * numemodeout + eout) * ncomp + compOut) * nelem + e) * nqpts + q];
              evalue += bt[q * nnodes + tid] * qfvalue * b[q * nnodes + tid];
            }
            elemdiagarray[(compOut * nelem + e) * nnodes + tid] += evalue;
          }
        }
      }
    }
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalCore_Sycl(CeedOperator op, CeedVector assembled, CeedRequest *request, const bool pointBlock) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Sycl *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  Ceed_Sycl *sycl_data;
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  // Assemble QFunction
  CeedVector          assembledqf;
  CeedElemRestriction rstr;
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembledqf, &rstr, request));
  CeedCallBackend(CeedElemRestrictionDestroy(&rstr));

  // Setup
  if (!impl->diag) {
    CeedCallBackend(CeedOperatorAssembleDiagonalSetup_Sycl(op, pointBlock));
  }
  CeedOperatorDiag_Sycl *diag = impl->diag;
  assert(diag != NULL);

  // Restriction
  if (pointBlock && !diag->pbdiagrstr) {
    CeedElemRestriction pbdiagrstr;
    CeedCallBackend(CreatePBRestriction(diag->diagrstr, &pbdiagrstr));
    diag->pbdiagrstr = pbdiagrstr;
  }
  CeedElemRestriction diagrstr = pointBlock ? diag->pbdiagrstr : diag->diagrstr;

  // Create diagonal vector
  CeedVector elemdiag = pointBlock ? diag->pbelemdiag : diag->elemdiag;
  if (!elemdiag) {
    CeedCallBackend(CeedElemRestrictionCreateVector(diagrstr, NULL, &elemdiag));
    if (pointBlock) diag->pbelemdiag = elemdiag;
    else diag->elemdiag = elemdiag;
  }
  CeedCallBackend(CeedVectorSetValue(elemdiag, 0.0));

  // Assemble element operator diagonals
  CeedScalar       *elemdiagarray;
  const CeedScalar *assembledqfarray;
  CeedCallBackend(CeedVectorGetArray(elemdiag, CEED_MEM_DEVICE, &elemdiagarray));
  CeedCallBackend(CeedVectorGetArrayRead(assembledqf, CEED_MEM_DEVICE, &assembledqfarray));
  CeedInt nelem;
  CeedCallBackend(CeedElemRestrictionGetNumElements(diagrstr, &nelem));

  // Compute the diagonal of B^T D B
  // Umesh: This needs to be reviewed later
  // if (pointBlock) {
  //  CeedCallBackend(CeedOperatorLinearPointBlockDiagonal_Sycl(sycl_data->sycl_queue, nelem, diag, assembledqfarray, elemdiagarray));
  //} else {
  CeedCallBackend(CeedOperatorLinearDiagonal_Sycl(sycl_data->sycl_queue, pointBlock, nelem, diag, assembledqfarray, elemdiagarray));
  // }

  // Wait for queue to complete and handle exceptions
  sycl_data->sycl_queue.wait_and_throw();

  // Restore arrays
  CeedCallBackend(CeedVectorRestoreArray(elemdiag, &elemdiagarray));
  CeedCallBackend(CeedVectorRestoreArrayRead(assembledqf, &assembledqfarray));

  // Assemble local operator diagonal
  CeedCallBackend(CeedElemRestrictionApply(diagrstr, CEED_TRANSPOSE, elemdiag, assembled, request));

  // Cleanup
  CeedCallBackend(CeedVectorDestroy(&assembledqf));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonal_Sycl(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Sycl(op, assembled, request, false));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Sycl(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Sycl(op, assembled, request, true));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Single operator assembly setup
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssembleSetup_Sycl(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Sycl *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Get input and output fields
  CeedInt            num_input_fields, num_output_fields;
  CeedOperatorField *input_fields;
  CeedOperatorField *output_fields;
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &input_fields, &num_output_fields, &output_fields));

  // Determine active input basis eval mode
  CeedQFunction qf;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedQFunctionField *qf_fields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  // Note that the kernel will treat each dimension of a gradient action separately;
  // i.e., when an active input has a CEED_EVAL_GRAD mode, num_emode_in will increment by dim.
  // However, for the purposes of loading the B matrices, it will be treated as one mode, and we will load/copy the entire gradient matrix at once, so
  // num_B_in_mats_to_load will be incremented by 1.
  CeedInt             num_emode_in = 0, dim = 1, num_B_in_mats_to_load = 0, size_B_in = 0;
  CeedEvalMode       *eval_mode_in = NULL;  // will be of size num_B_in_mats_load
  CeedBasis           basis_in     = NULL;
  CeedInt             nqpts = 0, esize = 0;
  CeedElemRestriction rstr_in = NULL;
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;
    CeedCallBackend(CeedOperatorFieldGetVector(input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedOperatorFieldGetBasis(input_fields[i], &basis_in));
      CeedCallBackend(CeedBasisGetDimension(basis_in, &dim));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_in, &nqpts));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(input_fields[i], &rstr_in));
      CeedCallBackend(CeedElemRestrictionGetElementSize(rstr_in, &esize));
      CeedEvalMode eval_mode;
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      if (eval_mode != CEED_EVAL_NONE) {
        CeedCallBackend(CeedRealloc(num_B_in_mats_to_load + 1, &eval_mode_in));
        eval_mode_in[num_B_in_mats_to_load] = eval_mode;
        num_B_in_mats_to_load += 1;
        if (eval_mode == CEED_EVAL_GRAD) {
          num_emode_in += dim;
          size_B_in += dim * esize * nqpts;
        } else {
          num_emode_in += 1;
          size_B_in += esize * nqpts;
        }
      }
    }
  }

  // Determine active output basis; basis_out and rstr_out only used if same as input, TODO
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  CeedInt             num_emode_out = 0, num_B_out_mats_to_load = 0, size_B_out = 0;
  CeedEvalMode       *eval_mode_out = NULL;
  CeedBasis           basis_out     = NULL;
  CeedElemRestriction rstr_out      = NULL;
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedVector vec;
    CeedCallBackend(CeedOperatorFieldGetVector(output_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedOperatorFieldGetBasis(output_fields[i], &basis_out));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(output_fields[i], &rstr_out));
      if (rstr_out && rstr_out != rstr_in) {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement multi-field non-composite operator assembly");
        // LCOV_EXCL_STOP
      }
      CeedEvalMode eval_mode;
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      if (eval_mode != CEED_EVAL_NONE) {
        CeedCallBackend(CeedRealloc(num_B_out_mats_to_load + 1, &eval_mode_out));
        eval_mode_out[num_B_out_mats_to_load] = eval_mode;
        num_B_out_mats_to_load += 1;
        if (eval_mode == CEED_EVAL_GRAD) {
          num_emode_out += dim;
          size_B_out += dim * esize * nqpts;
        } else {
          num_emode_out += 1;
          size_B_out += esize * nqpts;
        }
      }
    }
  }

  if (num_emode_in == 0 || num_emode_out == 0) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Cannot assemble operator without inputs/outputs");
    // LCOV_EXCL_STOP
  }

  CeedInt nelem, ncomp;
  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr_in, &nelem));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr_in, &ncomp));

  CeedCallBackend(CeedCalloc(1, &impl->asmb));
  CeedOperatorAssemble_Sycl *asmb = impl->asmb;
  asmb->nelem                     = nelem;

  Ceed_Sycl *sycl_data;
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  // Kernel setup
  int elemsPerBlock   = 1;
  asmb->elemsPerBlock = elemsPerBlock;
  CeedInt block_size  = esize * esize * elemsPerBlock;
  /* CeedInt maxThreadsPerBlock = sycl_data->sycl_device.get_info<sycl::info::device::max_work_group_size>();
  bool    fallback           = block_size > maxThreadsPerBlock;
  asmb->fallback             = fallback;
  if (fallback) {
    // Use fallback kernel with 1D threadblock
    block_size         = esize * elemsPerBlock;
    asmb->block_size_x = esize;
    asmb->block_size_y = 1;
  } else {  // Use kernel with 2D threadblock
    asmb->block_size_x = esize;
    asmb->block_size_y = esize;
  }*/
  asmb->block_size_x = esize;
  asmb->block_size_y = esize;
  asmb->numemodein   = num_emode_in;
  asmb->numemodeout  = num_emode_out;
  asmb->nqpts        = nqpts;
  asmb->nnodes       = esize;
  asmb->block_size   = block_size;
  asmb->ncomp        = ncomp;

  // Build 'full' B matrices (not 1D arrays used for tensor-product matrices
  const CeedScalar *interp_in, *grad_in;
  CeedCallBackend(CeedBasisGetInterp(basis_in, &interp_in));
  CeedCallBackend(CeedBasisGetGrad(basis_in, &grad_in));

  // Load into B_in, in order that they will be used in eval_mode
  CeedInt mat_start = 0;
  CeedCallSycl(ceed, asmb->d_B_in = sycl::malloc_device<CeedScalar>(size_B_in, sycl_data->sycl_device, sycl_data->sycl_context));
  for (int i = 0; i < num_B_in_mats_to_load; i++) {
    CeedEvalMode eval_mode = eval_mode_in[i];
    if (eval_mode == CEED_EVAL_INTERP) {
      // Order queue
      sycl::event e = sycl_data->sycl_queue.ext_oneapi_submit_barrier();
      sycl_data->sycl_queue.copy<CeedScalar>(interp_in, &asmb->d_B_in[mat_start], esize * nqpts, {e});
      mat_start += esize * nqpts;
    } else if (eval_mode == CEED_EVAL_GRAD) {
      // Order queue
      sycl::event e = sycl_data->sycl_queue.ext_oneapi_submit_barrier();
      sycl_data->sycl_queue.copy<CeedScalar>(grad_in, &asmb->d_B_in[mat_start], dim * esize * nqpts, {e});
      mat_start += dim * esize * nqpts;
    }
  }

  const CeedScalar *interp_out, *grad_out;
  // Note that this function currently assumes 1 basis, so this should always be true
  // for now
  if (basis_out == basis_in) {
    interp_out = interp_in;
    grad_out   = grad_in;
  } else {
    CeedCallBackend(CeedBasisGetInterp(basis_out, &interp_out));
    CeedCallBackend(CeedBasisGetGrad(basis_out, &grad_out));
  }

  // Load into B_out, in order that they will be used in eval_mode
  mat_start = 0;
  CeedCallSycl(ceed, asmb->d_B_out = sycl::malloc_device<CeedScalar>(size_B_out, sycl_data->sycl_device, sycl_data->sycl_context));
  for (int i = 0; i < num_B_out_mats_to_load; i++) {
    CeedEvalMode eval_mode = eval_mode_out[i];
    if (eval_mode == CEED_EVAL_INTERP) {
      // Order queue
      sycl::event e = sycl_data->sycl_queue.ext_oneapi_submit_barrier();
      sycl_data->sycl_queue.copy<CeedScalar>(interp_out, &asmb->d_B_out[mat_start], esize * nqpts, {e});
      mat_start += esize * nqpts;
    } else if (eval_mode == CEED_EVAL_GRAD) {
      // Order queue
      sycl::event e = sycl_data->sycl_queue.ext_oneapi_submit_barrier();
      sycl_data->sycl_queue.copy<CeedScalar>(grad_out, &asmb->d_B_out[mat_start], dim * esize * nqpts, {e});
      mat_start += dim * esize * nqpts;
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Matrix assembly kernel for low-order elements (3D thread block)
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssemble_Sycl(sycl::queue &sycl_queue, const CeedOperator_Sycl *impl, const CeedScalar *qf_array,
                                           CeedScalar *values_array) {
  // This kernels assumes B_in and B_out have the same number of quadrature points and basis points.
  // TODO: expand to more general cases
  CeedOperatorAssemble_Sycl *asmb        = impl->asmb;
  const CeedInt              nelem       = asmb->nelem;
  const CeedInt              nnodes      = asmb->nnodes;
  const CeedInt              ncomp       = asmb->ncomp;
  const CeedInt              nqpts       = asmb->nqpts;
  const CeedInt              numemodein  = asmb->numemodein;
  const CeedInt              numemodeout = asmb->numemodeout;

  // Strides for final output ordering, determined by the reference (inference) implementation of the symbolic assembly, slowest --> fastest: element,
  // comp_in, comp_out, node_row, node_col
  const CeedInt comp_out_stride = nnodes * nnodes;
  const CeedInt comp_in_stride  = comp_out_stride * ncomp;
  const CeedInt e_stride        = comp_in_stride * ncomp;
  // Strides for QF array, slowest --> fastest: emode_in, comp_in, emode_out, comp_out, elem, qpt
  const CeedInt qe_stride         = nqpts;
  const CeedInt qcomp_out_stride  = nelem * qe_stride;
  const CeedInt qemode_out_stride = qcomp_out_stride * ncomp;
  const CeedInt qcomp_in_stride   = qemode_out_stride * numemodeout;
  const CeedInt qemode_in_stride  = qcomp_in_stride * ncomp;

  CeedScalar *B_in, *B_out;
  B_in                       = asmb->d_B_in;
  B_out                      = asmb->d_B_out;
  const CeedInt block_size_x = asmb->block_size_x;
  const CeedInt block_size_y = asmb->block_size_y;

  sycl::range<3> kernel_range(nelem, block_size_y, block_size_x);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedOperatorSyclLinearAssemble>(kernel_range, {e}, [=](sycl::id<3> idx) {
    const int e = idx.get(0);  // Element index
    const int l = idx.get(1);  // The output column index of each B^TDB operation
    const int i = idx.get(2);  // The output row index of each B^TDB operation
                               // such that we have (Bout^T)_ij D_jk Bin_kl = C_il
    for (CeedInt comp_in = 0; comp_in < ncomp; comp_in++) {
      for (CeedInt comp_out = 0; comp_out < ncomp; comp_out++) {
        CeedScalar result        = 0.0;
        CeedInt    qf_index_comp = qcomp_in_stride * comp_in + qcomp_out_stride * comp_out + qe_stride * e;
        for (CeedInt emode_in = 0; emode_in < numemodein; emode_in++) {
          CeedInt b_in_index = emode_in * nqpts * nnodes;
          for (CeedInt emode_out = 0; emode_out < numemodeout; emode_out++) {
            CeedInt b_out_index = emode_out * nqpts * nnodes;
            CeedInt qf_index    = qf_index_comp + qemode_out_stride * emode_out + qemode_in_stride * emode_in;
            // Perform the B^T D B operation for this 'chunk' of D (the qf_array)
            for (CeedInt j = 0; j < nqpts; j++) {
              result += B_out[b_out_index + j * nnodes + i] * qf_array[qf_index + j] * B_in[b_in_index + j * nnodes + l];
            }
          }  // end of emode_out
        }    // end of emode_in
        CeedInt val_index       = comp_in_stride * comp_in + comp_out_stride * comp_out + e_stride * e + nnodes * i + l;
        values_array[val_index] = result;
      }  // end of out component
    }    // end of in component
  });

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Fallback kernel for larger orders (1D thread block)
//------------------------------------------------------------------------------
/*
static int CeedOperatorLinearAssembleFallback_Sycl(sycl::queue &sycl_queue, const CeedOperator_Sycl *impl, const CeedScalar *qf_array,
                                                   CeedScalar *values_array) {
  // This kernel assumes B_in and B_out have the same number of quadrature points and basis points.
  // TODO: expand to more general cases
  CeedOperatorAssemble_Sycl *asmb        = impl->asmb;
  const CeedInt              nelem       = asmb->nelem;
  const CeedInt              nnodes      = asmb->nnodes;
  const CeedInt              ncomp       = asmb->ncomp;
  const CeedInt              nqpts       = asmb->nqpts;
  const CeedInt              numemodein  = asmb->numemodein;
  const CeedInt              numemodeout = asmb->numemodeout;

  // Strides for final output ordering, determined by the reference (interface) implementation of the symbolic assembly, slowest --> fastest: elememt,
  // comp_in, comp_out, node_row, node_col
  const CeedInt comp_out_stride = nnodes * nnodes;
  const CeedInt comp_in_stride  = comp_out_stride * ncomp;
  const CeedInt e_stride        = comp_in_stride * ncomp;
  // Strides for QF array, slowest --> fastest: emode_in, comp_in, emode_out, comp_out, elem, qpt
  const CeedInt qe_stride         = nqpts;
  const CeedInt qcomp_out_stride  = nelem * qe_stride;
  const CeedInt qemode_out_stride = qcomp_out_stride * ncomp;
  const CeedInt qcomp_in_stride   = qemode_out_stride * numemodeout;
  const CeedInt qemode_in_stride  = qcomp_in_stride * ncomp;

  CeedScalar *B_in, *B_out;
  B_in                        = asmb->d_B_in;
  B_out                       = asmb->d_B_out;
  const CeedInt elemsPerBlock = asmb->elemsPerBlock;
  const CeedInt block_size_x  = asmb->block_size_x;
  const CeedInt block_size_y  = asmb->block_size_y;  // This will be 1 for the fallback kernel

  const CeedInt     grid = nelem / elemsPerBlock + ((nelem / elemsPerBlock * elemsPerBlock < nelem) ? 1 : 0);
  sycl::range<3>    local_range(block_size_x, block_size_y, elemsPerBlock);
  sycl::range<3>    global_range(grid * block_size_x, block_size_y, elemsPerBlock);
  sycl::nd_range<3> kernel_range(global_range, local_range);

  sycl_queue.parallel_for<CeedOperatorSyclLinearAssembleFallback>(kernel_range, [=](sycl::nd_item<3> work_item) {
    const CeedInt blockIdx  = work_item.get_group(0);
    const CeedInt gridDimx  = work_item.get_group_range(0);
    const CeedInt threadIdx = work_item.get_local_id(0);
    const CeedInt threadIdz = work_item.get_local_id(2);
    const CeedInt blockDimz = work_item.get_local_range(2);

    const int l = threadIdx;  // The output column index of each B^TDB operation
                              // such that we have (Bout^T)_ij D_jk Bin_kl = C_il
    for (CeedInt e = blockIdx * blockDimz + threadIdz; e < nelem; e += gridDimx * blockDimz) {
      for (CeedInt comp_in = 0; comp_in < ncomp; comp_in++) {
        for (CeedInt comp_out = 0; comp_out < ncomp; comp_out++) {
          for (CeedInt i = 0; i < nnodes; i++) {
            CeedScalar result        = 0.0;
            CeedInt    qf_index_comp = qcomp_in_stride * comp_in + qcomp_out_stride * comp_out + qe_stride * e;
            for (CeedInt emode_in = 0; emode_in < numemodein; emode_in++) {
              CeedInt b_in_index = emode_in * nqpts * nnodes;
              for (CeedInt emode_out = 0; emode_out < numemodeout; emode_out++) {
                CeedInt b_out_index = emode_out * nqpts * nnodes;
                CeedInt qf_index    = qf_index_comp + qemode_out_stride * emode_out + qemode_in_stride * emode_in;
                // Perform the B^T D B operation for this 'chunk' of D (the qf_array)
                for (CeedInt j = 0; j < nqpts; j++) {
                  result += B_out[b_out_index + j * nnodes + i] * qf_array[qf_index + j] * B_in[b_in_index + j * nnodes + l];
                }
              }  // end of emode_out
            }    // end of emode_in
            CeedInt val_index       = comp_in_stride * comp_in + comp_out_stride * comp_out + e_stride * e + nnodes * i + l;
            values_array[val_index] = result;
          }  // end of loop over element node index, i
        }    // end of out component
      }      // end of in component
    }        // end of element loop
  });
  return CEED_ERROR_SUCCESS;
}*/

//------------------------------------------------------------------------------
// Assemble matrix data for COO matrix of assembled operator.
// The sparsity pattern is set by CeedOperatorLinearAssembleSymbolic.
//
// Note that this (and other assembly routines) currently assume only one active
// input restriction/basis per operator (could have multiple basis eval modes).
// TODO: allow multiple active input restrictions/basis objects
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssemble_Sycl(CeedOperator op, CeedInt offset, CeedVector values) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Sycl *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  Ceed_Sycl *sycl_data;
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  // Setup
  if (!impl->asmb) {
    CeedCallBackend(CeedSingleOperatorAssembleSetup_Sycl(op));
    assert(impl->asmb != NULL);
  }

  // Assemble QFunction
  CeedVector          assembled_qf;
  CeedElemRestriction rstr_q;
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &rstr_q, CEED_REQUEST_IMMEDIATE));
  CeedCallBackend(CeedElemRestrictionDestroy(&rstr_q));
  CeedScalar *values_array;
  CeedCallBackend(CeedVectorGetArrayWrite(values, CEED_MEM_DEVICE, &values_array));
  values_array += offset;
  const CeedScalar *qf_array;
  CeedCallBackend(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_DEVICE, &qf_array));

  // Compute B^T D B
  CeedCallBackend(CeedOperatorLinearAssemble_Sycl(sycl_data->sycl_queue, impl, qf_array, values_array));

  // Wait for kernels to be completed
  // Kris: Review if this is necessary -- enqueing an async barrier may be sufficient
  sycl_data->sycl_queue.wait_and_throw();

  // Restore arrays
  CeedCallBackend(CeedVectorRestoreArray(values, &values_array));
  CeedCallBackend(CeedVectorRestoreArrayRead(assembled_qf, &qf_array));

  // Cleanup
  CeedCallBackend(CeedVectorDestroy(&assembled_qf));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Sycl(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Sycl *impl;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonal_Sycl));
  CeedCallBackend(
      CeedSetBackendFunctionCpp(ceed, "Operator", op, "LinearAssembleAddPointBlockDiagonal", CeedOperatorLinearAssembleAddPointBlockDiagonal_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "LinearAssembleSingle", CeedSingleOperatorAssemble_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Sycl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
