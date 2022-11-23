// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <assert.h>
#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>
#include <hip/hip_runtime.h>
#include <stdbool.h>
#include <string.h>

#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Hip(CeedOperator op) {
  CeedOperator_Hip *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));

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

  // QFunction diagonal assembly data
  for (CeedInt i = 0; i < impl->qfnumactivein; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->qfactivein[i]));
  }
  CeedCallBackend(CeedFree(&impl->qfactivein));

  // Diag data
  if (impl->diag) {
    Ceed ceed;
    CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
    CeedCallHip(ceed, hipModuleUnload(impl->diag->module));
    CeedCallBackend(CeedFree(&impl->diag->h_emodein));
    CeedCallBackend(CeedFree(&impl->diag->h_emodeout));
    CeedCallHip(ceed, hipFree(impl->diag->d_emodein));
    CeedCallHip(ceed, hipFree(impl->diag->d_emodeout));
    CeedCallHip(ceed, hipFree(impl->diag->d_identity));
    CeedCallHip(ceed, hipFree(impl->diag->d_interpin));
    CeedCallHip(ceed, hipFree(impl->diag->d_interpout));
    CeedCallHip(ceed, hipFree(impl->diag->d_gradin));
    CeedCallHip(ceed, hipFree(impl->diag->d_gradout));
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->diag->pbdiagrstr));
    CeedCallBackend(CeedVectorDestroy(&impl->diag->elemdiag));
    CeedCallBackend(CeedVectorDestroy(&impl->diag->pbelemdiag));
  }
  CeedCallBackend(CeedFree(&impl->diag));

  if (impl->asmb) {
    Ceed ceed;
    CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
    CeedCallHip(ceed, hipModuleUnload(impl->asmb->module));
    CeedCallHip(ceed, hipFree(impl->asmb->d_B_in));
    CeedCallHip(ceed, hipFree(impl->asmb->d_B_out));
  }
  CeedCallBackend(CeedFree(&impl->asmb));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup infields or outfields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Hip(CeedQFunction qf, CeedOperator op, bool isinput, CeedVector *evecs, CeedVector *qvecs, CeedInt starte,
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
      // must be passive input, with emode NONE, and have a strided restriction with
      // CEED_STRIDES_BACKEND.

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
        // We do not need an E-Vector, but will use the input field vector's data
        // directly in the operator application.
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
// CeedOperator needs to connect all the named fields (be they active or passive)
//   to the named inputs and outputs of its CeedQFunction.
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Hip(CeedOperator op) {
  bool setupdone;
  CeedCallBackend(CeedOperatorIsSetupDone(op, &setupdone));
  if (setupdone) return CEED_ERROR_SUCCESS;
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Hip *impl;
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
  CeedCallBackend(CeedOperatorSetupFields_Hip(qf, op, true, impl->evecs, impl->qvecsin, 0, numinputfields, Q, numelements));

  // Outfields
  CeedCallBackend(CeedOperatorSetupFields_Hip(qf, op, false, impl->evecs, impl->qvecsout, numinputfields, numoutputfields, Q, numelements));

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Hip(CeedInt numinputfields, CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
                                              CeedVector invec, const bool skipactive, CeedScalar *edata[2 * CEED_FIELD_MAX], CeedOperator_Hip *impl,
                                              CeedRequest *request) {
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
static inline int CeedOperatorInputBasis_Hip(CeedInt numelements, CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
                                             CeedInt numinputfields, const bool skipactive, CeedScalar *edata[2 * CEED_FIELD_MAX],
                                             CeedOperator_Hip *impl) {
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
static inline int CeedOperatorRestoreInputs_Hip(CeedInt numinputfields, CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
                                                const bool skipactive, CeedScalar *edata[2 * CEED_FIELD_MAX], CeedOperator_Hip *impl) {
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
      if (!impl->evecs[i]) {  // This was a skiprestrict case
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
static int CeedOperatorApplyAdd_Hip(CeedOperator op, CeedVector invec, CeedVector outvec, CeedRequest *request) {
  CeedOperator_Hip *impl;
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
  CeedScalar         *edata[2 * CEED_FIELD_MAX];

  // Setup
  CeedCallBackend(CeedOperatorSetup_Hip(op));

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Hip(numinputfields, qfinputfields, opinputfields, invec, false, edata, impl, request));

  // Input basis apply if needed
  CeedCallBackend(CeedOperatorInputBasis_Hip(numelements, qfinputfields, opinputfields, numinputfields, false, edata, impl));

  // Output pointers, as necessary
  for (CeedInt i = 0; i < numoutputfields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode));
    if (emode == CEED_EVAL_NONE) {
      // Set the output Q-Vector to use the E-Vector data directly.
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
      case CEED_EVAL_WEIGHT: {
        Ceed ceed;
        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        break;  // Should not occur
      }
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
  CeedCallBackend(CeedOperatorRestoreInputs_Hip(numinputfields, qfinputfields, opinputfields, false, edata, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for assembling linear QFunction
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Hip(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                              CeedRequest *request) {
  CeedOperator_Hip *impl;
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
  CeedCallBackend(CeedOperatorSetup_Hip(op));

  // Check for identity
  bool identityqf;
  CeedCallBackend(CeedQFunctionIsIdentity(qf, &identityqf));
  if (identityqf) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Assembling identity QFunctions not supported");
    // LCOV_EXCL_STOP
  }

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Hip(numinputfields, qfinputfields, opinputfields, NULL, true, edata, impl, request));

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
  CeedCallBackend(CeedOperatorInputBasis_Hip(numelements, qfinputfields, opinputfields, numinputfields, true, edata, impl));

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
  CeedCallBackend(CeedOperatorRestoreInputs_Hip(numinputfields, qfinputfields, opinputfields, true, edata, impl));

  // Restore output
  CeedCallBackend(CeedVectorRestoreArray(*assembled, &a));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Hip(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Hip(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Hip(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Hip(op, false, &assembled, &rstr, request);
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
static inline int CeedOperatorAssembleDiagonalSetup_Hip(CeedOperator op, const bool pointBlock) {
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
  CeedOperator_Hip *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedCalloc(1, &impl->diag));
  CeedOperatorDiag_Hip *diag = impl->diag;
  diag->basisin              = basisin;
  diag->basisout             = basisout;
  diag->h_emodein            = emodein;
  diag->h_emodeout           = emodeout;
  diag->numemodein           = numemodein;
  diag->numemodeout          = numemodeout;

  // Assemble kernel

  char *diagonal_kernel_path, *diagonal_kernel_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-operator-assemble-diagonal.h", &diagonal_kernel_path));
  CeedDebug256(ceed, 2, "----- Loading Diagonal Assembly Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, diagonal_kernel_path, &diagonal_kernel_source));
  CeedDebug256(ceed, 2, "----- Loading Diagonal Assembly Source Complete! -----\n");
  CeedInt nnodes, nqpts;
  CeedCallBackend(CeedBasisGetNumNodes(basisin, &nnodes));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basisin, &nqpts));
  diag->nnodes = nnodes;
  CeedCallBackend(CeedCompileHip(ceed, diagonal_kernel_source, &diag->module, 5, "NUMEMODEIN", numemodein, "NUMEMODEOUT", numemodeout, "NNODES",
                                 nnodes, "NQPTS", nqpts, "NCOMP", ncomp));
  CeedCallBackend(CeedGetKernelHip(ceed, diag->module, "linearDiagonal", &diag->linearDiagonal));
  CeedCallBackend(CeedGetKernelHip(ceed, diag->module, "linearPointBlockDiagonal", &diag->linearPointBlock));
  CeedCallBackend(CeedFree(&diagonal_kernel_path));
  CeedCallBackend(CeedFree(&diagonal_kernel_source));

  // Basis matrices
  const CeedInt     qBytes = nqpts * sizeof(CeedScalar);
  const CeedInt     iBytes = qBytes * nnodes;
  const CeedInt     gBytes = qBytes * nnodes * dim;
  const CeedInt     eBytes = sizeof(CeedEvalMode);
  const CeedScalar *interpin, *interpout, *gradin, *gradout;

  // CEED_EVAL_NONE
  CeedScalar *identity = NULL;
  bool        evalNone = false;
  for (CeedInt i = 0; i < numemodein; i++) evalNone = evalNone || (emodein[i] == CEED_EVAL_NONE);
  for (CeedInt i = 0; i < numemodeout; i++) evalNone = evalNone || (emodeout[i] == CEED_EVAL_NONE);
  if (evalNone) {
    CeedCallBackend(CeedCalloc(nqpts * nnodes, &identity));
    for (CeedInt i = 0; i < (nnodes < nqpts ? nnodes : nqpts); i++) identity[i * nnodes + i] = 1.0;
    CeedCallHip(ceed, hipMalloc((void **)&diag->d_identity, iBytes));
    CeedCallHip(ceed, hipMemcpy(diag->d_identity, identity, iBytes, hipMemcpyHostToDevice));
  }

  // CEED_EVAL_INTERP
  CeedCallBackend(CeedBasisGetInterp(basisin, &interpin));
  CeedCallHip(ceed, hipMalloc((void **)&diag->d_interpin, iBytes));
  CeedCallHip(ceed, hipMemcpy(diag->d_interpin, interpin, iBytes, hipMemcpyHostToDevice));
  CeedCallBackend(CeedBasisGetInterp(basisout, &interpout));
  CeedCallHip(ceed, hipMalloc((void **)&diag->d_interpout, iBytes));
  CeedCallHip(ceed, hipMemcpy(diag->d_interpout, interpout, iBytes, hipMemcpyHostToDevice));

  // CEED_EVAL_GRAD
  CeedCallBackend(CeedBasisGetGrad(basisin, &gradin));
  CeedCallHip(ceed, hipMalloc((void **)&diag->d_gradin, gBytes));
  CeedCallHip(ceed, hipMemcpy(diag->d_gradin, gradin, gBytes, hipMemcpyHostToDevice));
  CeedCallBackend(CeedBasisGetGrad(basisout, &gradout));
  CeedCallHip(ceed, hipMalloc((void **)&diag->d_gradout, gBytes));
  CeedCallHip(ceed, hipMemcpy(diag->d_gradout, gradout, gBytes, hipMemcpyHostToDevice));

  // Arrays of emodes
  CeedCallHip(ceed, hipMalloc((void **)&diag->d_emodein, numemodein * eBytes));
  CeedCallHip(ceed, hipMemcpy(diag->d_emodein, emodein, numemodein * eBytes, hipMemcpyHostToDevice));
  CeedCallHip(ceed, hipMalloc((void **)&diag->d_emodeout, numemodeout * eBytes));
  CeedCallHip(ceed, hipMemcpy(diag->d_emodeout, emodeout, numemodeout * eBytes, hipMemcpyHostToDevice));

  // Restriction
  diag->diagrstr = rstrout;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalCore_Hip(CeedOperator op, CeedVector assembled, CeedRequest *request, const bool pointBlock) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Hip *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Assemble QFunction
  CeedVector          assembledqf;
  CeedElemRestriction rstr;
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembledqf, &rstr, request));
  CeedCallBackend(CeedElemRestrictionDestroy(&rstr));

  // Setup
  if (!impl->diag) CeedCallBackend(CeedOperatorAssembleDiagonalSetup_Hip(op, pointBlock));
  CeedOperatorDiag_Hip *diag = impl->diag;
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
    // Element diagonal vector
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
  int   elemsPerBlock = 1;
  int   grid          = nelem / elemsPerBlock + ((nelem / elemsPerBlock * elemsPerBlock < nelem) ? 1 : 0);
  void *args[]        = {(void *)&nelem,   &diag->d_identity, &diag->d_interpin, &diag->d_gradin,   &diag->d_interpout,
                         &diag->d_gradout, &diag->d_emodein,  &diag->d_emodeout, &assembledqfarray, &elemdiagarray};
  if (pointBlock) {
    CeedCallBackend(CeedRunKernelDimHip(ceed, diag->linearPointBlock, grid, diag->nnodes, 1, elemsPerBlock, args));
  } else {
    CeedCallBackend(CeedRunKernelDimHip(ceed, diag->linearDiagonal, grid, diag->nnodes, 1, elemsPerBlock, args));
  }

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
static int CeedOperatorLinearAssembleAddDiagonal_Hip(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Hip(op, assembled, request, false));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Hip(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Hip(op, assembled, request, true));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Single operator assembly setup
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssembleSetup_Hip(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Hip *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Get intput and output fields
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
  // i.e., when an active input has a CEED_EVAL_GRAD mode, num_emode_in will increment
  // by dim.  However, for the purposes of loading the B matrices, it will be treated
  // as one mode, and we will load/copy the entire gradient matrix at once, so
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
  CeedOperatorAssemble_Hip *asmb = impl->asmb;
  asmb->nelem                    = nelem;

  // Compile kernels
  int elemsPerBlock   = 1;
  asmb->elemsPerBlock = elemsPerBlock;
  CeedInt block_size  = esize * esize * elemsPerBlock;
  char   *assembly_kernel_path, *assembly_kernel_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-operator-assemble.h", &assembly_kernel_path));
  CeedDebug256(ceed, 2, "----- Loading Assembly Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, assembly_kernel_path, &assembly_kernel_source));
  CeedDebug256(ceed, 2, "----- Loading Assembly Source Complete! -----\n");
  bool fallback = block_size > 1024;
  if (fallback) {  // Use fallback kernel with 1D threadblock
    block_size         = esize * elemsPerBlock;
    asmb->block_size_x = esize;
    asmb->block_size_y = 1;
  } else {  // Use kernel with 2D threadblock
    asmb->block_size_x = esize;
    asmb->block_size_y = esize;
  }
  CeedCallBackend(CeedCompileHip(ceed, assembly_kernel_source, &asmb->module, 7, "NELEM", nelem, "NUMEMODEIN", num_emode_in, "NUMEMODEOUT",
                                 num_emode_out, "NQPTS", nqpts, "NNODES", esize, "BLOCK_SIZE", block_size, "NCOMP", ncomp));
  CeedCallBackend(CeedGetKernelHip(ceed, asmb->module, fallback ? "linearAssembleFallback" : "linearAssemble", &asmb->linearAssemble));
  CeedCallBackend(CeedFree(&assembly_kernel_path));
  CeedCallBackend(CeedFree(&assembly_kernel_source));

  // Build 'full' B matrices (not 1D arrays used for tensor-product matrices)
  const CeedScalar *interp_in, *grad_in;
  CeedCallBackend(CeedBasisGetInterp(basis_in, &interp_in));
  CeedCallBackend(CeedBasisGetGrad(basis_in, &grad_in));

  // Load into B_in, in order that they will be used in eval_mode
  const CeedInt inBytes   = size_B_in * sizeof(CeedScalar);
  CeedInt       mat_start = 0;
  CeedCallHip(ceed, hipMalloc((void **)&asmb->d_B_in, inBytes));
  for (int i = 0; i < num_B_in_mats_to_load; i++) {
    CeedEvalMode eval_mode = eval_mode_in[i];
    if (eval_mode == CEED_EVAL_INTERP) {
      CeedCallHip(ceed, hipMemcpy(&asmb->d_B_in[mat_start], interp_in, esize * nqpts * sizeof(CeedScalar), hipMemcpyHostToDevice));
      mat_start += esize * nqpts;
    } else if (eval_mode == CEED_EVAL_GRAD) {
      CeedCallHip(ceed, hipMemcpy(&asmb->d_B_in[mat_start], grad_in, dim * esize * nqpts * sizeof(CeedScalar), hipMemcpyHostToDevice));
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
  const CeedInt outBytes = size_B_out * sizeof(CeedScalar);
  mat_start              = 0;
  CeedCallHip(ceed, hipMalloc((void **)&asmb->d_B_out, outBytes));
  for (int i = 0; i < num_B_out_mats_to_load; i++) {
    CeedEvalMode eval_mode = eval_mode_out[i];
    if (eval_mode == CEED_EVAL_INTERP) {
      CeedCallHip(ceed, hipMemcpy(&asmb->d_B_out[mat_start], interp_out, esize * nqpts * sizeof(CeedScalar), hipMemcpyHostToDevice));
      mat_start += esize * nqpts;
    } else if (eval_mode == CEED_EVAL_GRAD) {
      CeedCallHip(ceed, hipMemcpy(&asmb->d_B_out[mat_start], grad_out, dim * esize * nqpts * sizeof(CeedScalar), hipMemcpyHostToDevice));
      mat_start += dim * esize * nqpts;
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble matrix data for COO matrix of assembled operator.
// The sparsity pattern is set by CeedOperatorLinearAssembleSymbolic.
//
// Note that this (and other assembly routines) currently assume only one
// active input restriction/basis per operator (could have multiple basis eval
// modes).
// TODO: allow multiple active input restrictions/basis objects
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssemble_Hip(CeedOperator op, CeedInt offset, CeedVector values) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Hip *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Setup
  if (!impl->asmb) {
    CeedCallBackend(CeedSingleOperatorAssembleSetup_Hip(op));
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
  const CeedInt nelem         = impl->asmb->nelem;  // to satisfy clang-tidy
  const CeedInt elemsPerBlock = impl->asmb->elemsPerBlock;
  const CeedInt grid          = nelem / elemsPerBlock + ((nelem / elemsPerBlock * elemsPerBlock < nelem) ? 1 : 0);
  void         *args[]        = {&impl->asmb->d_B_in, &impl->asmb->d_B_out, &qf_array, &values_array};
  CeedCallBackend(
      CeedRunKernelDimHip(ceed, impl->asmb->linearAssemble, grid, impl->asmb->block_size_x, impl->asmb->block_size_y, elemsPerBlock, args));

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
int CeedOperatorCreate_Hip(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Hip *impl;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonal_Hip));
  CeedCallBackend(
      CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddPointBlockDiagonal", CeedOperatorLinearAssembleAddPointBlockDiagonal_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleSingle", CeedSingleOperatorAssemble_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
