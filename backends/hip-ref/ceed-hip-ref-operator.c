// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include "ceed-hip-ref.h"
#include "../hip/ceed-hip-compile.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Hip(CeedOperator op) {
  int ierr;
  CeedOperator_Hip *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);

  // Apply data
  for (CeedInt i = 0; i < impl->numein + impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->evecs); CeedChkBackend(ierr);

  for (CeedInt i = 0; i < impl->numein; i++) {
    ierr = CeedVectorDestroy(&impl->qvecsin[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->qvecsin); CeedChkBackend(ierr);

  for (CeedInt i = 0; i < impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->qvecsout[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->qvecsout); CeedChkBackend(ierr);

  // QFunction diagonal assembly data
  for (CeedInt i=0; i<impl->qfnumactivein; i++) {
    ierr = CeedVectorDestroy(&impl->qfactivein[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->qfactivein); CeedChkBackend(ierr);

  // Diag data
  if (impl->diag) {
    Ceed ceed;
    ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
    CeedChk_Hip(ceed, hipModuleUnload(impl->diag->module));
    ierr = CeedFree(&impl->diag->h_emodein); CeedChkBackend(ierr);
    ierr = CeedFree(&impl->diag->h_emodeout); CeedChkBackend(ierr);
    ierr = hipFree(impl->diag->d_emodein); CeedChk_Hip(ceed, ierr);
    ierr = hipFree(impl->diag->d_emodeout); CeedChk_Hip(ceed, ierr);
    ierr = hipFree(impl->diag->d_identity); CeedChk_Hip(ceed, ierr);
    ierr = hipFree(impl->diag->d_interpin); CeedChk_Hip(ceed, ierr);
    ierr = hipFree(impl->diag->d_interpout); CeedChk_Hip(ceed, ierr);
    ierr = hipFree(impl->diag->d_gradin); CeedChk_Hip(ceed, ierr);
    ierr = hipFree(impl->diag->d_gradout); CeedChk_Hip(ceed, ierr);
    ierr = CeedElemRestrictionDestroy(&impl->diag->pbdiagrstr);
    CeedChkBackend(ierr);
    ierr = CeedVectorDestroy(&impl->diag->elemdiag); CeedChkBackend(ierr);
    ierr = CeedVectorDestroy(&impl->diag->pbelemdiag); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->diag); CeedChkBackend(ierr);

  if (impl->asmb) {
    Ceed ceed;
    ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
    CeedChk_Hip(ceed, hipModuleUnload(impl->asmb->module));
    ierr = hipFree(impl->asmb->d_B_in); CeedChk_Hip(ceed, ierr);
    ierr = hipFree(impl->asmb->d_B_out); CeedChk_Hip(ceed, ierr);
  }
  ierr = CeedFree(&impl->asmb); CeedChkBackend(ierr);

  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup infields or outfields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Hip(CeedQFunction qf, CeedOperator op,
                                       bool isinput, CeedVector *evecs,
                                       CeedVector *qvecs, CeedInt starte,
                                       CeedInt numfields, CeedInt Q,
                                       CeedInt numelements) {
  CeedInt dim, ierr, size;
  CeedSize q_size;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedBasis basis;
  CeedElemRestriction Erestrict;
  CeedOperatorField *opfields;
  CeedQFunctionField *qffields;
  CeedVector fieldvec;
  bool strided;
  bool skiprestrict;

  if (isinput) {
    ierr = CeedOperatorGetFields(op, NULL, &opfields, NULL, NULL);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionGetFields(qf, NULL, &qffields, NULL, NULL);
    CeedChkBackend(ierr);
  } else {
    ierr = CeedOperatorGetFields(op, NULL, NULL, NULL, &opfields);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qffields);
    CeedChkBackend(ierr);
  }

  // Loop over fields
  for (CeedInt i = 0; i < numfields; i++) {
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChkBackend(ierr);

    strided = false;
    skiprestrict = false;
    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &Erestrict);
      CeedChkBackend(ierr);

      // Check whether this field can skip the element restriction:
      // must be passive input, with emode NONE, and have a strided restriction with
      // CEED_STRIDES_BACKEND.

      // First, check whether the field is input or output:
      if (isinput) {
        // Check for passive input:
        ierr = CeedOperatorFieldGetVector(opfields[i], &fieldvec); CeedChkBackend(ierr);
        if (fieldvec != CEED_VECTOR_ACTIVE) {
          // Check emode
          if (emode == CEED_EVAL_NONE) {
            // Check for strided restriction
            ierr = CeedElemRestrictionIsStrided(Erestrict, &strided);
            CeedChkBackend(ierr);
            if (strided) {
              // Check if vector is already in preferred backend ordering
              ierr = CeedElemRestrictionHasBackendStrides(Erestrict,
                     &skiprestrict); CeedChkBackend(ierr);
            }
          }
        }
      }
      if (skiprestrict) {
        // We do not need an E-Vector, but will use the input field vector's data
        // directly in the operator application.
        evecs[i + starte] = NULL;
      } else {
        ierr = CeedElemRestrictionCreateVector(Erestrict, NULL,
                                               &evecs[i + starte]);
        CeedChkBackend(ierr);
      }
    }

    switch (emode) {
    case CEED_EVAL_NONE:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChkBackend(ierr);
      q_size = (CeedSize)numelements * Q * size;
      ierr = CeedVectorCreate(ceed, q_size, &qvecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChkBackend(ierr);
      q_size = (CeedSize)numelements * Q * size;
      ierr = CeedVectorCreate(ceed, q_size, &qvecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      q_size = (CeedSize)numelements * Q * size;
      ierr = CeedVectorCreate(ceed, q_size, &qvecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChkBackend(ierr);
      q_size = (CeedSize)numelements * Q;
      ierr = CeedVectorCreate(ceed, q_size, &qvecs[i]); CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
                            CEED_EVAL_WEIGHT, NULL, qvecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// CeedOperator needs to connect all the named fields (be they active or passive)
//   to the named inputs and outputs of its CeedQFunction.
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Hip(CeedOperator op) {
  int ierr;
  bool setupdone;
  ierr = CeedOperatorIsSetupDone(op, &setupdone); CeedChkBackend(ierr);
  if (setupdone)
    return CEED_ERROR_SUCCESS;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Hip *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt Q, numelements, numinputfields, numoutputfields;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &numinputfields, &opinputfields,
                               &numoutputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qfinputfields, NULL, &qfoutputfields);
  CeedChkBackend(ierr);

  // Allocate
  ierr = CeedCalloc(numinputfields + numoutputfields, &impl->evecs);
  CeedChkBackend(ierr);

  ierr = CeedCalloc(CEED_FIELD_MAX, &impl->qvecsin); CeedChkBackend(ierr);
  ierr = CeedCalloc(CEED_FIELD_MAX, &impl->qvecsout); CeedChkBackend(ierr);

  impl->numein = numinputfields; impl->numeout = numoutputfields;

  // Set up infield and outfield evecs and qvecs
  // Infields
  ierr = CeedOperatorSetupFields_Hip(qf, op, true,
                                     impl->evecs, impl->qvecsin, 0,
                                     numinputfields, Q, numelements);
  CeedChkBackend(ierr);

  // Outfields
  ierr = CeedOperatorSetupFields_Hip(qf, op, false,
                                     impl->evecs, impl->qvecsout,
                                     numinputfields, numoutputfields, Q,
                                     numelements); CeedChkBackend(ierr);

  ierr = CeedOperatorSetSetupDone(op); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Hip(CeedInt numinputfields,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    CeedVector invec, const bool skipactive, CeedScalar *edata[2*CEED_FIELD_MAX],
    CeedOperator_Hip *impl, CeedRequest *request) {
  CeedInt ierr;
  CeedEvalMode emode;
  CeedVector vec;
  CeedElemRestriction Erestrict;

  for (CeedInt i = 0; i < numinputfields; i++) {
    // Get input vector
    ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      if (skipactive)
        continue;
      else
        vec = invec;
    }

    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
      // Get input element restriction
      ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
      CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        vec = invec;
      // Restrict, if necessary
      if (!impl->evecs[i]) {
        // No restriction for this field; read data directly from vec.
        ierr = CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE,
                                      (const CeedScalar **) &edata[i]);
        CeedChkBackend(ierr);
      } else {
        ierr = CeedElemRestrictionApply(Erestrict, CEED_NOTRANSPOSE, vec,
                                        impl->evecs[i], request); CeedChkBackend(ierr);
        // Get evec
        ierr = CeedVectorGetArrayRead(impl->evecs[i], CEED_MEM_DEVICE,
                                      (const CeedScalar **) &edata[i]);
        CeedChkBackend(ierr);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Hip(CeedInt numelements,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    CeedInt numinputfields, const bool skipactive,
    CeedScalar *edata[2*CEED_FIELD_MAX], CeedOperator_Hip *impl) {
  CeedInt ierr;
  CeedInt elemsize, size;
  CeedElemRestriction Erestrict;
  CeedEvalMode emode;
  CeedBasis basis;

  for (CeedInt i=0; i<numinputfields; i++) {
    // Skip active input
    if (skipactive) {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        continue;
    }
    // Get elemsize, emode, size
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetSize(qfinputfields[i], &size); CeedChkBackend(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      ierr = CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_DEVICE,
                                CEED_USE_POINTER, edata[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
                            CEED_EVAL_INTERP, impl->evecs[i],
                            impl->qvecsin[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
                            CEED_EVAL_GRAD, impl->evecs[i],
                            impl->qvecsin[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_WEIGHT:
      break; // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Hip(CeedInt numinputfields,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    const bool skipactive, CeedScalar *edata[2*CEED_FIELD_MAX],
    CeedOperator_Hip *impl) {
  CeedInt ierr;
  CeedEvalMode emode;
  CeedVector vec;

  for (CeedInt i = 0; i < numinputfields; i++) {
    // Skip active input
    if (skipactive) {
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        continue;
    }
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      if (!impl->evecs[i]) {  // This was a skiprestrict case
        ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
        ierr = CeedVectorRestoreArrayRead(vec,
                                          (const CeedScalar **)&edata[i]);
        CeedChkBackend(ierr);
      } else {
        ierr = CeedVectorRestoreArrayRead(impl->evecs[i],
                                          (const CeedScalar **) &edata[i]);
        CeedChkBackend(ierr);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Hip(CeedOperator op, CeedVector invec,
                                    CeedVector outvec, CeedRequest *request) {
  int ierr;
  CeedOperator_Hip *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt Q, numelements, elemsize, numinputfields, numoutputfields, size;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &numinputfields, &opinputfields,
                               &numoutputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qfinputfields, NULL, &qfoutputfields);
  CeedChkBackend(ierr);
  CeedEvalMode emode;
  CeedVector vec;
  CeedBasis basis;
  CeedElemRestriction Erestrict;
  CeedScalar *edata[2*CEED_FIELD_MAX];

  // Setup
  ierr = CeedOperatorSetup_Hip(op); CeedChkBackend(ierr);

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Hip(numinputfields, qfinputfields,
                                     opinputfields, invec, false, edata,
                                     impl, request); CeedChkBackend(ierr);

  // Input basis apply if needed
  ierr = CeedOperatorInputBasis_Hip(numelements, qfinputfields, opinputfields,
                                    numinputfields, false, edata, impl);
  CeedChkBackend(ierr);

  // Output pointers, as necessary
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_NONE) {
      // Set the output Q-Vector to use the E-Vector data directly.
      ierr = CeedVectorGetArrayWrite(impl->evecs[i + impl->numein], CEED_MEM_DEVICE,
                                     &edata[i + numinputfields]); CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->qvecsout[i], CEED_MEM_DEVICE,
                                CEED_USE_POINTER, edata[i + numinputfields]);
      CeedChkBackend(ierr);
    }
  }

  // Q function
  ierr = CeedQFunctionApply(qf, numelements * Q, impl->qvecsin, impl->qvecsout);
  CeedChkBackend(ierr);

  // Output basis apply if needed
  for (CeedInt i = 0; i < numoutputfields; i++) {
    // Get elemsize, emode, size
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &size);
    CeedChkBackend(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_TRANSPOSE,
                            CEED_EVAL_INTERP, impl->qvecsout[i],
                            impl->evecs[i + impl->numein]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_TRANSPOSE,
                            CEED_EVAL_GRAD, impl->qvecsout[i],
                            impl->evecs[i + impl->numein]); CeedChkBackend(ierr);
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
      break; // Should not occur
    }
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
      // LCOV_EXCL_STOP
    }
  }

  // Output restriction
  for (CeedInt i = 0; i < numoutputfields; i++) {
    // Restore evec
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_NONE) {
      ierr = CeedVectorRestoreArray(impl->evecs[i+impl->numein],
                                    &edata[i + numinputfields]);
      CeedChkBackend(ierr);
    }
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec);
    CeedChkBackend(ierr);
    // Restrict
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    // Active
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;

    ierr = CeedElemRestrictionApply(Erestrict, CEED_TRANSPOSE,
                                    impl->evecs[i + impl->numein], vec,
                                    request); CeedChkBackend(ierr);
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Hip(numinputfields, qfinputfields,
                                       opinputfields, false, edata, impl);
  CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for assembling linear QFunction
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Hip(CeedOperator op,
    bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
    CeedRequest *request) {
  int ierr;
  CeedOperator_Hip *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt Q, numelements, numinputfields, numoutputfields, size;
  CeedSize q_size;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &numinputfields, &opinputfields,
                               &numoutputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qfinputfields, NULL, &qfoutputfields);
  CeedChkBackend(ierr);
  CeedVector vec;
  CeedInt numactivein = impl->qfnumactivein, numactiveout = impl->qfnumactiveout;
  CeedVector *activein = impl->qfactivein;
  CeedScalar *a, *tmp;
  Ceed ceed, ceedparent;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  ierr = CeedGetOperatorFallbackParentCeed(ceed, &ceedparent);
  CeedChkBackend(ierr);
  ceedparent = ceedparent ? ceedparent : ceed;
  CeedScalar *edata[2*CEED_FIELD_MAX];

  // Setup
  ierr = CeedOperatorSetup_Hip(op); CeedChkBackend(ierr);

  // Check for identity
  bool identityqf;
  ierr = CeedQFunctionIsIdentity(qf, &identityqf); CeedChkBackend(ierr);
  if (identityqf)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Assembling identity QFunctions not supported");
  // LCOV_EXCL_STOP

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Hip(numinputfields, qfinputfields,
                                     opinputfields, NULL, true, edata,
                                     impl, request); CeedChkBackend(ierr);

  // Count number of active input fields
  if (!numactivein) {
    for (CeedInt i=0; i<numinputfields; i++) {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
      // Check if active input
      if (vec == CEED_VECTOR_ACTIVE) {
        ierr = CeedQFunctionFieldGetSize(qfinputfields[i], &size); CeedChkBackend(ierr);
        ierr = CeedVectorSetValue(impl->qvecsin[i], 0.0); CeedChkBackend(ierr);
        ierr = CeedVectorGetArray(impl->qvecsin[i], CEED_MEM_DEVICE, &tmp);
        CeedChkBackend(ierr);
        ierr = CeedRealloc(numactivein + size, &activein); CeedChkBackend(ierr);
        for (CeedInt field = 0; field < size; field++) {
          q_size = (CeedSize)Q*numelements;
          ierr = CeedVectorCreate(ceed, q_size, &activein[numactivein+field]);
          CeedChkBackend(ierr);
          ierr = CeedVectorSetArray(activein[numactivein+field], CEED_MEM_DEVICE,
                                    CEED_USE_POINTER, &tmp[field*Q*numelements]);
          CeedChkBackend(ierr);
        }
        numactivein += size;
        ierr = CeedVectorRestoreArray(impl->qvecsin[i], &tmp); CeedChkBackend(ierr);
      }
    }
    impl->qfnumactivein = numactivein;
    impl->qfactivein = activein;
  }

  // Count number of active output fields
  if (!numactiveout) {
    for (CeedInt i=0; i<numoutputfields; i++) {
      // Get output vector
      ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec);
      CeedChkBackend(ierr);
      // Check if active output
      if (vec == CEED_VECTOR_ACTIVE) {
        ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &size);
        CeedChkBackend(ierr);
        numactiveout += size;
      }
    }
    impl->qfnumactiveout = numactiveout;
  }

  // Check sizes
  if (!numactivein || !numactiveout)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Cannot assemble QFunction without active inputs "
                     "and outputs");
  // LCOV_EXCL_STOP

  // Build objects if needed
  if (build_objects) {
    // Create output restriction
    CeedInt strides[3] = {1, numelements*Q, Q}; /* *NOPAD* */
    ierr = CeedElemRestrictionCreateStrided(ceedparent, numelements, Q,
                                            numactivein*numactiveout,
                                            numactivein*numactiveout*numelements*Q,
                                            strides, rstr); CeedChkBackend(ierr);
    // Create assembled vector
    CeedSize l_size = (CeedSize)numelements*Q*numactivein*numactiveout;
    ierr = CeedVectorCreate(ceedparent, l_size, assembled); CeedChkBackend(ierr);
  }
  ierr = CeedVectorSetValue(*assembled, 0.0); CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(*assembled, CEED_MEM_DEVICE, &a);
  CeedChkBackend(ierr);

  // Input basis apply
  ierr = CeedOperatorInputBasis_Hip(numelements, qfinputfields, opinputfields,
                                    numinputfields, true, edata, impl);
  CeedChkBackend(ierr);

  // Assemble QFunction
  for (CeedInt in=0; in<numactivein; in++) {
    // Set Inputs
    ierr = CeedVectorSetValue(activein[in], 1.0); CeedChkBackend(ierr);
    if (numactivein > 1) {
      ierr = CeedVectorSetValue(activein[(in+numactivein-1)%numactivein],
                                0.0); CeedChkBackend(ierr);
    }
    // Set Outputs
    for (CeedInt out=0; out<numoutputfields; out++) {
      // Get output vector
      ierr = CeedOperatorFieldGetVector(opoutputfields[out], &vec);
      CeedChkBackend(ierr);
      // Check if active output
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedVectorSetArray(impl->qvecsout[out], CEED_MEM_DEVICE,
                           CEED_USE_POINTER, a); CeedChkBackend(ierr);
        ierr = CeedQFunctionFieldGetSize(qfoutputfields[out], &size);
        CeedChkBackend(ierr);
        a += size*Q*numelements; // Advance the pointer by the size of the output
      }
    }
    // Apply QFunction
    ierr = CeedQFunctionApply(qf, Q*numelements, impl->qvecsin, impl->qvecsout);
    CeedChkBackend(ierr);
  }

  // Un-set output Qvecs to prevent accidental overwrite of Assembled
  for (CeedInt out=0; out<numoutputfields; out++) {
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[out], &vec);
    CeedChkBackend(ierr);
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedVectorTakeArray(impl->qvecsout[out], CEED_MEM_DEVICE, NULL);
      CeedChkBackend(ierr);
    }
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Hip(numinputfields, qfinputfields,
                                       opinputfields, true, edata, impl);
  CeedChkBackend(ierr);

  // Restore output
  ierr = CeedVectorRestoreArray(*assembled, &a); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Hip(CeedOperator op,
    CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Hip(op, true, assembled, rstr,
         request);
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Hip(CeedOperator op,
    CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Hip(op, false, &assembled, &rstr,
         request);
}

//------------------------------------------------------------------------------
// Create point block restriction
//------------------------------------------------------------------------------
static int CreatePBRestriction(CeedElemRestriction rstr,
                               CeedElemRestriction *pbRstr) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(rstr, &ceed); CeedChkBackend(ierr);
  const CeedInt *offsets;
  ierr = CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets);
  CeedChkBackend(ierr);

  // Expand offsets
  CeedInt nelem, ncomp, elemsize, compstride, max = 1, *pbOffsets;
  ierr = CeedElemRestrictionGetNumElements(rstr, &nelem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstr, &ncomp); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstr, &elemsize); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetCompStride(rstr, &compstride);
  CeedChkBackend(ierr);
  CeedInt shift = ncomp;
  if (compstride != 1)
    shift *= ncomp;
  ierr = CeedCalloc(nelem*elemsize, &pbOffsets); CeedChkBackend(ierr);
  for (CeedInt i = 0; i < nelem*elemsize; i++) {
    pbOffsets[i] = offsets[i]*shift;
    if (pbOffsets[i] > max)
      max = pbOffsets[i];
  }

  // Create new restriction
  ierr = CeedElemRestrictionCreate(ceed, nelem, elemsize, ncomp*ncomp, 1,
                                   max + ncomp*ncomp, CEED_MEM_HOST,
                                   CEED_OWN_POINTER, pbOffsets, pbRstr);
  CeedChkBackend(ierr);

  // Cleanup
  ierr = CeedElemRestrictionRestoreOffsets(rstr, &offsets); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble diagonal setup
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalSetup_Hip(CeedOperator op,
    const bool pointBlock) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);

  // Determine active input basis
  CeedOperatorField *opfields;
  CeedQFunctionField *qffields;
  ierr = CeedOperatorGetFields(op, NULL, &opfields, NULL, NULL);
  CeedChkBackend(ierr);
  ierr = CeedQFunctionGetFields(qf, NULL, &qffields, NULL, NULL);
  CeedChkBackend(ierr);
  CeedInt numemodein = 0, ncomp = 0, dim = 1;
  CeedEvalMode *emodein = NULL;
  CeedBasis basisin = NULL;
  CeedElemRestriction rstrin = NULL;
  for (CeedInt i = 0; i < numinputfields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(opfields[i], &vec); CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedElemRestriction rstr;
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basisin); CeedChkBackend(ierr);
      ierr = CeedBasisGetNumComponents(basisin, &ncomp); CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basisin, &dim); CeedChkBackend(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &rstr);
      CeedChkBackend(ierr);
      if (rstrin && rstrin != rstr)
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND,
                         "Multi-field non-composite operator diagonal assembly not supported");
      // LCOV_EXCL_STOP
      rstrin = rstr;
      CeedEvalMode emode;
      ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
      CeedChkBackend(ierr);
      switch (emode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(numemodein + 1, &emodein); CeedChkBackend(ierr);
        emodein[numemodein] = emode;
        numemodein += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(numemodein + dim, &emodein); CeedChkBackend(ierr);
        for (CeedInt d = 0; d < dim; d++)
          emodein[numemodein+d] = emode;
        numemodein += dim;
        break;
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break; // Caught by QF Assembly
      }
    }
  }

  // Determine active output basis
  ierr = CeedOperatorGetFields(op, NULL, NULL, NULL, &opfields);
  CeedChkBackend(ierr);
  ierr = CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qffields);
  CeedChkBackend(ierr);
  CeedInt numemodeout = 0;
  CeedEvalMode *emodeout = NULL;
  CeedBasis basisout = NULL;
  CeedElemRestriction rstrout = NULL;
  for (CeedInt i = 0; i < numoutputfields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(opfields[i], &vec); CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedElemRestriction rstr;
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basisout); CeedChkBackend(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &rstr);
      CeedChkBackend(ierr);
      if (rstrout && rstrout != rstr)
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND,
                         "Multi-field non-composite operator diagonal assembly not supported");
      // LCOV_EXCL_STOP
      rstrout = rstr;
      CeedEvalMode emode;
      ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChkBackend(ierr);
      switch (emode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(numemodeout + 1, &emodeout); CeedChkBackend(ierr);
        emodeout[numemodeout] = emode;
        numemodeout += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(numemodeout + dim, &emodeout); CeedChkBackend(ierr);
        for (CeedInt d = 0; d < dim; d++)
          emodeout[numemodeout+d] = emode;
        numemodeout += dim;
        break;
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break; // Caught by QF Assembly
      }
    }
  }

  // Operator data struct
  CeedOperator_Hip *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  ierr = CeedCalloc(1, &impl->diag); CeedChkBackend(ierr);
  CeedOperatorDiag_Hip *diag = impl->diag;
  diag->basisin = basisin;
  diag->basisout = basisout;
  diag->h_emodein = emodein;
  diag->h_emodeout = emodeout;
  diag->numemodein = numemodein;
  diag->numemodeout = numemodeout;

  // Assemble kernel

  char *diagonal_kernel_path, *diagonal_kernel_source;
  ierr = CeedGetJitAbsolutePath(ceed,
                                "ceed/jit-source/hip/hip-ref-operator-assemble-diagonal.h",
                                &diagonal_kernel_path); CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Diagonal Assembly Kernel Source -----\n");
  ierr = CeedLoadSourceToBuffer(ceed, diagonal_kernel_path,
                                &diagonal_kernel_source);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2,
               "----- Loading Diagonal Assembly Source Complete! -----\n");
  CeedInt nnodes, nqpts;
  ierr = CeedBasisGetNumNodes(basisin, &nnodes); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basisin, &nqpts); CeedChkBackend(ierr);
  diag->nnodes = nnodes;
  ierr = CeedCompileHip(ceed, diagonal_kernel_source, &diag->module, 5,
                        "NUMEMODEIN", numemodein,
                        "NUMEMODEOUT", numemodeout,
                        "NNODES", nnodes,
                        "NQPTS", nqpts,
                        "NCOMP", ncomp
                       ); CeedChk_Hip(ceed, ierr);
  ierr = CeedGetKernelHip(ceed, diag->module, "linearDiagonal",
                          &diag->linearDiagonal); CeedChk_Hip(ceed, ierr);
  ierr = CeedGetKernelHip(ceed, diag->module, "linearPointBlockDiagonal",
                          &diag->linearPointBlock);
  CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&diagonal_kernel_path); CeedChkBackend(ierr);
  ierr = CeedFree(&diagonal_kernel_source); CeedChkBackend(ierr);

  // Basis matrices
  const CeedInt qBytes = nqpts * sizeof(CeedScalar);
  const CeedInt iBytes = qBytes * nnodes;
  const CeedInt gBytes = qBytes * nnodes * dim;
  const CeedInt eBytes = sizeof(CeedEvalMode);
  const CeedScalar *interpin, *interpout, *gradin, *gradout;

  // CEED_EVAL_NONE
  CeedScalar *identity = NULL;
  bool evalNone = false;
  for (CeedInt i=0; i<numemodein; i++)
    evalNone = evalNone || (emodein[i] == CEED_EVAL_NONE);
  for (CeedInt i=0; i<numemodeout; i++)
    evalNone = evalNone || (emodeout[i] == CEED_EVAL_NONE);
  if (evalNone) {
    ierr = CeedCalloc(nqpts*nnodes, &identity); CeedChkBackend(ierr);
    for (CeedInt i=0; i<(nnodes<nqpts?nnodes:nqpts); i++)
      identity[i*nnodes+i] = 1.0;
    ierr = hipMalloc((void **)&diag->d_identity, iBytes); CeedChk_Hip(ceed, ierr);
    ierr = hipMemcpy(diag->d_identity, identity, iBytes,
                     hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);
  }

  // CEED_EVAL_INTERP
  ierr = CeedBasisGetInterp(basisin, &interpin); CeedChkBackend(ierr);
  ierr = hipMalloc((void **)&diag->d_interpin, iBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(diag->d_interpin, interpin, iBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);
  ierr = CeedBasisGetInterp(basisout, &interpout); CeedChkBackend(ierr);
  ierr = hipMalloc((void **)&diag->d_interpout, iBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(diag->d_interpout, interpout, iBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  // CEED_EVAL_GRAD
  ierr = CeedBasisGetGrad(basisin, &gradin); CeedChkBackend(ierr);
  ierr = hipMalloc((void **)&diag->d_gradin, gBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(diag->d_gradin, gradin, gBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);
  ierr = CeedBasisGetGrad(basisout, &gradout); CeedChkBackend(ierr);
  ierr = hipMalloc((void **)&diag->d_gradout, gBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(diag->d_gradout, gradout, gBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  // Arrays of emodes
  ierr = hipMalloc((void **)&diag->d_emodein, numemodein * eBytes);
  CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(diag->d_emodein, emodein, numemodein * eBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);
  ierr = hipMalloc((void **)&diag->d_emodeout, numemodeout * eBytes);
  CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(diag->d_emodeout, emodeout, numemodeout * eBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  // Restriction
  diag->diagrstr = rstrout;

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalCore_Hip(CeedOperator op,
    CeedVector assembled, CeedRequest *request, const bool pointBlock) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Hip *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);

  // Assemble QFunction
  CeedVector assembledqf;
  CeedElemRestriction rstr;
  ierr = CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembledqf,
         &rstr, request); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr); CeedChkBackend(ierr);

  // Setup
  if (!impl->diag) {
    ierr = CeedOperatorAssembleDiagonalSetup_Hip(op, pointBlock);
    CeedChkBackend(ierr);
  }
  CeedOperatorDiag_Hip *diag = impl->diag;
  assert(diag != NULL);

  // Restriction
  if (pointBlock && !diag->pbdiagrstr) {
    CeedElemRestriction pbdiagrstr;
    ierr = CreatePBRestriction(diag->diagrstr, &pbdiagrstr); CeedChkBackend(ierr);
    diag->pbdiagrstr = pbdiagrstr;
  }
  CeedElemRestriction diagrstr = pointBlock ? diag->pbdiagrstr : diag->diagrstr;

  // Create diagonal vector
  CeedVector elemdiag = pointBlock ? diag->pbelemdiag : diag->elemdiag;
  if (!elemdiag) {
    // Element diagonal vector
    ierr = CeedElemRestrictionCreateVector(diagrstr, NULL, &elemdiag);
    CeedChkBackend(ierr);
    if (pointBlock)
      diag->pbelemdiag = elemdiag;
    else
      diag->elemdiag = elemdiag;
  }
  ierr = CeedVectorSetValue(elemdiag, 0.0); CeedChkBackend(ierr);

  // Assemble element operator diagonals
  CeedScalar *elemdiagarray;
  const CeedScalar *assembledqfarray;
  ierr = CeedVectorGetArray(elemdiag, CEED_MEM_DEVICE, &elemdiagarray);
  CeedChkBackend(ierr);
  ierr = CeedVectorGetArrayRead(assembledqf, CEED_MEM_DEVICE, &assembledqfarray);
  CeedChkBackend(ierr);
  CeedInt nelem;
  ierr = CeedElemRestrictionGetNumElements(diagrstr, &nelem);
  CeedChkBackend(ierr);

  // Compute the diagonal of B^T D B
  int elemsPerBlock = 1;
  int grid = nelem/elemsPerBlock+((nelem/elemsPerBlock*elemsPerBlock<nelem)?1:0);
  void *args[] = {(void *) &nelem, &diag->d_identity,
                  &diag->d_interpin, &diag->d_gradin, &diag->d_interpout,
                  &diag->d_gradout, &diag->d_emodein, &diag->d_emodeout,
                  &assembledqfarray, &elemdiagarray
                 };
  if (pointBlock) {
    ierr = CeedRunKernelDimHip(ceed, diag->linearPointBlock, grid,
                               diag->nnodes, 1, elemsPerBlock, args);
    CeedChkBackend(ierr);
  } else {
    ierr = CeedRunKernelDimHip(ceed, diag->linearDiagonal, grid,
                               diag->nnodes, 1, elemsPerBlock, args);
    CeedChkBackend(ierr);
  }

  // Restore arrays
  ierr = CeedVectorRestoreArray(elemdiag, &elemdiagarray); CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArrayRead(assembledqf, &assembledqfarray);
  CeedChkBackend(ierr);

  // Assemble local operator diagonal
  ierr = CeedElemRestrictionApply(diagrstr, CEED_TRANSPOSE, elemdiag,
                                  assembled, request); CeedChkBackend(ierr);

  // Cleanup
  ierr = CeedVectorDestroy(&assembledqf); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonal_Hip(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr = CeedOperatorAssembleDiagonalCore_Hip(op, assembled, request, false);
  CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Hip(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr = CeedOperatorAssembleDiagonalCore_Hip(op, assembled, request, true);
  CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Single operator assembly setup
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssembleSetup_Hip(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Hip *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);

  // Get intput and output fields
  CeedInt num_input_fields, num_output_fields;
  CeedOperatorField *input_fields;
  CeedOperatorField *output_fields;
  ierr = CeedOperatorGetFields(op, &num_input_fields, &input_fields,
                               &num_output_fields, &output_fields); CeedChkBackend(ierr);

  // Determine active input basis eval mode
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedQFunctionField *qf_fields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL);
  CeedChkBackend(ierr);
  // Note that the kernel will treat each dimension of a gradient action separately;
  // i.e., when an active input has a CEED_EVAL_GRAD mode, num_emode_in will increment
  // by dim.  However, for the purposes of loading the B matrices, it will be treated
  // as one mode, and we will load/copy the entire gradient matrix at once, so
  // num_B_in_mats_to_load will be incremented by 1.
  CeedInt num_emode_in = 0, dim = 1, num_B_in_mats_to_load = 0, size_B_in = 0;
  CeedEvalMode *eval_mode_in = NULL; //will be of size num_B_in_mats_load
  CeedBasis basis_in = NULL;
  CeedInt nqpts = 0, esize = 0;
  CeedElemRestriction rstr_in = NULL;
  for (CeedInt i=0; i<num_input_fields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(input_fields[i], &vec); CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorFieldGetBasis(input_fields[i], &basis_in);
      CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis_in, &dim); CeedChkBackend(ierr);
      ierr = CeedBasisGetNumQuadraturePoints(basis_in, &nqpts); CeedChkBackend(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(input_fields[i], &rstr_in);
      CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetElementSize(rstr_in, &esize); CeedChkBackend(ierr);
      CeedEvalMode eval_mode;
      ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode);
      CeedChkBackend(ierr);
      if (eval_mode != CEED_EVAL_NONE) {
        ierr = CeedRealloc(num_B_in_mats_to_load + 1, &eval_mode_in);
        CeedChkBackend(ierr);
        eval_mode_in[num_B_in_mats_to_load] = eval_mode;
        num_B_in_mats_to_load += 1;
        if (eval_mode == CEED_EVAL_GRAD) {
          num_emode_in += dim;
          size_B_in += dim * esize * nqpts;
        } else {
          num_emode_in +=1;
          size_B_in += esize * nqpts;
        }
      }
    }
  }

  // Determine active output basis; basis_out and rstr_out only used if same as input, TODO
  ierr = CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields);
  CeedChkBackend(ierr);
  CeedInt num_emode_out = 0, num_B_out_mats_to_load = 0, size_B_out = 0;
  CeedEvalMode *eval_mode_out = NULL;
  CeedBasis basis_out = NULL;
  CeedElemRestriction rstr_out = NULL;
  for (CeedInt i=0; i<num_output_fields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(output_fields[i], &vec); CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorFieldGetBasis(output_fields[i], &basis_out);
      CeedChkBackend(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(output_fields[i], &rstr_out);
      CeedChkBackend(ierr);
      if (rstr_out && rstr_out != rstr_in)
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND,
                         "Multi-field non-composite operator assembly not supported");
      // LCOV_EXCL_STOP
      CeedEvalMode eval_mode;
      ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode);
      CeedChkBackend(ierr);
      if (eval_mode != CEED_EVAL_NONE) {
        ierr = CeedRealloc(num_B_out_mats_to_load + 1, &eval_mode_out);
        CeedChkBackend(ierr);
        eval_mode_out[num_B_out_mats_to_load] = eval_mode;
        num_B_out_mats_to_load += 1;
        if (eval_mode == CEED_EVAL_GRAD) {
          num_emode_out += dim;
          size_B_out += dim * esize * nqpts;
        } else {
          num_emode_out +=1;
          size_B_out += esize * nqpts;
        }
      }
    }
  }

  if (num_emode_in == 0 || num_emode_out == 0)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "Cannot assemble operator without inputs/outputs");
  // LCOV_EXCL_STOP

  CeedInt nelem, ncomp;
  ierr = CeedElemRestrictionGetNumElements(rstr_in, &nelem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstr_in, &ncomp);
  CeedChkBackend(ierr);

  ierr = CeedCalloc(1, &impl->asmb); CeedChkBackend(ierr);
  CeedOperatorAssemble_Hip *asmb = impl->asmb;
  asmb->nelem = nelem;

  // Compile kernels
  int elemsPerBlock = 1;
  asmb->elemsPerBlock = elemsPerBlock;
  CeedInt block_size = esize * esize * elemsPerBlock;
  char *assembly_kernel_path, *assembly_kernel_source;
  ierr = CeedGetJitAbsolutePath(ceed,
                                "ceed/jit-source/hip/hip-ref-operator-assemble.h",
                                &assembly_kernel_path); CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Assembly Kernel Source -----\n");
  ierr = CeedLoadSourceToBuffer(ceed, assembly_kernel_path,
                                &assembly_kernel_source);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Assembly Source Complete! -----\n");
  bool fallback = block_size > 1024;
  if (fallback) { // Use fallback kernel with 1D threadblock
    block_size = esize * elemsPerBlock;
    asmb->block_size_x = esize;
    asmb->block_size_y = 1;
  } else {  // Use kernel with 2D threadblock
    asmb->block_size_x = esize;
    asmb->block_size_y = esize;
  }
  ierr = CeedCompileHip(ceed, assembly_kernel_source, &asmb->module, 7,
                        "NELEM", nelem,
                        "NUMEMODEIN", num_emode_in,
                        "NUMEMODEOUT", num_emode_out,
                        "NQPTS", nqpts,
                        "NNODES", esize,
                        "BLOCK_SIZE", block_size,
                        "NCOMP", ncomp
                       ); CeedChk_Hip(ceed, ierr);
  ierr = CeedGetKernelHip(ceed, asmb->module,
                          fallback ? "linearAssembleFallback" : "linearAssemble",
                          &asmb->linearAssemble); CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&assembly_kernel_path); CeedChkBackend(ierr);
  ierr = CeedFree(&assembly_kernel_source); CeedChkBackend(ierr);

  // Build 'full' B matrices (not 1D arrays used for tensor-product matrices)
  const CeedScalar *interp_in, *grad_in;
  ierr = CeedBasisGetInterp(basis_in, &interp_in); CeedChkBackend(ierr);
  ierr = CeedBasisGetGrad(basis_in, &grad_in); CeedChkBackend(ierr);

  // Load into B_in, in order that they will be used in eval_mode
  const CeedInt inBytes = size_B_in * sizeof(CeedScalar);
  CeedInt mat_start = 0;
  ierr = hipMalloc((void **) &asmb->d_B_in, inBytes); CeedChk_Hip(ceed, ierr);
  for (int i = 0; i < num_B_in_mats_to_load; i++) {
    CeedEvalMode eval_mode = eval_mode_in[i];
    if (eval_mode == CEED_EVAL_INTERP) {
      ierr = hipMemcpy(&asmb->d_B_in[mat_start], interp_in,
                       esize * nqpts * sizeof(CeedScalar),
                       hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);
      mat_start += esize * nqpts;
    } else if (eval_mode == CEED_EVAL_GRAD) {
      ierr = hipMemcpy(&asmb->d_B_in[mat_start], grad_in,
                       dim * esize * nqpts * sizeof(CeedScalar),
                       hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);
      mat_start += dim * esize * nqpts;
    }
  }

  const CeedScalar *interp_out, *grad_out;
  // Note that this function currently assumes 1 basis, so this should always be true
  // for now
  if (basis_out == basis_in) {
    interp_out = interp_in;
    grad_out = grad_in;
  } else {
    ierr = CeedBasisGetInterp(basis_out, &interp_out); CeedChkBackend(ierr);
    ierr = CeedBasisGetGrad(basis_out, &grad_out); CeedChkBackend(ierr);
  }

  // Load into B_out, in order that they will be used in eval_mode
  const CeedInt outBytes = size_B_out * sizeof(CeedScalar);
  mat_start = 0;
  ierr = hipMalloc((void **) &asmb->d_B_out, outBytes); CeedChk_Hip(ceed, ierr);
  for (int i = 0; i < num_B_out_mats_to_load; i++) {
    CeedEvalMode eval_mode = eval_mode_out[i];
    if (eval_mode == CEED_EVAL_INTERP) {
      ierr = hipMemcpy(&asmb->d_B_out[mat_start], interp_out,
                       esize * nqpts * sizeof(CeedScalar),
                       hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);
      mat_start += esize * nqpts;
    } else if (eval_mode == CEED_EVAL_GRAD) {
      ierr = hipMemcpy(&asmb->d_B_out[mat_start], grad_out,
                       dim * esize * nqpts * sizeof(CeedScalar),
                       hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);
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
static int CeedSingleOperatorAssemble_Hip(CeedOperator op, CeedInt offset,
    CeedVector values) {

  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Hip *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);

  // Setup
  if (!impl->asmb) {
    ierr = CeedSingleOperatorAssembleSetup_Hip(op);
    CeedChkBackend(ierr);
    assert(impl->asmb != NULL);
  }

  // Assemble QFunction
  CeedVector assembled_qf;
  CeedElemRestriction rstr_q;
  ierr = CeedOperatorLinearAssembleQFunctionBuildOrUpdate(
           op, &assembled_qf, &rstr_q, CEED_REQUEST_IMMEDIATE); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_q); CeedChkBackend(ierr);
  CeedScalar *values_array;
  ierr = CeedVectorGetArrayWrite(values, CEED_MEM_DEVICE, &values_array);
  CeedChkBackend(ierr);
  values_array += offset;
  const CeedScalar *qf_array;
  ierr = CeedVectorGetArrayRead(assembled_qf, CEED_MEM_DEVICE, &qf_array);
  CeedChkBackend(ierr);

  // Compute B^T D B
  const CeedInt nelem = impl->asmb->nelem; // to satisfy clang-tidy
  const CeedInt elemsPerBlock = impl->asmb->elemsPerBlock;
  const CeedInt grid = nelem/elemsPerBlock+((
                         nelem/elemsPerBlock*elemsPerBlock<nelem)?1:0);
  void *args[] = {&impl->asmb->d_B_in, &impl->asmb->d_B_out,
                  &qf_array, &values_array
                 };
  ierr = CeedRunKernelDimHip(ceed, impl->asmb->linearAssemble, grid,
                             impl->asmb->block_size_x, impl->asmb->block_size_y,
                             elemsPerBlock, args);
  CeedChkBackend(ierr);


  // Restore arrays
  ierr = CeedVectorRestoreArray(values, &values_array); CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArrayRead(assembled_qf, &qf_array);
  CeedChkBackend(ierr);

  // Cleanup
  ierr = CeedVectorDestroy(&assembled_qf); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Hip(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Hip *impl;

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedOperatorSetData(op, impl); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction",
                                CeedOperatorLinearAssembleQFunction_Hip);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleQFunctionUpdate",
                                CeedOperatorLinearAssembleQFunctionUpdate_Hip);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal",
                                CeedOperatorLinearAssembleAddDiagonal_Hip);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleAddPointBlockDiagonal",
                                CeedOperatorLinearAssembleAddPointBlockDiagonal_Hip);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleSingle", CeedSingleOperatorAssemble_Hip);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd",
                                CeedOperatorApplyAdd_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Hip); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
