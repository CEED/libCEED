// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include "ceed-hip.h"
#include "ceed-hip-compile.h"

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
  ierr = CeedFree(&impl->edata); CeedChkBackend(ierr);

  for (CeedInt i = 0; i < impl->numein; i++) {
    ierr = CeedVectorDestroy(&impl->qvecsin[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->qvecsin); CeedChkBackend(ierr);

  for (CeedInt i = 0; i < impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->qvecsout[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->qvecsout); CeedChkBackend(ierr);

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
  }
  ierr = CeedFree(&impl->diag); CeedChkBackend(ierr);

  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup infields or outfields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Hip(CeedQFunction qf, CeedOperator op,
                                       bool inOrOut, CeedVector *evecs,
                                       CeedVector *qvecs, CeedInt starte,
                                       CeedInt numfields, CeedInt Q,
                                       CeedInt numelements) {
  CeedInt dim, ierr, size;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedBasis basis;
  CeedElemRestriction Erestrict;
  CeedOperatorField *opfields;
  CeedQFunctionField *qffields;
  CeedVector fieldvec;
  bool strided;
  bool skiprestrict;

  if (inOrOut) {
    ierr = CeedOperatorGetFields(op, NULL, &opfields);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionGetFields(qf, NULL, &qffields);
    CeedChkBackend(ierr);
  } else {
    ierr = CeedOperatorGetFields(op, &opfields, NULL);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionGetFields(qf, &qffields, NULL);
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
      if (!inOrOut) {
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
      ierr = CeedVectorCreate(ceed, numelements * Q * size, &qvecs[i]);
      CeedChkBackend(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, numelements * Q * size, &qvecs[i]);
      CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, numelements * Q * size, &qvecs[i]);
      CeedChkBackend(ierr);
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, numelements * Q, &qvecs[i]); CeedChkBackend(ierr);
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
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChkBackend(ierr);

  // Allocate
  ierr = CeedCalloc(numinputfields + numoutputfields, &impl->evecs);
  CeedChkBackend(ierr);
  ierr = CeedCalloc(numinputfields + numoutputfields, &impl->edata);
  CeedChkBackend(ierr);

  ierr = CeedCalloc(16, &impl->qvecsin); CeedChkBackend(ierr);
  ierr = CeedCalloc(16, &impl->qvecsout); CeedChkBackend(ierr);

  impl->numein = numinputfields; impl->numeout = numoutputfields;

  // Set up infield and outfield evecs and qvecs
  // Infields
  ierr = CeedOperatorSetupFields_Hip(qf, op, 0,
                                     impl->evecs, impl->qvecsin, 0,
                                     numinputfields, Q, numelements);
  CeedChkBackend(ierr);

  // Outfields
  ierr = CeedOperatorSetupFields_Hip(qf, op, 1,
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
    CeedVector invec, const bool skipactive, CeedOperator_Hip *impl,
    CeedRequest *request) {
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
                                      (const CeedScalar **) &impl->edata[i]);
        CeedChkBackend(ierr);
      } else {
        ierr = CeedElemRestrictionApply(Erestrict, CEED_NOTRANSPOSE, vec,
                                        impl->evecs[i], request); CeedChkBackend(ierr);
        // Get evec
        ierr = CeedVectorGetArrayRead(impl->evecs[i], CEED_MEM_DEVICE,
                                      (const CeedScalar **) &impl->edata[i]);
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
    CeedInt numinputfields, const bool skipactive, CeedOperator_Hip *impl) {
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
                                CEED_USE_POINTER,
                                impl->edata[i]); CeedChkBackend(ierr);
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
    const bool skipactive, CeedOperator_Hip *impl) {
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
                                          (const CeedScalar **)&impl->edata[i]);
        CeedChkBackend(ierr);
      } else {
        ierr = CeedVectorRestoreArrayRead(impl->evecs[i],
                                          (const CeedScalar **) &impl->edata[i]);
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
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChkBackend(ierr);
  CeedEvalMode emode;
  CeedVector vec;
  CeedBasis basis;
  CeedElemRestriction Erestrict;

  // Setup
  ierr = CeedOperatorSetup_Hip(op); CeedChkBackend(ierr);

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Hip(numinputfields, qfinputfields,
                                     opinputfields, invec, false, impl,
                                     request); CeedChkBackend(ierr);

  // Input basis apply if needed
  ierr = CeedOperatorInputBasis_Hip(numelements, qfinputfields, opinputfields,
                                    numinputfields, false, impl);
  CeedChkBackend(ierr);

  // Output pointers, as necessary
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_NONE) {
      // Set the output Q-Vector to use the E-Vector data directly.
      ierr = CeedVectorGetArray(impl->evecs[i + impl->numein], CEED_MEM_DEVICE,
                                &impl->edata[i + numinputfields]); CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->qvecsout[i], CEED_MEM_DEVICE,
                                CEED_USE_POINTER,
                                impl->edata[i + numinputfields]);
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
                                    &impl->edata[i + numinputfields]);
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
                                       opinputfields, false, impl);
  CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Hip(CeedOperator op,
    CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  int ierr;
  CeedOperator_Hip *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt Q, numelements, numinputfields, numoutputfields, size;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChkBackend(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChkBackend(ierr);
  CeedVector vec;
  CeedInt numactivein = 0, numactiveout = 0;
  CeedVector *activein = NULL;
  CeedScalar *a, *tmp;
  Ceed ceed, ceedparent;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  ierr = CeedGetOperatorFallbackParentCeed(ceed, &ceedparent);
  CeedChkBackend(ierr);
  ceedparent = ceedparent ? ceedparent : ceed;

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
                                     opinputfields, NULL, true, impl, request);
  CeedChkBackend(ierr);

  // Count number of active input fields
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
        ierr = CeedVectorCreate(ceed, Q*numelements,
                                &activein[numactivein+field]); CeedChkBackend(ierr);
        ierr = CeedVectorSetArray(activein[numactivein+field], CEED_MEM_DEVICE,
                                  CEED_USE_POINTER, &tmp[field*Q*numelements]);
        CeedChkBackend(ierr);
      }
      numactivein += size;
      ierr = CeedVectorRestoreArray(impl->qvecsin[i], &tmp); CeedChkBackend(ierr);
    }
  }

  // Count number of active output fields
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

  // Check sizes
  if (!numactivein || !numactiveout)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Cannot assemble QFunction without active inputs "
                     "and outputs");
  // LCOV_EXCL_STOP

  // Create output restriction
  CeedInt strides[3] = {1, numelements*Q, Q}; /* *NOPAD* */
  ierr = CeedElemRestrictionCreateStrided(ceedparent, numelements, Q,
                                          numactivein*numactiveout,
                                          numactivein*numactiveout*numelements*Q,
                                          strides, rstr); CeedChkBackend(ierr);
  // Create assembled vector
  ierr = CeedVectorCreate(ceedparent, numelements*Q*numactivein*numactiveout,
                          assembled); CeedChkBackend(ierr);
  ierr = CeedVectorSetValue(*assembled, 0.0); CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(*assembled, CEED_MEM_DEVICE, &a);
  CeedChkBackend(ierr);

  // Input basis apply
  ierr = CeedOperatorInputBasis_Hip(numelements, qfinputfields, opinputfields,
                                    numinputfields, true, impl);
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
                                       opinputfields, true, impl);
  CeedChkBackend(ierr);

  // Restore output
  ierr = CeedVectorRestoreArray(*assembled, &a); CeedChkBackend(ierr);

  // Cleanup
  for (CeedInt i=0; i<numactivein; i++) {
    ierr = CeedVectorDestroy(&activein[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&activein); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Diagonal assembly kernels
//------------------------------------------------------------------------------
// *INDENT-OFF*
static const char *diagonalkernels = QUOTE(

typedef enum {
  /// Perform no evaluation (either because there is no data or it is already at
  /// quadrature points)
  CEED_EVAL_NONE   = 0,
  /// Interpolate from nodes to quadrature points
  CEED_EVAL_INTERP = 1,
  /// Evaluate gradients at quadrature points from input in a nodal basis
  CEED_EVAL_GRAD   = 2,
  /// Evaluate divergence at quadrature points from input in a nodal basis
  CEED_EVAL_DIV    = 4,
  /// Evaluate curl at quadrature points from input in a nodal basis
  CEED_EVAL_CURL   = 8,
  /// Using no input, evaluate quadrature weights on the reference element
  CEED_EVAL_WEIGHT = 16,
} CeedEvalMode;

//------------------------------------------------------------------------------
// Get Basis Emode Pointer
//------------------------------------------------------------------------------
extern "C" __device__ void CeedOperatorGetBasisPointer_Hip(const CeedScalar **basisptr,
    CeedEvalMode emode, const CeedScalar *identity, const CeedScalar *interp,
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
    break; // Caught by QF Assembly
  }
}

//------------------------------------------------------------------------------
// Core code for diagonal assembly
//------------------------------------------------------------------------------
__device__ void diagonalCore(const CeedInt nelem,
    const CeedScalar maxnorm, const bool pointBlock,
    const CeedScalar *identity,
    const CeedScalar *interpin, const CeedScalar *gradin,
    const CeedScalar *interpout, const CeedScalar *gradout,
    const CeedEvalMode *emodein, const CeedEvalMode *emodeout,
    const CeedScalar *__restrict__ assembledqfarray,
    CeedScalar *__restrict__ elemdiagarray) {
  const int tid = threadIdx.x; // running with P threads, tid is evec node
  const CeedScalar qfvaluebound = maxnorm*1e-12;

  // Compute the diagonal of B^T D B
  // Each element
  for (CeedInt e = blockIdx.x*blockDim.z + threadIdx.z; e < nelem;
       e += gridDim.x*blockDim.z) {
    CeedInt dout = -1;
    // Each basis eval mode pair
    for (CeedInt eout = 0; eout < NUMEMODEOUT; eout++) {
      const CeedScalar *bt = NULL;
      if (emodeout[eout] == CEED_EVAL_GRAD)
        dout += 1;
      CeedOperatorGetBasisPointer_Hip(&bt, emodeout[eout], identity, interpout,
                                      &gradout[dout*NQPTS*NNODES]);
      CeedInt din = -1;
      for (CeedInt ein = 0; ein < NUMEMODEIN; ein++) {
        const CeedScalar *b = NULL;
        if (emodein[ein] == CEED_EVAL_GRAD)
          din += 1;
        CeedOperatorGetBasisPointer_Hip(&b, emodein[ein], identity, interpin,
                                        &gradin[din*NQPTS*NNODES]);
        // Each component
        for (CeedInt compOut = 0; compOut < NCOMP; compOut++) {
          // Each qpoint/node pair
          if (pointBlock) {
            // Point Block Diagonal
            for (CeedInt compIn = 0; compIn < NCOMP; compIn++) {
              CeedScalar evalue = 0.;
              for (CeedInt q = 0; q < NQPTS; q++) {
                const CeedScalar qfvalue =
                  assembledqfarray[((((ein*NCOMP+compIn)*NUMEMODEOUT+eout)*
                                     NCOMP+compOut)*nelem+e)*NQPTS+q];
                if (abs(qfvalue) > qfvaluebound)
                  evalue += bt[q*NNODES+tid] * qfvalue * b[q*NNODES+tid];
              }
              elemdiagarray[((compOut*NCOMP+compIn)*nelem+e)*NNODES+tid] += evalue;
            }
          } else {
            // Diagonal Only
            CeedScalar evalue = 0.;
            for (CeedInt q = 0; q < NQPTS; q++) {
              const CeedScalar qfvalue =
                assembledqfarray[((((ein*NCOMP+compOut)*NUMEMODEOUT+eout)*
                                   NCOMP+compOut)*nelem+e)*NQPTS+q];
              if (abs(qfvalue) > qfvaluebound)
                evalue += bt[q*NNODES+tid] * qfvalue * b[q*NNODES+tid];
            }
            elemdiagarray[(compOut*nelem+e)*NNODES+tid] += evalue;
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// Linear diagonal
//------------------------------------------------------------------------------
extern "C" __global__ void linearDiagonal(const CeedInt nelem,
    const CeedScalar maxnorm, const CeedScalar *identity,
    const CeedScalar *interpin, const CeedScalar *gradin,
    const CeedScalar *interpout, const CeedScalar *gradout,
    const CeedEvalMode *emodein, const CeedEvalMode *emodeout,
    const CeedScalar *__restrict__ assembledqfarray,
    CeedScalar *__restrict__ elemdiagarray) {
  diagonalCore(nelem, maxnorm, false, identity, interpin, gradin, interpout,
               gradout, emodein, emodeout, assembledqfarray, elemdiagarray);
}

//------------------------------------------------------------------------------
// Linear point block diagonal
//------------------------------------------------------------------------------
extern "C" __global__ void linearPointBlockDiagonal(const CeedInt nelem,
    const CeedScalar maxnorm, const CeedScalar *identity,
    const CeedScalar *interpin, const CeedScalar *gradin,
    const CeedScalar *interpout, const CeedScalar *gradout,
    const CeedEvalMode *emodein, const CeedEvalMode *emodeout,
    const CeedScalar *__restrict__ assembledqfarray,
    CeedScalar *__restrict__ elemdiagarray) {
  diagonalCore(nelem, maxnorm, true, identity, interpin, gradin, interpout,
               gradout, emodein, emodeout, assembledqfarray, elemdiagarray);
}

);
// *INDENT-ON*

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
  ierr = CeedOperatorGetFields(op, &opfields, NULL); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetFields(qf, &qffields, NULL); CeedChkBackend(ierr);
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
  ierr = CeedOperatorGetFields(op, NULL, &opfields); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetFields(qf, NULL, &qffields); CeedChkBackend(ierr);
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
  CeedInt nnodes, nqpts;
  ierr = CeedBasisGetNumNodes(basisin, &nnodes); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basisin, &nqpts); CeedChkBackend(ierr);
  diag->nnodes = nnodes;
  ierr = CeedCompileHip(ceed, diagonalkernels, &diag->module, 5,
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
  ierr = CeedOperatorLinearAssembleQFunction(op,  &assembledqf, &rstr, request);
  CeedChkBackend(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr); CeedChkBackend(ierr);
  CeedScalar maxnorm = 0;
  ierr = CeedVectorNorm(assembledqf, CEED_NORM_MAX, &maxnorm);
  CeedChkBackend(ierr);

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
  CeedVector elemdiag;
  ierr = CeedElemRestrictionCreateVector(diagrstr, NULL, &elemdiag);
  CeedChkBackend(ierr);
  ierr = CeedVectorSetValue(elemdiag, 0.0); CeedChkBackend(ierr);

  // Assemble element operator diagonals
  CeedScalar *elemdiagarray, *assembledqfarray;
  ierr = CeedVectorGetArray(elemdiag, CEED_MEM_DEVICE, &elemdiagarray);
  CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(assembledqf, CEED_MEM_DEVICE, &assembledqfarray);
  CeedChkBackend(ierr);
  CeedInt nelem;
  ierr = CeedElemRestrictionGetNumElements(diagrstr, &nelem);
  CeedChkBackend(ierr);

  // Compute the diagonal of B^T D B
  int elemsPerBlock = 1;
  int grid = nelem/elemsPerBlock+((nelem/elemsPerBlock*elemsPerBlock<nelem)?1:0);
  void *args[] = {(void *) &nelem, (void *) &maxnorm, &diag->d_identity,
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
  ierr = CeedVectorRestoreArray(assembledqf, &assembledqfarray);
  CeedChkBackend(ierr);

  // Assemble local operator diagonal
  ierr = CeedElemRestrictionApply(diagrstr, CEED_TRANSPOSE, elemdiag,
                                  assembled, request); CeedChkBackend(ierr);

  // Cleanup
  ierr = CeedVectorDestroy(&assembledqf); CeedChkBackend(ierr);
  ierr = CeedVectorDestroy(&elemdiag); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble composite diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleAddDiagonalCompositeCore_Hip(
  CeedOperator op, CeedVector assembled, CeedRequest *request,
  const bool pointBlock) {
  int ierr;
  CeedInt numSub;
  CeedOperator *subOperators;
  ierr = CeedOperatorGetNumSub(op, &numSub); CeedChkBackend(ierr);
  ierr = CeedOperatorGetSubList(op, &subOperators); CeedChkBackend(ierr);
  for (CeedInt i = 0; i < numSub; i++) {
    ierr = CeedOperatorAssembleDiagonalCore_Hip(subOperators[i], assembled,
           request, pointBlock); CeedChkBackend(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonal_Hip(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr;
  bool isComposite;
  ierr = CeedOperatorIsComposite(op, &isComposite); CeedChkBackend(ierr);
  if (isComposite) {
    return CeedOperatorLinearAssembleAddDiagonalCompositeCore_Hip(op, assembled,
           request, false);
  } else {
    return CeedOperatorAssembleDiagonalCore_Hip(op, assembled, request, false);
  }
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Hip(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr;
  bool isComposite;
  ierr = CeedOperatorIsComposite(op, &isComposite); CeedChkBackend(ierr);
  if (isComposite) {
    return CeedOperatorLinearAssembleAddDiagonalCompositeCore_Hip(op, assembled,
           request, true);
  } else {
    return CeedOperatorAssembleDiagonalCore_Hip(op, assembled, request, true);
  }
}

//------------------------------------------------------------------------------
// Create FDM element inverse not supported
//------------------------------------------------------------------------------
static int CeedOperatorCreateFDMElementInverse_Hip(CeedOperator op) {
  // LCOV_EXCL_START
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  return CeedError(ceed, CEED_ERROR_BACKEND,
                   "Backend does not implement FDM inverse creation");
  // LCOV_EXCL_STOP
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
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal",
                                CeedOperatorLinearAssembleAddDiagonal_Hip);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleAddPointBlockDiagonal",
                                CeedOperatorLinearAssembleAddPointBlockDiagonal_Hip);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "CreateFDMElementInverse",
                                CeedOperatorCreateFDMElementInverse_Hip);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd",
                                CeedOperatorApplyAdd_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Hip); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Composite Operator Create
//------------------------------------------------------------------------------
int CeedCompositeOperatorCreate_Hip(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal",
                                CeedOperatorLinearAssembleAddDiagonal_Hip);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleAddPointBlockDiagonal",
                                CeedOperatorLinearAssembleAddPointBlockDiagonal_Hip);
  CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
