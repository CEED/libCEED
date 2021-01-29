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

#include <ceed.h>
#include <ceed-backend.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <string.h>
#include "ceed-cuda.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Cuda(CeedOperator op) {
  int ierr;
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChk(ierr);

  // Apply data
  for (CeedInt i = 0; i < impl->numein + impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->evecs); CeedChk(ierr);
  ierr = CeedFree(&impl->edata); CeedChk(ierr);

  for (CeedInt i = 0; i < impl->numein; i++) {
    ierr = CeedVectorDestroy(&impl->qvecsin[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->qvecsin); CeedChk(ierr);

  for (CeedInt i = 0; i < impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->qvecsout[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->qvecsout); CeedChk(ierr);

  // Diag data
  if (impl->diag) {
    Ceed ceed;
    ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
    CeedChk_Cu(ceed, cuModuleUnload(impl->diag->module));
    ierr = CeedFree(&impl->diag->h_emodein); CeedChk(ierr);
    ierr = CeedFree(&impl->diag->h_emodeout); CeedChk(ierr);
    ierr = cudaFree(impl->diag->d_emodein); CeedChk_Cu(ceed, ierr);
    ierr = cudaFree(impl->diag->d_emodeout); CeedChk_Cu(ceed, ierr);
    ierr = cudaFree(impl->diag->d_identity); CeedChk_Cu(ceed, ierr);
    ierr = cudaFree(impl->diag->d_interpin); CeedChk_Cu(ceed, ierr);
    ierr = cudaFree(impl->diag->d_interpout); CeedChk_Cu(ceed, ierr);
    ierr = cudaFree(impl->diag->d_gradin); CeedChk_Cu(ceed, ierr);
    ierr = cudaFree(impl->diag->d_gradout); CeedChk_Cu(ceed, ierr);
    ierr = CeedElemRestrictionDestroy(&impl->diag->pbdiagrstr); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->diag); CeedChk(ierr);

  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Setup infields or outfields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Cuda(CeedQFunction qf, CeedOperator op,
                                        bool inOrOut, CeedVector *evecs,
                                        CeedVector *qvecs, CeedInt starte,
                                        CeedInt numfields, CeedInt Q,
                                        CeedInt numelements) {
  CeedInt dim, ierr, size;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedBasis basis;
  CeedElemRestriction Erestrict;
  CeedOperatorField *opfields;
  CeedQFunctionField *qffields;
  CeedVector fieldvec;
  bool strided;
  bool skiprestrict;

  if (inOrOut) {
    ierr = CeedOperatorGetFields(op, NULL, &opfields);
    CeedChk(ierr);
    ierr = CeedQFunctionGetFields(qf, NULL, &qffields);
    CeedChk(ierr);
  } else {
    ierr = CeedOperatorGetFields(op, &opfields, NULL);
    CeedChk(ierr);
    ierr = CeedQFunctionGetFields(qf, &qffields, NULL);
    CeedChk(ierr);
  }

  // Loop over fields
  for (CeedInt i = 0; i < numfields; i++) {
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChk(ierr);

    strided = false;
    skiprestrict = false;
    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &Erestrict);
      CeedChk(ierr);

      // Check whether this field can skip the element restriction:
      // must be passive input, with emode NONE, and have a strided restriction with
      // CEED_STRIDES_BACKEND.

      // First, check whether the field is input or output:
      if (!inOrOut) {
        // Check for passive input:
        ierr = CeedOperatorFieldGetVector(opfields[i], &fieldvec); CeedChk(ierr);
        if (fieldvec != CEED_VECTOR_ACTIVE) {
          // Check emode
          if (emode == CEED_EVAL_NONE) {
            // Check for strided restriction
            ierr = CeedElemRestrictionIsStrided(Erestrict, &strided);
            CeedChk(ierr);
            if (strided) {
              // Check if vector is already in preferred backend ordering
              ierr = CeedElemRestrictionHasBackendStrides(Erestrict,
                     &skiprestrict); CeedChk(ierr);
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
        CeedChk(ierr);
      }
    }

    switch (emode) {
    case CEED_EVAL_NONE:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, numelements * Q * size, &qvecs[i]);
      CeedChk(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, numelements * Q * size, &qvecs[i]);
      CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, numelements * Q * size, &qvecs[i]);
      CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, numelements * Q, &qvecs[i]); CeedChk(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
                            CEED_EVAL_WEIGHT, NULL, qvecs[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
// CeedOperator needs to connect all the named fields (be they active or passive)
//   to the named inputs and outputs of its CeedQFunction.
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Cuda(CeedOperator op) {
  int ierr;
  bool setupdone;
  ierr = CeedOperatorIsSetupDone(op, &setupdone); CeedChk(ierr);
  if (setupdone)
    return 0;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, numelements, numinputfields, numoutputfields;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);

  // Allocate
  ierr = CeedCalloc(numinputfields + numoutputfields, &impl->evecs);
  CeedChk(ierr);
  ierr = CeedCalloc(numinputfields + numoutputfields, &impl->edata);
  CeedChk(ierr);

  ierr = CeedCalloc(16, &impl->qvecsin); CeedChk(ierr);
  ierr = CeedCalloc(16, &impl->qvecsout); CeedChk(ierr);

  impl->numein = numinputfields; impl->numeout = numoutputfields;

  // Set up infield and outfield evecs and qvecs
  // Infields
  ierr = CeedOperatorSetupFields_Cuda(qf, op, 0,
                                      impl->evecs, impl->qvecsin, 0,
                                      numinputfields, Q, numelements);
  CeedChk(ierr);

  // Outfields
  ierr = CeedOperatorSetupFields_Cuda(qf, op, 1,
                                      impl->evecs, impl->qvecsout,
                                      numinputfields, numoutputfields, Q,
                                      numelements); CeedChk(ierr);

  ierr = CeedOperatorSetSetupDone(op); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Cuda(CeedInt numinputfields,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    CeedVector invec, const bool skipactive, CeedOperator_Cuda *impl,
    CeedRequest *request) {
  CeedInt ierr;
  CeedEvalMode emode;
  CeedVector vec;
  CeedElemRestriction Erestrict;

  for (CeedInt i = 0; i < numinputfields; i++) {
    // Get input vector
    ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      if (skipactive)
        continue;
      else
        vec = invec;
    }

    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      // Get input element restriction
      ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
      CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        vec = invec;
      // Restrict, if necessary
      if (!impl->evecs[i]) {
        // No restriction for this field; read data directly from vec.
        ierr = CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE,
                                      (const CeedScalar **) &impl->edata[i]);
        CeedChk(ierr);
      } else {
        ierr = CeedElemRestrictionApply(Erestrict, CEED_NOTRANSPOSE, vec,
                                        impl->evecs[i], request); CeedChk(ierr);
        // Get evec
        ierr = CeedVectorGetArrayRead(impl->evecs[i], CEED_MEM_DEVICE,
                                      (const CeedScalar **) &impl->edata[i]);
        CeedChk(ierr);
      }
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Cuda(CeedInt numelements,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    CeedInt numinputfields, const bool skipactive, CeedOperator_Cuda *impl) {
  CeedInt ierr;
  CeedInt elemsize, size;
  CeedElemRestriction Erestrict;
  CeedEvalMode emode;
  CeedBasis basis;

  for (CeedInt i=0; i<numinputfields; i++) {
    // Skip active input
    if (skipactive) {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        continue;
    }
    // Get elemsize, emode, size
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetSize(qfinputfields[i], &size); CeedChk(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      ierr = CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_DEVICE,
                                CEED_USE_POINTER,
                                impl->edata[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
                            CEED_EVAL_INTERP, impl->evecs[i],
                            impl->qvecsin[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE,
                            CEED_EVAL_GRAD, impl->evecs[i],
                            impl->qvecsin[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT:
      break; // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Cuda(CeedInt numinputfields,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    const bool skipactive, CeedOperator_Cuda *impl) {
  CeedInt ierr;
  CeedEvalMode emode;
  CeedVector vec;

  for (CeedInt i = 0; i < numinputfields; i++) {
    // Skip active input
    if (skipactive) {
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        continue;
    }
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      if (!impl->evecs[i]) {  // This was a skiprestrict case
        ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
        ierr = CeedVectorRestoreArrayRead(vec,
                                          (const CeedScalar **)&impl->edata[i]);
        CeedChk(ierr);
      } else {
        ierr = CeedVectorRestoreArrayRead(impl->evecs[i],
                                          (const CeedScalar **) &impl->edata[i]);
        CeedChk(ierr);
      }
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Cuda(CeedOperator op, CeedVector invec,
                                     CeedVector outvec, CeedRequest *request) {
  int ierr;
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, numelements, elemsize, numinputfields, numoutputfields, size;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);
  CeedEvalMode emode;
  CeedVector vec;
  CeedBasis basis;
  CeedElemRestriction Erestrict;

  // Setup
  ierr = CeedOperatorSetup_Cuda(op); CeedChk(ierr);

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Cuda(numinputfields, qfinputfields,
                                      opinputfields, invec, false, impl,
                                      request); CeedChk(ierr);

  // Input basis apply if needed
  ierr = CeedOperatorInputBasis_Cuda(numelements, qfinputfields, opinputfields,
                                     numinputfields, false, impl);
  CeedChk(ierr);

  // Output pointers, as necessary
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_NONE) {
      // Set the output Q-Vector to use the E-Vector data directly.
      ierr = CeedVectorGetArray(impl->evecs[i + impl->numein], CEED_MEM_DEVICE,
                                &impl->edata[i + numinputfields]); CeedChk(ierr);
      ierr = CeedVectorSetArray(impl->qvecsout[i], CEED_MEM_DEVICE,
                                CEED_USE_POINTER,
                                impl->edata[i + numinputfields]);
      CeedChk(ierr);
    }
  }

  // Q function
  ierr = CeedQFunctionApply(qf, numelements * Q, impl->qvecsin, impl->qvecsout);
  CeedChk(ierr);

  // Output basis apply if needed
  for (CeedInt i = 0; i < numoutputfields; i++) {
    // Get elemsize, emode, size
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &size); CeedChk(ierr);
    // Basis action
    switch (emode) {
    case CEED_EVAL_NONE:
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      CeedChk(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_TRANSPOSE,
                            CEED_EVAL_INTERP, impl->qvecsout[i],
                            impl->evecs[i + impl->numein]); CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      CeedChk(ierr);
      ierr = CeedBasisApply(basis, numelements, CEED_TRANSPOSE,
                            CEED_EVAL_GRAD, impl->qvecsout[i],
                            impl->evecs[i + impl->numein]); CeedChk(ierr);
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
      return CeedError(ceed, 1,
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
    CeedChk(ierr);
    if (emode == CEED_EVAL_NONE) {
      ierr = CeedVectorRestoreArray(impl->evecs[i+impl->numein],
                                    &impl->edata[i + numinputfields]);
      CeedChk(ierr);
    }
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    // Restrict
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChk(ierr);
    // Active
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;

    ierr = CeedElemRestrictionApply(Erestrict, CEED_TRANSPOSE,
                                    impl->evecs[i + impl->numein], vec,
                                    request); CeedChk(ierr);
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Cuda(numinputfields, qfinputfields,
                                        opinputfields, false, impl);
  CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Cuda(CeedOperator op,
    CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  int ierr;
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, numelements, numinputfields, numoutputfields, size;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);
  CeedVector vec;
  CeedInt numactivein = 0, numactiveout = 0;
  CeedVector *activein = NULL;
  CeedScalar *a, *tmp;
  Ceed ceed, ceedparent;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  ierr = CeedGetOperatorFallbackParentCeed(ceed, &ceedparent); CeedChk(ierr);
  ceedparent = ceedparent ? ceedparent : ceed;

  // Setup
  ierr = CeedOperatorSetup_Cuda(op); CeedChk(ierr);

  // Check for identity
  bool identityqf;
  ierr = CeedQFunctionIsIdentity(qf, &identityqf); CeedChk(ierr);
  if (identityqf)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Assembling identity QFunctions not supported");
  // LCOV_EXCL_STOP

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Cuda(numinputfields, qfinputfields,
                                      opinputfields, NULL, true, impl, request);
  CeedChk(ierr);

  // Count number of active input fields
  for (CeedInt i=0; i<numinputfields; i++) {
    // Get input vector
    ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
    // Check if active input
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedQFunctionFieldGetSize(qfinputfields[i], &size); CeedChk(ierr);
      ierr = CeedVectorSetValue(impl->qvecsin[i], 0.0); CeedChk(ierr);
      ierr = CeedVectorGetArray(impl->qvecsin[i], CEED_MEM_DEVICE, &tmp);
      CeedChk(ierr);
      ierr = CeedRealloc(numactivein + size, &activein); CeedChk(ierr);
      for (CeedInt field = 0; field < size; field++) {
        ierr = CeedVectorCreate(ceed, Q*numelements,
                                &activein[numactivein+field]); CeedChk(ierr);
        ierr = CeedVectorSetArray(activein[numactivein+field], CEED_MEM_DEVICE,
                                  CEED_USE_POINTER, &tmp[field*Q*numelements]);
        CeedChk(ierr);
      }
      numactivein += size;
      ierr = CeedVectorRestoreArray(impl->qvecsin[i], &tmp); CeedChk(ierr);
    }
  }

  // Count number of active output fields
  for (CeedInt i=0; i<numoutputfields; i++) {
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &size); CeedChk(ierr);
      numactiveout += size;
    }
  }

  // Check sizes
  if (!numactivein || !numactiveout)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Cannot assemble QFunction without active inputs "
                     "and outputs");
  // LCOV_EXCL_STOP

  // Create output restriction
  CeedInt strides[3] = {1, numelements*Q, Q}; /* *NOPAD* */
  ierr = CeedElemRestrictionCreateStrided(ceedparent, numelements, Q,
                                          numactivein*numactiveout,
                                          numactivein*numactiveout*numelements*Q,
                                          strides, rstr); CeedChk(ierr);
  // Create assembled vector
  ierr = CeedVectorCreate(ceedparent, numelements*Q*numactivein*numactiveout,
                          assembled); CeedChk(ierr);
  ierr = CeedVectorSetValue(*assembled, 0.0); CeedChk(ierr);
  ierr = CeedVectorGetArray(*assembled, CEED_MEM_DEVICE, &a); CeedChk(ierr);

  // Input basis apply
  ierr = CeedOperatorInputBasis_Cuda(numelements, qfinputfields, opinputfields,
                                     numinputfields, true, impl);
  CeedChk(ierr);

  // Assemble QFunction
  for (CeedInt in=0; in<numactivein; in++) {
    // Set Inputs
    ierr = CeedVectorSetValue(activein[in], 1.0); CeedChk(ierr);
    if (numactivein > 1) {
      ierr = CeedVectorSetValue(activein[(in+numactivein-1)%numactivein],
                                0.0); CeedChk(ierr);
    }
    // Set Outputs
    for (CeedInt out=0; out<numoutputfields; out++) {
      // Get output vector
      ierr = CeedOperatorFieldGetVector(opoutputfields[out], &vec);
      CeedChk(ierr);
      // Check if active output
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedVectorSetArray(impl->qvecsout[out], CEED_MEM_DEVICE,
                           CEED_USE_POINTER, a); CeedChk(ierr);
        ierr = CeedQFunctionFieldGetSize(qfoutputfields[out], &size);
        CeedChk(ierr);
        a += size*Q*numelements; // Advance the pointer by the size of the output
      }
    }
    // Apply QFunction
    ierr = CeedQFunctionApply(qf, Q*numelements, impl->qvecsin, impl->qvecsout);
    CeedChk(ierr);
  }


  // Un-set output Qvecs to prevent accidental overwrite of Assembled
  for (CeedInt out=0; out<numoutputfields; out++) {
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[out], &vec);
    CeedChk(ierr);
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedVectorTakeArray(impl->qvecsout[out], CEED_MEM_DEVICE, NULL);
      CeedChk(ierr);
    }
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Cuda(numinputfields, qfinputfields,
                                        opinputfields, true, impl);
  CeedChk(ierr);

  // Restore output
  ierr = CeedVectorRestoreArray(*assembled, &a); CeedChk(ierr);

  // Cleanup
  for (CeedInt i=0; i<numactivein; i++) {
    ierr = CeedVectorDestroy(&activein[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&activein); CeedChk(ierr);

  return 0;
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
extern "C" __device__ void CeedOperatorGetBasisPointer_Cuda(const CeedScalar **basisptr,
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
      CeedOperatorGetBasisPointer_Cuda(&bt, emodeout[eout], identity, interpout,
                                      &gradout[dout*NQPTS*NNODES]);
      CeedInt din = -1;
      for (CeedInt ein = 0; ein < NUMEMODEIN; ein++) {
        const CeedScalar *b = NULL;
        if (emodein[ein] == CEED_EVAL_GRAD)
          din += 1;
        CeedOperatorGetBasisPointer_Cuda(&b, emodein[ein], identity, interpin,
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
  ierr = CeedElemRestrictionGetCeed(rstr, &ceed); CeedChk(ierr);
  const CeedInt *offsets;
  ierr = CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets);
  CeedChk(ierr);

  // Expand offsets
  CeedInt nelem, ncomp, elemsize, compstride, max = 1, *pbOffsets;
  ierr = CeedElemRestrictionGetNumElements(rstr, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstr, &ncomp); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstr, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetCompStride(rstr, &compstride); CeedChk(ierr);
  CeedInt shift = ncomp;
  if (compstride != 1)
    shift *= ncomp;
  ierr = CeedCalloc(nelem*elemsize, &pbOffsets); CeedChk(ierr);
  for (CeedInt i = 0; i < nelem*elemsize; i++) {
    pbOffsets[i] = offsets[i]*shift;
    if (pbOffsets[i] > max)
      max = pbOffsets[i];
  }

  // Create new restriction
  ierr = CeedElemRestrictionCreate(ceed, nelem, elemsize, ncomp*ncomp, 1,
                                   max + ncomp*ncomp, CEED_MEM_HOST,
                                   CEED_OWN_POINTER, pbOffsets, pbRstr);
  CeedChk(ierr);

  // Cleanup
  ierr = CeedElemRestrictionRestoreOffsets(rstr, &offsets); CeedChk(ierr);

  return 0;
}

//------------------------------------------------------------------------------
// Assemble diagonal setup
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalSetup_Cuda(CeedOperator op,
    const bool pointBlock) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);

  // Determine active input basis
  CeedOperatorField *opfields;
  CeedQFunctionField *qffields;
  ierr = CeedOperatorGetFields(op, &opfields, NULL); CeedChk(ierr);
  ierr = CeedQFunctionGetFields(qf, &qffields, NULL); CeedChk(ierr);
  CeedInt numemodein = 0, ncomp = 0, dim = 1;
  CeedEvalMode *emodein = NULL;
  CeedBasis basisin = NULL;
  CeedElemRestriction rstrin = NULL;
  for (CeedInt i = 0; i < numinputfields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(opfields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedElemRestriction rstr;
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basisin); CeedChk(ierr);
      ierr = CeedBasisGetNumComponents(basisin, &ncomp); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basisin, &dim); CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &rstr);
      CeedChk(ierr);
      if (rstrin && rstrin != rstr)
        // LCOV_EXCL_START
        return CeedError(ceed, 1,
                         "Multi-field non-composite operator diagonal assembly not supported");
      // LCOV_EXCL_STOP
      rstrin = rstr;
      CeedEvalMode emode;
      ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
      CeedChk(ierr);
      switch (emode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(numemodein + 1, &emodein); CeedChk(ierr);
        emodein[numemodein] = emode;
        numemodein += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(numemodein + dim, &emodein); CeedChk(ierr);
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
  ierr = CeedOperatorGetFields(op, NULL, &opfields); CeedChk(ierr);
  ierr = CeedQFunctionGetFields(qf, NULL, &qffields); CeedChk(ierr);
  CeedInt numemodeout = 0;
  CeedEvalMode *emodeout = NULL;
  CeedBasis basisout = NULL;
  CeedElemRestriction rstrout = NULL;
  for (CeedInt i = 0; i < numoutputfields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(opfields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedElemRestriction rstr;
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basisout); CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &rstr);
      CeedChk(ierr);
      if (rstrout && rstrout != rstr)
        // LCOV_EXCL_START
        return CeedError(ceed, 1,
                         "Multi-field non-composite operator diagonal assembly not supported");
      // LCOV_EXCL_STOP
      rstrout = rstr;
      CeedEvalMode emode;
      ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChk(ierr);
      switch (emode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(numemodeout + 1, &emodeout); CeedChk(ierr);
        emodeout[numemodeout] = emode;
        numemodeout += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(numemodeout + dim, &emodeout); CeedChk(ierr);
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
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChk(ierr);
  ierr = CeedCalloc(1, &impl->diag); CeedChk(ierr);
  CeedOperatorDiag_Cuda *diag = impl->diag;
  diag->basisin = basisin;
  diag->basisout = basisout;
  diag->h_emodein = emodein;
  diag->h_emodeout = emodeout;
  diag->numemodein = numemodein;
  diag->numemodeout = numemodeout;

  // Assemble kernel
  CeedInt nnodes, nqpts;
  ierr = CeedBasisGetNumNodes(basisin, &nnodes); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basisin, &nqpts); CeedChk(ierr);
  diag->nnodes = nnodes;
  ierr = CeedCompileCuda(ceed, diagonalkernels, &diag->module, 5,
                         "NUMEMODEIN", numemodein,
                         "NUMEMODEOUT", numemodeout,
                         "NNODES", nnodes,
                         "NQPTS", nqpts,
                         "NCOMP", ncomp
                        ); CeedChk_Cu(ceed, ierr);
  ierr = CeedGetKernelCuda(ceed, diag->module, "linearDiagonal",
                           &diag->linearDiagonal); CeedChk_Cu(ceed, ierr);
  ierr = CeedGetKernelCuda(ceed, diag->module, "linearPointBlockDiagonal",
                           &diag->linearPointBlock);
  CeedChk_Cu(ceed, ierr);

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
    ierr = CeedCalloc(nqpts*nnodes, &identity); CeedChk(ierr);
    for (CeedInt i=0; i<(nnodes<nqpts?nnodes:nqpts); i++)
      identity[i*nnodes+i] = 1.0;
    ierr = cudaMalloc((void **)&diag->d_identity, iBytes); CeedChk_Cu(ceed, ierr);
    ierr = cudaMemcpy(diag->d_identity, identity, iBytes,
                      cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
  }

  // CEED_EVAL_INTERP
  ierr = CeedBasisGetInterp(basisin, &interpin); CeedChk(ierr);
  ierr = cudaMalloc((void **)&diag->d_interpin, iBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(diag->d_interpin, interpin, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
  ierr = CeedBasisGetInterp(basisout, &interpout); CeedChk(ierr);
  ierr = cudaMalloc((void **)&diag->d_interpout, iBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(diag->d_interpout, interpout, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  // CEED_EVAL_GRAD
  ierr = CeedBasisGetGrad(basisin, &gradin); CeedChk(ierr);
  ierr = cudaMalloc((void **)&diag->d_gradin, gBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(diag->d_gradin, gradin, gBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
  ierr = CeedBasisGetGrad(basisout, &gradout); CeedChk(ierr);
  ierr = cudaMalloc((void **)&diag->d_gradout, gBytes); CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(diag->d_gradout, gradout, gBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  // Arrays of emodes
  ierr = cudaMalloc((void **)&diag->d_emodein, numemodein * eBytes);
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(diag->d_emodein, emodein, numemodein * eBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
  ierr = cudaMalloc((void **)&diag->d_emodeout, numemodeout * eBytes);
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(diag->d_emodeout, emodeout, numemodeout * eBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  // Restriction
  diag->diagrstr = rstrout;

  return 0;
}

//------------------------------------------------------------------------------
// Assemble diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalCore_Cuda(CeedOperator op,
    CeedVector assembled, CeedRequest *request, const bool pointBlock) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChk(ierr);

  // Assemble QFunction
  CeedVector assembledqf;
  CeedElemRestriction rstr;
  ierr = CeedOperatorLinearAssembleQFunction(op,  &assembledqf, &rstr, request);
  CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr); CeedChk(ierr);
  CeedScalar maxnorm = 0;
  ierr = CeedVectorNorm(assembledqf, CEED_NORM_MAX, &maxnorm); CeedChk(ierr);

  // Setup
  if (!impl->diag) {
    ierr = CeedOperatorAssembleDiagonalSetup_Cuda(op, pointBlock); CeedChk(ierr);
  }
  CeedOperatorDiag_Cuda *diag = impl->diag;
  assert(diag != NULL);

  // Restriction
  if (pointBlock && !diag->pbdiagrstr) {
    CeedElemRestriction pbdiagrstr;
    ierr = CreatePBRestriction(diag->diagrstr, &pbdiagrstr); CeedChk(ierr);
    diag->pbdiagrstr = pbdiagrstr;
  }
  CeedElemRestriction diagrstr = pointBlock ? diag->pbdiagrstr : diag->diagrstr;

  // Create diagonal vector
  CeedVector elemdiag;
  ierr = CeedElemRestrictionCreateVector(diagrstr, NULL, &elemdiag);
  CeedChk(ierr);
  ierr = CeedVectorSetValue(elemdiag, 0.0); CeedChk(ierr);

  // Assemble element operator diagonals
  CeedScalar *elemdiagarray, *assembledqfarray;
  ierr = CeedVectorGetArray(elemdiag, CEED_MEM_DEVICE, &elemdiagarray);
  CeedChk(ierr);
  ierr = CeedVectorGetArray(assembledqf, CEED_MEM_DEVICE, &assembledqfarray);
  CeedChk(ierr);
  CeedInt nelem;
  ierr = CeedElemRestrictionGetNumElements(diagrstr, &nelem); CeedChk(ierr);

  // Compute the diagonal of B^T D B
  int elemsPerBlock = 1;
  int grid = nelem/elemsPerBlock+((nelem/elemsPerBlock*elemsPerBlock<nelem)?1:0);
  void *args[] = {(void *) &nelem, (void *) &maxnorm, &diag->d_identity,
                  &diag->d_interpin, &diag->d_gradin, &diag->d_interpout,
                  &diag->d_gradout, &diag->d_emodein, &diag->d_emodeout,
                  &assembledqfarray, &elemdiagarray
                 };
  if (pointBlock) {
    ierr = CeedRunKernelDimCuda(ceed, diag->linearPointBlock, grid,
                                diag->nnodes, 1, elemsPerBlock, args);
    CeedChk(ierr);
  } else {
    ierr = CeedRunKernelDimCuda(ceed, diag->linearDiagonal, grid,
                                diag->nnodes, 1, elemsPerBlock, args);
    CeedChk(ierr);
  }

  // Restore arrays
  ierr = CeedVectorRestoreArray(elemdiag, &elemdiagarray); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(assembledqf, &assembledqfarray); CeedChk(ierr);

  // Assemble local operator diagonal
  ierr = CeedElemRestrictionApply(diagrstr, CEED_TRANSPOSE, elemdiag,
                                  assembled, request); CeedChk(ierr);

  // Cleanup
  ierr = CeedVectorDestroy(&assembledqf); CeedChk(ierr);
  ierr = CeedVectorDestroy(&elemdiag); CeedChk(ierr);

  return 0;
}

//------------------------------------------------------------------------------
// Assemble composite diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleAddDiagonalCompositeCore_Cuda(
  CeedOperator op, CeedVector assembled, CeedRequest *request,
  const bool pointBlock) {
  int ierr;
  CeedInt numSub;
  CeedOperator *subOperators;
  ierr = CeedOperatorGetNumSub(op, &numSub); CeedChk(ierr);
  ierr = CeedOperatorGetSubList(op, &subOperators); CeedChk(ierr);
  for (CeedInt i = 0; i < numSub; i++) {
    ierr = CeedOperatorAssembleDiagonalCore_Cuda(subOperators[i], assembled,
           request, pointBlock); CeedChk(ierr);
  }
  return 0;
}

//------------------------------------------------------------------------------
// Assemble Linear Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonal_Cuda(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr;
  bool isComposite;
  ierr = CeedOperatorIsComposite(op, &isComposite); CeedChk(ierr);
  if (isComposite) {
    return CeedOperatorLinearAssembleAddDiagonalCompositeCore_Cuda(op, assembled,
           request, false);
  } else {
    return CeedOperatorAssembleDiagonalCore_Cuda(op, assembled, request, false);
  }
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Cuda(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr;
  bool isComposite;
  ierr = CeedOperatorIsComposite(op, &isComposite); CeedChk(ierr);
  if (isComposite) {
    return CeedOperatorLinearAssembleAddDiagonalCompositeCore_Cuda(op, assembled,
           request, true);
  } else {
    return CeedOperatorAssembleDiagonalCore_Cuda(op, assembled, request, true);
  }
}

//------------------------------------------------------------------------------
// Create FDM element inverse not supported
//------------------------------------------------------------------------------
static int CeedOperatorCreateFDMElementInverse_Cuda(CeedOperator op) {
  // LCOV_EXCL_START
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement FDM inverse creation");
  // LCOV_EXCL_STOP
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Cuda(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda *impl;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedOperatorSetData(op, impl); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction",
                                CeedOperatorLinearAssembleQFunction_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal",
                                CeedOperatorLinearAssembleAddDiagonal_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleAddPointBlockDiagonal",
                                CeedOperatorLinearAssembleAddPointBlockDiagonal_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "CreateFDMElementInverse",
                                CeedOperatorCreateFDMElementInverse_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd",
                                CeedOperatorApplyAdd_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Cuda); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Composite Operator Create
//------------------------------------------------------------------------------
int CeedCompositeOperatorCreate_Cuda(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal",
                                CeedOperatorLinearAssembleAddDiagonal_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleAddPointBlockDiagonal",
                                CeedOperatorLinearAssembleAddPointBlockDiagonal_Cuda);
  CeedChk(ierr);
  return 0;
}
//------------------------------------------------------------------------------
