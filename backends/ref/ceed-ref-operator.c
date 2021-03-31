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
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Ref(CeedQFunction qf, CeedOperator op,
                                       bool inOrOut,
                                       CeedVector *fullevecs, CeedVector *evecs,
                                       CeedVector *qvecs, CeedInt starte,
                                       CeedInt numfields, CeedInt Q) {
  CeedInt dim, ierr, size, P;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedBasis basis;
  CeedElemRestriction Erestrict;
  CeedOperatorField *opfields;
  CeedQFunctionField *qffields;
  if (inOrOut) {
    ierr = CeedOperatorGetFields(op, NULL, &opfields); CeedChkBackend(ierr);
    ierr = CeedQFunctionGetFields(qf, NULL, &qffields); CeedChkBackend(ierr);
  } else {
    ierr = CeedOperatorGetFields(op, &opfields, NULL); CeedChkBackend(ierr);
    ierr = CeedQFunctionGetFields(qf, &qffields, NULL); CeedChkBackend(ierr);
  }

  // Loop over fields
  for (CeedInt i=0; i<numfields; i++) {
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChkBackend(ierr);

    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &Erestrict);
      CeedChkBackend(ierr);
      ierr = CeedElemRestrictionCreateVector(Erestrict, NULL,
                                             &fullevecs[i+starte]);
      CeedChkBackend(ierr);
    }

    switch(emode) {
    case CEED_EVAL_NONE:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size, &qvecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetElementSize(Erestrict, &P);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, P*size, &evecs[i]); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size, &qvecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetElementSize(Erestrict, &P);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, P*size/dim, &evecs[i]); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size, &qvecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q, &qvecs[i]); CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                            CEED_VECTOR_NONE, qvecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_DIV:
      break; // Not implemented
    case CEED_EVAL_CURL:
      break; // Not implemented
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator
//------------------------------------------------------------------------------/*
static int CeedOperatorSetup_Ref(CeedOperator op) {
  int ierr;
  bool setupdone;
  ierr = CeedOperatorIsSetupDone(op, &setupdone); CeedChkBackend(ierr);
  if (setupdone) return CEED_ERROR_SUCCESS;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Ref *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt Q, numinputfields, numoutputfields;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  ierr = CeedQFunctionIsIdentity(qf, &impl->identityqf); CeedChkBackend(ierr);
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

  ierr = CeedCalloc(16, &impl->inputstate); CeedChkBackend(ierr);
  ierr = CeedCalloc(16, &impl->evecsin); CeedChkBackend(ierr);
  ierr = CeedCalloc(16, &impl->evecsout); CeedChkBackend(ierr);
  ierr = CeedCalloc(16, &impl->qvecsin); CeedChkBackend(ierr);
  ierr = CeedCalloc(16, &impl->qvecsout); CeedChkBackend(ierr);

  impl->numein = numinputfields; impl->numeout = numoutputfields;

  // Set up infield and outfield evecs and qvecs
  // Infields
  ierr = CeedOperatorSetupFields_Ref(qf, op, 0, impl->evecs,
                                     impl->evecsin, impl->qvecsin, 0,
                                     numinputfields, Q);
  CeedChkBackend(ierr);
  // Outfields
  ierr = CeedOperatorSetupFields_Ref(qf, op, 1, impl->evecs,
                                     impl->evecsout, impl->qvecsout,
                                     numinputfields, numoutputfields, Q);
  CeedChkBackend(ierr);

  // Identity QFunctions
  if (impl->identityqf) {
    CeedEvalMode inmode, outmode;
    CeedQFunctionField *infields, *outfields;
    ierr = CeedQFunctionGetFields(qf, &infields, &outfields); CeedChkBackend(ierr);

    for (CeedInt i=0; i<numinputfields; i++) {
      ierr = CeedQFunctionFieldGetEvalMode(infields[i], &inmode);
      CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(outfields[i], &outmode);
      CeedChkBackend(ierr);

      ierr = CeedVectorDestroy(&impl->qvecsout[i]); CeedChkBackend(ierr);
      impl->qvecsout[i] = impl->qvecsin[i];
      ierr = CeedVectorAddReference(impl->qvecsin[i]); CeedChkBackend(ierr);
    }
  }

  ierr = CeedOperatorSetSetupDone(op); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Ref(CeedInt numinputfields,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    CeedVector invec, const bool skipactive, CeedOperator_Ref *impl,
    CeedRequest *request) {
  CeedInt ierr;
  CeedEvalMode emode;
  CeedVector vec;
  CeedElemRestriction Erestrict;
  uint64_t state;

  for (CeedInt i=0; i<numinputfields; i++) {
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
    // Restrict and Evec
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Restrict
      ierr = CeedVectorGetState(vec, &state); CeedChkBackend(ierr);
      // Skip restriction if input is unchanged
      if (state != impl->inputstate[i] || vec == invec) {
        ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
        CeedChkBackend(ierr);
        ierr = CeedElemRestrictionApply(Erestrict, CEED_NOTRANSPOSE, vec,
                                        impl->evecs[i], request); CeedChkBackend(ierr);
        impl->inputstate[i] = state;
      }
      // Get evec
      ierr = CeedVectorGetArrayRead(impl->evecs[i], CEED_MEM_HOST,
                                    (const CeedScalar **) &impl->edata[i]);
      CeedChkBackend(ierr);
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Ref(CeedInt e, CeedInt Q,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    CeedInt numinputfields, const bool skipactive, CeedOperator_Ref *impl) {
  CeedInt ierr;
  CeedInt dim, elemsize, size;
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
    switch(emode) {
    case CEED_EVAL_NONE:
      ierr = CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                &impl->edata[i][e*Q*size]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->evecsin[i], CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                &impl->edata[i][e*elemsize*size]);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE,
                            CEED_EVAL_INTERP, impl->evecsin[i],
                            impl->qvecsin[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->evecsin[i], CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                &impl->edata[i][e*elemsize*size/dim]);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE,
                            CEED_EVAL_GRAD, impl->evecsin[i],
                            impl->qvecsin[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_WEIGHT:
      break;  // No action
    // LCOV_EXCL_START
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL: {
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis);
      CeedChkBackend(ierr);
      Ceed ceed;
      ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Ceed evaluation mode not implemented");
      // LCOV_EXCL_STOP
    }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Output Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorOutputBasis_Ref(CeedInt e, CeedInt Q,
    CeedQFunctionField *qfoutputfields, CeedOperatorField *opoutputfields,
    CeedInt numinputfields, CeedInt numoutputfields, CeedOperator op,
    CeedOperator_Ref *impl) {
  CeedInt ierr;
  CeedInt dim, elemsize, size;
  CeedElemRestriction Erestrict;
  CeedEvalMode emode;
  CeedBasis basis;

  for (CeedInt i=0; i<numoutputfields; i++) {
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
    switch(emode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->evecsout[i], CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                &impl->edata[i + numinputfields][e*elemsize*size]);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_TRANSPOSE,
                            CEED_EVAL_INTERP, impl->qvecsout[i],
                            impl->evecsout[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->evecsout[i], CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                &impl->edata[i + numinputfields][e*elemsize*size/dim]);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_TRANSPOSE,
                            CEED_EVAL_GRAD, impl->qvecsout[i],
                            impl->evecsout[i]); CeedChkBackend(ierr);
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_WEIGHT cannot be an output "
                       "evaluation mode");
    }
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Ceed evaluation mode not implemented");
      // LCOV_EXCL_STOP
    }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Ref(CeedInt numinputfields,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    const bool skipactive, CeedOperator_Ref *impl) {
  CeedInt ierr;
  CeedEvalMode emode;

  for (CeedInt i=0; i<numinputfields; i++) {
    // Skip active inputs
    if (skipactive) {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        continue;
    }
    // Restore input
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedVectorRestoreArrayRead(impl->evecs[i],
                                        (const CeedScalar **) &impl->edata[i]);
      CeedChkBackend(ierr);
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Apply
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Ref(CeedOperator op, CeedVector invec,
                                    CeedVector outvec, CeedRequest *request) {
  int ierr;
  CeedOperator_Ref *impl;
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
  CeedEvalMode emode;
  CeedVector vec;
  CeedElemRestriction Erestrict;

  // Setup
  ierr = CeedOperatorSetup_Ref(op); CeedChkBackend(ierr);

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Ref(numinputfields, qfinputfields,
                                     opinputfields, invec, false, impl,
                                     request); CeedChkBackend(ierr);

  // Output Evecs
  for (CeedInt i=0; i<numoutputfields; i++) {
    ierr = CeedVectorGetArray(impl->evecs[i+impl->numein], CEED_MEM_HOST,
                              &impl->edata[i + numinputfields]); CeedChkBackend(ierr);
  }

  // Loop through elements
  for (CeedInt e=0; e<numelements; e++) {
    // Output pointers
    for (CeedInt i=0; i<numoutputfields; i++) {
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChkBackend(ierr);
      if (emode == CEED_EVAL_NONE) {
        ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &size);
        CeedChkBackend(ierr);
        ierr = CeedVectorSetArray(impl->qvecsout[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &impl->edata[i + numinputfields][e*Q*size]);
        CeedChkBackend(ierr);
      }
    }

    // Input basis apply
    ierr = CeedOperatorInputBasis_Ref(e, Q, qfinputfields, opinputfields,
                                      numinputfields, false, impl);
    CeedChkBackend(ierr);

    // Q function
    if (!impl->identityqf) {
      ierr = CeedQFunctionApply(qf, Q, impl->qvecsin, impl->qvecsout);
      CeedChkBackend(ierr);
    }

    // Output basis apply
    ierr = CeedOperatorOutputBasis_Ref(e, Q, qfoutputfields, opoutputfields,
                                       numinputfields, numoutputfields, op, impl);
    CeedChkBackend(ierr);
  }

  // Output restriction
  for (CeedInt i=0; i<numoutputfields; i++) {
    // Restore evec
    ierr = CeedVectorRestoreArray(impl->evecs[i+impl->numein],
                                  &impl->edata[i + numinputfields]);
    CeedChkBackend(ierr);
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec);
    CeedChkBackend(ierr);
    // Active
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;
    // Restrict
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionApply(Erestrict, CEED_TRANSPOSE,
                                    impl->evecs[i+impl->numein], vec, request);
    CeedChkBackend(ierr);
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Ref(numinputfields, qfinputfields,
                                       opinputfields, false, impl);
  CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Ref(CeedOperator op,
    CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  int ierr;
  CeedOperator_Ref *impl;
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
  ierr = CeedOperatorSetup_Ref(op); CeedChkBackend(ierr);

  // Check for identity
  if (impl->identityqf)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Assembling identity QFunctions not supported");
  // LCOV_EXCL_STOP

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Ref(numinputfields, qfinputfields,
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
      ierr = CeedVectorGetArray(impl->qvecsin[i], CEED_MEM_HOST, &tmp);
      CeedChkBackend(ierr);
      ierr = CeedRealloc(numactivein + size, &activein); CeedChkBackend(ierr);
      for (CeedInt field=0; field<size; field++) {
        ierr = CeedVectorCreate(ceed, Q, &activein[numactivein+field]);
        CeedChkBackend(ierr);
        ierr = CeedVectorSetArray(activein[numactivein+field], CEED_MEM_HOST,
                                  CEED_USE_POINTER, &tmp[field*Q]);
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
  CeedInt strides[3] = {1, Q, numactivein*numactiveout*Q}; /* *NOPAD* */
  ierr = CeedElemRestrictionCreateStrided(ceedparent, numelements, Q,
                                          numactivein*numactiveout,
                                          numactivein*numactiveout*numelements*Q,
                                          strides, rstr); CeedChkBackend(ierr);
  // Create assembled vector
  ierr = CeedVectorCreate(ceedparent, numelements*Q*numactivein*numactiveout,
                          assembled); CeedChkBackend(ierr);
  ierr = CeedVectorSetValue(*assembled, 0.0); CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(*assembled, CEED_MEM_HOST, &a); CeedChkBackend(ierr);

  // Loop through elements
  for (CeedInt e=0; e<numelements; e++) {
    // Input basis apply
    ierr = CeedOperatorInputBasis_Ref(e, Q, qfinputfields, opinputfields,
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
          CeedVectorSetArray(impl->qvecsout[out], CEED_MEM_HOST,
                             CEED_USE_POINTER, a); CeedChkBackend(ierr);
          ierr = CeedQFunctionFieldGetSize(qfoutputfields[out], &size);
          CeedChkBackend(ierr);
          a += size*Q; // Advance the pointer by the size of the output
        }
      }
      // Apply QFunction
      ierr = CeedQFunctionApply(qf, Q, impl->qvecsin, impl->qvecsout);
      CeedChkBackend(ierr);
    }
  }

  // Un-set output Qvecs to prevent accidental overwrite of Assembled
  for (CeedInt out=0; out<numoutputfields; out++) {
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[out], &vec);
    CeedChkBackend(ierr);
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedVectorTakeArray(impl->qvecsout[out], CEED_MEM_HOST, NULL);
      CeedChkBackend(ierr);
    }
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Ref(numinputfields, qfinputfields,
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
// Get Basis Emode Pointer
//------------------------------------------------------------------------------
static inline void CeedOperatorGetBasisPointer_Ref(const CeedScalar **basisptr,
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
// Create point block restriction
//------------------------------------------------------------------------------
static int CreatePBRestriction_Ref(CeedElemRestriction rstr,
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
// Assemble diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleAddDiagonalCore_Ref(CeedOperator op,
    CeedVector assembled, CeedRequest *request, const bool pointBlock) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);

  // Assemble QFunction
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt numinputfields, numoutputfields;
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);
  CeedVector assembledqf;
  CeedElemRestriction rstr;
  ierr = CeedOperatorLinearAssembleQFunction(op,  &assembledqf, &rstr, request);
  CeedChkBackend(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr); CeedChkBackend(ierr);
  CeedScalar maxnorm = 0;
  ierr = CeedVectorNorm(assembledqf, CEED_NORM_MAX, &maxnorm);
  CeedChkBackend(ierr);

  // Determine active input basis
  CeedOperatorField *opfields;
  CeedQFunctionField *qffields;
  ierr = CeedOperatorGetFields(op, &opfields, NULL); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetFields(qf, &qffields, NULL); CeedChkBackend(ierr);
  CeedInt numemodein = 0, ncomp, dim = 1;
  CeedEvalMode *emodein = NULL;
  CeedBasis basisin = NULL;
  CeedElemRestriction rstrin = NULL;
  for (CeedInt i=0; i<numinputfields; i++) {
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
        for (CeedInt d=0; d<dim; d++)
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
  for (CeedInt i=0; i<numoutputfields; i++) {
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
        for (CeedInt d=0; d<dim; d++)
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

  // Assemble point-block diagonal restriction, if needed
  CeedElemRestriction diagrstr = rstrout;
  if (pointBlock) {
    ierr = CreatePBRestriction_Ref(rstrout, &diagrstr); CeedChkBackend(ierr);
  }

  // Create diagonal vector
  CeedVector elemdiag;
  ierr = CeedElemRestrictionCreateVector(diagrstr, NULL, &elemdiag);
  CeedChkBackend(ierr);

  // Assemble element operator diagonals
  CeedScalar *elemdiagarray, *assembledqfarray;
  ierr = CeedVectorSetValue(elemdiag, 0.0); CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(elemdiag, CEED_MEM_HOST, &elemdiagarray);
  CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(assembledqf, CEED_MEM_HOST, &assembledqfarray);
  CeedChkBackend(ierr);
  CeedInt nelem, nnodes, nqpts;
  ierr = CeedElemRestrictionGetNumElements(diagrstr, &nelem);
  CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basisin, &nnodes); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basisin, &nqpts); CeedChkBackend(ierr);
  // Basis matrices
  const CeedScalar *interpin, *interpout, *gradin, *gradout;
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
  }
  ierr = CeedBasisGetInterp(basisin, &interpin); CeedChkBackend(ierr);
  ierr = CeedBasisGetInterp(basisout, &interpout); CeedChkBackend(ierr);
  ierr = CeedBasisGetGrad(basisin, &gradin); CeedChkBackend(ierr);
  ierr = CeedBasisGetGrad(basisout, &gradout); CeedChkBackend(ierr);
  // Compute the diagonal of B^T D B
  // Each element
  const CeedScalar qfvaluebound = maxnorm*1e-12;
  for (CeedInt e=0; e<nelem; e++) {
    CeedInt dout = -1;
    // Each basis eval mode pair
    for (CeedInt eout=0; eout<numemodeout; eout++) {
      const CeedScalar *bt = NULL;
      if (emodeout[eout] == CEED_EVAL_GRAD)
        dout += 1;
      CeedOperatorGetBasisPointer_Ref(&bt, emodeout[eout], identity, interpout,
                                      &gradout[dout*nqpts*nnodes]);
      CeedInt din = -1;
      for (CeedInt ein=0; ein<numemodein; ein++) {
        const CeedScalar *b = NULL;
        if (emodein[ein] == CEED_EVAL_GRAD)
          din += 1;
        CeedOperatorGetBasisPointer_Ref(&b, emodein[ein], identity, interpin,
                                        &gradin[din*nqpts*nnodes]);
        // Each component
        for (CeedInt compOut=0; compOut<ncomp; compOut++)
          // Each qpoint/node pair
          for (CeedInt q=0; q<nqpts; q++)
            if (pointBlock) {
              // Point Block Diagonal
              for (CeedInt compIn=0; compIn<ncomp; compIn++) {
                const CeedScalar qfvalue =
                  assembledqfarray[((((e*numemodein+ein)*ncomp+compIn)*
                                     numemodeout+eout)*ncomp+compOut)*nqpts+q];
                if (fabs(qfvalue) > qfvaluebound)
                  for (CeedInt n=0; n<nnodes; n++)
                    elemdiagarray[((e*ncomp+compOut)*ncomp+compIn)*nnodes+n] +=
                      bt[q*nnodes+n] * qfvalue * b[q*nnodes+n];
              }
            } else {
              // Diagonal Only
              const CeedScalar qfvalue =
                assembledqfarray[((((e*numemodein+ein)*ncomp+compOut)*
                                   numemodeout+eout)*ncomp+compOut)*nqpts+q];
              if (fabs(qfvalue) > qfvaluebound)
                for (CeedInt n=0; n<nnodes; n++)
                  elemdiagarray[(e*ncomp+compOut)*nnodes+n] +=
                    bt[q*nnodes+n] * qfvalue * b[q*nnodes+n];
            }
      }
    }
  }
  ierr = CeedVectorRestoreArray(elemdiag, &elemdiagarray); CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArray(assembledqf, &assembledqfarray);
  CeedChkBackend(ierr);

  // Assemble local operator diagonal
  ierr = CeedElemRestrictionApply(diagrstr, CEED_TRANSPOSE, elemdiag,
                                  assembled, request); CeedChkBackend(ierr);

  // Cleanup
  if (pointBlock) {
    ierr = CeedElemRestrictionDestroy(&diagrstr); CeedChkBackend(ierr);
  }
  ierr = CeedVectorDestroy(&assembledqf); CeedChkBackend(ierr);
  ierr = CeedVectorDestroy(&elemdiag); CeedChkBackend(ierr);
  ierr = CeedFree(&emodein); CeedChkBackend(ierr);
  ierr = CeedFree(&emodeout); CeedChkBackend(ierr);
  ierr = CeedFree(&identity); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble composite diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleAddDiagonalCompositeCore_Ref(
  CeedOperator op, CeedVector assembled, CeedRequest *request,
  const bool pointBlock) {
  int ierr;
  CeedInt numSub;
  CeedOperator *subOperators;
  ierr = CeedOperatorGetNumSub(op, &numSub); CeedChkBackend(ierr);
  ierr = CeedOperatorGetSubList(op, &subOperators); CeedChkBackend(ierr);
  for (CeedInt i = 0; i < numSub; i++) {
    ierr = CeedOperatorAssembleAddDiagonalCore_Ref(subOperators[i], assembled,
           request, pointBlock); CeedChkBackend(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonal_Ref(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr;
  bool isComposite;
  ierr = CeedOperatorIsComposite(op, &isComposite); CeedChkBackend(ierr);
  if (isComposite) {
    return CeedOperatorLinearAssembleAddDiagonalCompositeCore_Ref(op, assembled,
           request, false);
  } else {
    return CeedOperatorAssembleAddDiagonalCore_Ref(op, assembled, request, false);
  }
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Ref(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr;
  bool isComposite;
  ierr = CeedOperatorIsComposite(op, &isComposite); CeedChkBackend(ierr);
  if (isComposite) {
    return CeedOperatorLinearAssembleAddDiagonalCompositeCore_Ref(op, assembled,
           request, true);
  } else {
    return CeedOperatorAssembleAddDiagonalCore_Ref(op, assembled, request, true);
  }
}

//------------------------------------------------------------------------------
// Create FDM Element Inverse
//------------------------------------------------------------------------------
int CeedOperatorCreateFDMElementInverse_Ref(CeedOperator op,
    CeedOperator *fdminv, CeedRequest *request) {
  int ierr;
  Ceed ceed, ceedparent;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  ierr = CeedGetOperatorFallbackParentCeed(ceed, &ceedparent);
  CeedChkBackend(ierr);
  ceedparent = ceedparent ? ceedparent : ceed;
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);

  // Determine active input basis
  bool interp = false, grad = false;
  CeedBasis basis = NULL;
  CeedElemRestriction rstr = NULL;
  CeedOperatorField *opfields;
  CeedQFunctionField *qffields;
  ierr = CeedOperatorGetFields(op, &opfields, NULL); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetFields(qf, &qffields, NULL); CeedChkBackend(ierr);
  CeedInt numinputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, NULL); CeedChkBackend(ierr);
  for (CeedInt i=0; i<numinputfields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(opfields[i], &vec); CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedEvalMode emode;
      ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChkBackend(ierr);
      interp = interp || emode == CEED_EVAL_INTERP;
      grad = grad || emode == CEED_EVAL_GRAD;
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &rstr);
      CeedChkBackend(ierr);
    }
  }
  if (!basis)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No active field set");
  // LCOV_EXCL_STOP
  CeedInt P1d, Q1d, elemsize, nqpts, dim, ncomp = 1, nelem = 1, lsize = 1;
  ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basis, &elemsize); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpts); CeedChkBackend(ierr);
  ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumElements(rstr, &nelem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetLVectorSize(rstr, &lsize); CeedChkBackend(ierr);

  // Build and diagonalize 1D Mass and Laplacian
  bool tensorbasis;
  ierr = CeedBasisIsTensor(basis, &tensorbasis); CeedChkBackend(ierr);
  if (!tensorbasis)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "FDMElementInverse only supported for tensor "
                     "bases");
  // LCOV_EXCL_STOP
  CeedScalar *work, *mass, *laplace, *x, *x2, *lambda;
  ierr = CeedMalloc(Q1d*P1d, &work); CeedChkBackend(ierr);
  ierr = CeedMalloc(P1d*P1d, &mass); CeedChkBackend(ierr);
  ierr = CeedMalloc(P1d*P1d, &laplace); CeedChkBackend(ierr);
  ierr = CeedMalloc(P1d*P1d, &x); CeedChkBackend(ierr);
  ierr = CeedMalloc(P1d*P1d, &x2); CeedChkBackend(ierr);
  ierr = CeedMalloc(P1d, &lambda); CeedChkBackend(ierr);
  // -- Mass
  const CeedScalar *interp1d, *grad1d, *qweight1d;
  ierr = CeedBasisGetInterp1D(basis, &interp1d); CeedChkBackend(ierr);
  ierr = CeedBasisGetGrad1D(basis, &grad1d); CeedChkBackend(ierr);
  ierr = CeedBasisGetQWeights(basis, &qweight1d); CeedChkBackend(ierr);
  for (CeedInt i=0; i<Q1d; i++)
    for (CeedInt j=0; j<P1d; j++)
      work[i+j*Q1d] = interp1d[i*P1d+j]*qweight1d[i];
  ierr = CeedMatrixMultiply(ceed, (const CeedScalar *)work,
                            (const CeedScalar *)interp1d, mass, P1d, P1d, Q1d);
  CeedChkBackend(ierr);
  // -- Laplacian
  for (CeedInt i=0; i<Q1d; i++)
    for (CeedInt j=0; j<P1d; j++)
      work[i+j*Q1d] = grad1d[i*P1d+j]*qweight1d[i];
  ierr = CeedMatrixMultiply(ceed, (const CeedScalar *)work,
                            (const CeedScalar *)grad1d, laplace, P1d, P1d, Q1d);
  CeedChkBackend(ierr);
  // -- Diagonalize
  ierr = CeedSimultaneousDiagonalization(ceed, laplace, mass, x, lambda, P1d);
  CeedChkBackend(ierr);
  ierr = CeedFree(&work); CeedChkBackend(ierr);
  ierr = CeedFree(&mass); CeedChkBackend(ierr);
  ierr = CeedFree(&laplace); CeedChkBackend(ierr);
  for (CeedInt i=0; i<P1d; i++)
    for (CeedInt j=0; j<P1d; j++)
      x2[i+j*P1d] = x[j+i*P1d];
  ierr = CeedFree(&x); CeedChkBackend(ierr);

  // Assemble QFunction
  CeedVector assembled;
  CeedElemRestriction rstr_qf;
  ierr =  CeedOperatorLinearAssembleQFunction(op, &assembled, &rstr_qf,
          request); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_qf); CeedChkBackend(ierr);
  CeedScalar maxnorm = 0;
  ierr = CeedVectorNorm(assembled, CEED_NORM_MAX, &maxnorm); CeedChkBackend(ierr);

  // Calculate element averages
  CeedInt nfields = ((interp?1:0) + (grad?dim:0))*((interp?1:0) + (grad?dim:0));
  CeedScalar *elemavg;
  const CeedScalar *assembledarray, *qweightsarray;
  CeedVector qweights;
  ierr = CeedVectorCreate(ceedparent, nqpts, &qweights); CeedChkBackend(ierr);
  ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                        CEED_VECTOR_NONE, qweights); CeedChkBackend(ierr);
  ierr = CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembledarray);
  CeedChkBackend(ierr);
  ierr = CeedVectorGetArrayRead(qweights, CEED_MEM_HOST, &qweightsarray);
  CeedChkBackend(ierr);
  ierr = CeedCalloc(nelem, &elemavg); CeedChkBackend(ierr);
  for (CeedInt e=0; e<nelem; e++) {
    CeedInt count = 0;
    for (CeedInt q=0; q<nqpts; q++)
      for (CeedInt i=0; i<ncomp*ncomp*nfields; i++)
        if (fabs(assembledarray[e*nelem*nqpts*ncomp*ncomp*nfields +
                                                                  i*nqpts + q]) > maxnorm*1e-12) {
          elemavg[e] += assembledarray[e*nelem*nqpts*ncomp*ncomp*nfields +
                                       i*nqpts + q] / qweightsarray[q];
          count++;
        }
    if (count)
      elemavg[e] /= count;
  }
  ierr = CeedVectorRestoreArrayRead(assembled, &assembledarray);
  CeedChkBackend(ierr);
  ierr = CeedVectorDestroy(&assembled); CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArrayRead(qweights, &qweightsarray);
  CeedChkBackend(ierr);
  ierr = CeedVectorDestroy(&qweights); CeedChkBackend(ierr);

  // Build FDM diagonal
  CeedVector qdata;
  CeedScalar *qdataarray;
  ierr = CeedVectorCreate(ceedparent, nelem*ncomp*lsize, &qdata);
  CeedChkBackend(ierr);
  ierr = CeedVectorSetArray(qdata, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
  CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(qdata, CEED_MEM_HOST, &qdataarray);
  CeedChkBackend(ierr);
  for (CeedInt e=0; e<nelem; e++)
    for (CeedInt c=0; c<ncomp; c++)
      for (CeedInt n=0; n<lsize; n++) {
        if (interp)
          qdataarray[(e*ncomp+c)*lsize+n] = 1;
        if (grad)
          for (CeedInt d=0; d<dim; d++) {
            CeedInt i = (n / CeedIntPow(P1d, d)) % P1d;
            qdataarray[(e*ncomp+c)*lsize+n] += lambda[i];
          }
        qdataarray[(e*ncomp+c)*lsize+n] = 1 / (elemavg[e] *
                                               qdataarray[(e*ncomp+c)*lsize+n]);
      }
  ierr = CeedFree(&elemavg); CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArray(qdata, &qdataarray); CeedChkBackend(ierr);

  // Setup FDM operator
  // -- Basis
  CeedBasis fdm_basis;
  CeedScalar *graddummy, *qrefdummy, *qweightdummy;
  ierr = CeedCalloc(P1d*P1d, &graddummy); CeedChkBackend(ierr);
  ierr = CeedCalloc(P1d, &qrefdummy); CeedChkBackend(ierr);
  ierr = CeedCalloc(P1d, &qweightdummy); CeedChkBackend(ierr);
  ierr = CeedBasisCreateTensorH1(ceedparent, dim, ncomp, P1d, P1d, x2,
                                 graddummy, qrefdummy, qweightdummy,
                                 &fdm_basis); CeedChkBackend(ierr);
  ierr = CeedFree(&graddummy); CeedChkBackend(ierr);
  ierr = CeedFree(&qrefdummy); CeedChkBackend(ierr);
  ierr = CeedFree(&qweightdummy); CeedChkBackend(ierr);
  ierr = CeedFree(&x2); CeedChkBackend(ierr);
  ierr = CeedFree(&lambda); CeedChkBackend(ierr);

  // -- Restriction
  CeedElemRestriction rstr_i;
  CeedInt strides[3] = {1, lsize, lsize*ncomp};
  ierr = CeedElemRestrictionCreateStrided(ceedparent, nelem, lsize, ncomp,
                                          lsize*nelem*ncomp, strides, &rstr_i);
  CeedChkBackend(ierr);
  // -- QFunction
  CeedQFunction mass_qf;
  ierr = CeedQFunctionCreateInteriorByName(ceedparent, "MassApply", &mass_qf);
  CeedChkBackend(ierr);
  // -- Operator
  ierr = CeedOperatorCreate(ceedparent, mass_qf, NULL, NULL, fdminv);
  CeedChkBackend(ierr);
  CeedOperatorSetField(*fdminv, "u", rstr_i, fdm_basis, CEED_VECTOR_ACTIVE);
  CeedChkBackend(ierr);
  CeedOperatorSetField(*fdminv, "qdata", rstr_i, CEED_BASIS_COLLOCATED, qdata);
  CeedChkBackend(ierr);
  CeedOperatorSetField(*fdminv, "v", rstr_i, fdm_basis, CEED_VECTOR_ACTIVE);
  CeedChkBackend(ierr);

  // Cleanup
  ierr = CeedVectorDestroy(&qdata); CeedChkBackend(ierr);
  ierr = CeedBasisDestroy(&fdm_basis); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_i); CeedChkBackend(ierr);
  ierr = CeedQFunctionDestroy(&mass_qf); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Destroy
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Ref(CeedOperator op) {
  int ierr;
  CeedOperator_Ref *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);

  for (CeedInt i=0; i<impl->numein+impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->evecs); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->edata); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->inputstate); CeedChkBackend(ierr);

  for (CeedInt i=0; i<impl->numein; i++) {
    ierr = CeedVectorDestroy(&impl->evecsin[i]); CeedChkBackend(ierr);
    ierr = CeedVectorDestroy(&impl->qvecsin[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->evecsin); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->qvecsin); CeedChkBackend(ierr);

  for (CeedInt i=0; i<impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->evecsout[i]); CeedChkBackend(ierr);
    ierr = CeedVectorDestroy(&impl->qvecsout[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->evecsout); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->qvecsout); CeedChkBackend(ierr);

  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Create
//------------------------------------------------------------------------------
int CeedOperatorCreate_Ref(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Ref *impl;

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedOperatorSetData(op, impl); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction",
                                CeedOperatorLinearAssembleQFunction_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal",
                                CeedOperatorLinearAssembleAddDiagonal_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleAddPointBlockDiagonal",
                                CeedOperatorLinearAssembleAddPointBlockDiagonal_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "CreateFDMElementInverse",
                                CeedOperatorCreateFDMElementInverse_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd",
                                CeedOperatorApplyAdd_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Ref); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Composite Operator Create
//------------------------------------------------------------------------------
int CeedCompositeOperatorCreate_Ref(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal",
                                CeedOperatorLinearAssembleAddDiagonal_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleAddPointBlockDiagonal",
                                CeedOperatorLinearAssembleAddPointBlockDiagonal_Ref);
  CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
