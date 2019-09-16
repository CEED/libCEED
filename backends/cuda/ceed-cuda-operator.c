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

#include <ceed-backend.h>
#include "ceed-cuda.h"
#include <string.h>

static int CeedOperatorDestroy_Cuda(CeedOperator op) {
  int ierr;
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);

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

  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

/*
  Setup infields or outfields
 */
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

    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedElemRestrictionCreateVector(Erestrict, NULL,
                                             &evecs[i + starte]);
      CeedChk(ierr);
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
      ierr = CeedBasisApply(basis, numelements, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                            NULL, qvecs[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  return 0;
}

/*
  CeedOperator needs to connect all the named fields (be they active or passive)
  to the named inputs and outputs of its CeedQFunction.
 */
static int CeedOperatorSetup_Cuda(CeedOperator op) {
  int ierr;
  bool setupdone;
  ierr = CeedOperatorGetSetupStatus(op, &setupdone); CeedChk(ierr);
  if (setupdone) return 0;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);
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
                                      numinputfields, numoutputfields, Q, numelements);
  CeedChk(ierr);

  ierr = CeedOperatorSetSetupDone(op); CeedChk(ierr);

  return 0;
}

static int CeedOperatorApply_Cuda(CeedOperator op, CeedVector invec,
                                  CeedVector outvec, CeedRequest *request) {
  int ierr;
  CeedOperator_Cuda *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, numelements, elemsize, numinputfields, numoutputfields, size;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedTransposeMode lmode;
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
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        vec = invec;
      // Restrict
      ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode); CeedChk(ierr);
      ierr = CeedElemRestrictionApply(Erestrict, CEED_NOTRANSPOSE,
                                      lmode, vec, impl->evecs[i],
                                      request); CeedChk(ierr);
      // Get evec
      ierr = CeedVectorGetArrayRead(impl->evecs[i], CEED_MEM_DEVICE,
                                    (const CeedScalar **) &impl->edata[i]);
      CeedChk(ierr);
    }
  }

  // Input basis apply if needed
  for (CeedInt i = 0; i < numinputfields; i++) {
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
      break;  // No action
    case CEED_EVAL_DIV:
      break; // TODO: Not implemented
    case CEED_EVAL_CURL:
      break; // TODO: Not implemented
    }
  }
  // Output pointers
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_NONE) {
      ierr = CeedVectorGetArray(impl->evecs[i + impl->numein], CEED_MEM_DEVICE,
                                &impl->edata[i + numinputfields]); CeedChk(ierr);
      ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &size); CeedChk(ierr);
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
      break; // No action
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
    }
  }

  // Zero lvecs
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;
    ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
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
    // Active
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;
    // Restrict
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChk(ierr);
    ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode); CeedChk(ierr);
    ierr = CeedElemRestrictionApply(Erestrict, CEED_TRANSPOSE,
                                    lmode, impl->evecs[i + impl->numein], vec,
                                    request); CeedChk(ierr);
  }

  // Restore input arrays
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedVectorRestoreArrayRead(impl->evecs[i],
                                        (const CeedScalar **) &impl->edata[i]);
      CeedChk(ierr);
    }
  }

  return 0;
}

int CeedOperatorCreate_Cuda(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Cuda *impl;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedOperatorSetData(op, (void *)&impl);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Apply",
                                CeedOperatorApply_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Cuda); CeedChk(ierr);
  return 0;
}

int CeedCompositeOperatorCreate_Cuda(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement composite operators");
}
