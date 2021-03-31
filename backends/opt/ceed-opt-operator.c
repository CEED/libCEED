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
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "ceed-opt.h"

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Opt(CeedQFunction qf, CeedOperator op,
                                       bool inOrOut, const CeedInt blksize,
                                       CeedElemRestriction *blkrestr,
                                       CeedVector *fullevecs, CeedVector *evecs,
                                       CeedVector *qvecs, CeedInt starte,
                                       CeedInt numfields, CeedInt Q) {
  CeedInt dim, ierr, ncomp, size, P;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedBasis basis;
  CeedElemRestriction r;
  CeedOperatorField *opfields;
  CeedQFunctionField *qffields;
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
  for (CeedInt i=0; i<numfields; i++) {
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChkBackend(ierr);

    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &r);
      CeedChkBackend(ierr);
      Ceed ceed;
      ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
      CeedInt nelem, elemsize, lsize, compstride;
      ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetLVectorSize(r, &lsize); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChkBackend(ierr);

      bool strided;
      ierr = CeedElemRestrictionIsStrided(r, &strided); CeedChkBackend(ierr);
      if (strided) {
        CeedInt strides[3];
        ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChkBackend(ierr);
        ierr = CeedElemRestrictionCreateBlockedStrided(ceed, nelem, elemsize,
               blksize, ncomp, lsize, strides, &blkrestr[i+starte]);
        CeedChkBackend(ierr);
      } else {
        const CeedInt *offsets = NULL;
        ierr = CeedElemRestrictionGetOffsets(r, CEED_MEM_HOST, &offsets);
        CeedChkBackend(ierr);
        ierr = CeedElemRestrictionGetCompStride(r, &compstride); CeedChkBackend(ierr);
        ierr = CeedElemRestrictionCreateBlocked(ceed, nelem, elemsize,
                                                blksize, ncomp, compstride,
                                                lsize, CEED_MEM_HOST,
                                                CEED_COPY_VALUES, offsets,
                                                &blkrestr[i+starte]);
        CeedChkBackend(ierr);
        ierr = CeedElemRestrictionRestoreOffsets(r, &offsets); CeedChkBackend(ierr);
      }
      ierr = CeedElemRestrictionCreateVector(blkrestr[i+starte], NULL,
                                             &fullevecs[i+starte]);
      CeedChkBackend(ierr);
    }

    switch(emode) {
    case CEED_EVAL_NONE:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blksize, &evecs[i]); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blksize, &qvecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetElementSize(r, &P);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, P*size*blksize, &evecs[i]); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blksize, &qvecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetElementSize(r, &P);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, P*size/dim*blksize, &evecs[i]);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blksize, &qvecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*blksize, &qvecs[i]); CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, blksize, CEED_NOTRANSPOSE,
                            CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, qvecs[i]);
      CeedChkBackend(ierr);

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
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Opt(CeedOperator op) {
  int ierr;
  bool setupdone;
  ierr = CeedOperatorIsSetupDone(op, &setupdone); CeedChkBackend(ierr);
  if (setupdone) return CEED_ERROR_SUCCESS;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  Ceed_Opt *ceedimpl;
  ierr = CeedGetData(ceed, &ceedimpl); CeedChkBackend(ierr);
  const CeedInt blksize = ceedimpl->blksize;
  CeedOperator_Opt *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt Q, numinputfields, numoutputfields;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  ierr = CeedQFunctionIsIdentity(qf, &impl->identityqf); CeedChkBackend(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChkBackend(ierr);

  // Allocate
  ierr = CeedCalloc(numinputfields + numoutputfields, &impl->blkrestr);
  CeedChkBackend(ierr);
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

  // Set up infield and outfield pointer arrays
  // Infields
  ierr = CeedOperatorSetupFields_Opt(qf, op, 0, blksize, impl->blkrestr,
                                     impl->evecs, impl->evecsin,
                                     impl->qvecsin, 0,
                                     numinputfields, Q);
  CeedChkBackend(ierr);
  // Outfields
  ierr = CeedOperatorSetupFields_Opt(qf, op, 1, blksize, impl->blkrestr,
                                     impl->evecs, impl->evecsout,
                                     impl->qvecsout, numinputfields,
                                     numoutputfields, Q);
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
// Setup Input Fields
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Opt(CeedInt numinputfields,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    CeedVector invec, CeedOperator_Opt *impl, CeedRequest *request) {
  CeedInt ierr;
  CeedEvalMode emode;
  CeedVector vec;
  uint64_t state;

  for (CeedInt i=0; i<numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
      if (vec != CEED_VECTOR_ACTIVE) {
        // Restrict
        ierr = CeedVectorGetState(vec, &state); CeedChkBackend(ierr);
        if (state != impl->inputstate[i]) {
          ierr = CeedElemRestrictionApply(impl->blkrestr[i], CEED_NOTRANSPOSE,
                                          vec, impl->evecs[i], request);
          CeedChkBackend(ierr);
          impl->inputstate[i] = state;
        }
      } else {
        // Set Qvec for CEED_EVAL_NONE
        if (emode == CEED_EVAL_NONE) {
          ierr = CeedVectorGetArray(impl->evecsin[i], CEED_MEM_HOST,
                                    &impl->edata[i]); CeedChkBackend(ierr);
          ierr = CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_HOST,
                                    CEED_USE_POINTER,
                                    impl->edata[i]); CeedChkBackend(ierr);
          ierr = CeedVectorRestoreArray(impl->evecsin[i],
                                        &impl->edata[i]); CeedChkBackend(ierr);
        }
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
static inline int CeedOperatorInputBasis_Opt(CeedInt e, CeedInt Q,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    CeedInt numinputfields, CeedInt blksize, CeedVector invec, bool skipactive,
    CeedOperator_Opt *impl, CeedRequest *request) {
  CeedInt ierr;
  CeedInt dim, elemsize, size;
  CeedElemRestriction Erestrict;
  CeedEvalMode emode;
  CeedBasis basis;
  CeedVector vec;

  for (CeedInt i=0; i<numinputfields; i++) {
    ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
    // Skip active input
    if (skipactive) {
      if (vec == CEED_VECTOR_ACTIVE)
        continue;
    }

    CeedInt activein = 0;
    // Get elemsize, emode, size
    ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetSize(qfinputfields[i], &size); CeedChkBackend(ierr);
    // Restrict block active input
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedElemRestrictionApplyBlock(impl->blkrestr[i], e/blksize,
                                           CEED_NOTRANSPOSE, invec,
                                           impl->evecsin[i], request);
      CeedChkBackend(ierr);
      activein = 1;
    }
    // Basis action
    switch(emode) {
    case CEED_EVAL_NONE:
      if (!activein) {
        ierr = CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &impl->edata[i][e*Q*size]); CeedChkBackend(ierr);
      }
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis);
      CeedChkBackend(ierr);
      if (!activein) {
        ierr = CeedVectorSetArray(impl->evecsin[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &impl->edata[i][e*elemsize*size]);
        CeedChkBackend(ierr);
      }
      ierr = CeedBasisApply(basis, blksize, CEED_NOTRANSPOSE,
                            CEED_EVAL_INTERP, impl->evecsin[i],
                            impl->qvecsin[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis);
      CeedChkBackend(ierr);
      if (!activein) {
        ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
        ierr = CeedVectorSetArray(impl->evecsin[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &impl->edata[i][e*elemsize*size/dim]);
        CeedChkBackend(ierr);
      }
      ierr = CeedBasisApply(basis, blksize, CEED_NOTRANSPOSE,
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
static inline int CeedOperatorOutputBasis_Opt(CeedInt e, CeedInt Q,
    CeedQFunctionField *qfoutputfields, CeedOperatorField *opoutputfields,
    CeedInt blksize, CeedInt numinputfields, CeedInt numoutputfields,
    CeedOperator op, CeedVector outvec, CeedOperator_Opt *impl,
    CeedRequest *request) {
  CeedInt ierr;
  CeedElemRestriction Erestrict;
  CeedEvalMode emode;
  CeedBasis basis;
  CeedVector vec;

  for (CeedInt i=0; i<numoutputfields; i++) {
    // Get elemsize, emode, size
    ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    // Basis action
    switch(emode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, blksize, CEED_TRANSPOSE,
                            CEED_EVAL_INTERP, impl->qvecsout[i],
                            impl->evecsout[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, blksize, CEED_TRANSPOSE,
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
    // Restrict output block
    // Get output vector
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec);
    CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;
    // Restrict
    ierr = CeedElemRestrictionApplyBlock(impl->blkrestr[i+impl->numein],
                                         e/blksize, CEED_TRANSPOSE,
                                         impl->evecsout[i], vec, request);
    CeedChkBackend(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Opt(CeedInt numinputfields,
    CeedQFunctionField *qfinputfields, CeedOperatorField *opinputfields,
    CeedOperator_Opt *impl) {
  CeedInt ierr;
  CeedEvalMode emode;

  for (CeedInt i=0; i<numinputfields; i++) {
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
static int CeedOperatorApplyAdd_Opt(CeedOperator op, CeedVector invec,
                                    CeedVector outvec, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  Ceed_Opt *ceedimpl;
  ierr = CeedGetData(ceed, &ceedimpl); CeedChkBackend(ierr);
  CeedInt blksize = ceedimpl->blksize;
  CeedOperator_Opt *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedInt Q, numinputfields, numoutputfields, numelements;
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChkBackend(ierr);
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  CeedInt nblks = (numelements/blksize) + !!(numelements%blksize);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChkBackend(ierr);
  CeedEvalMode emode;

  // Setup
  ierr = CeedOperatorSetup_Opt(op); CeedChkBackend(ierr);

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Opt(numinputfields, qfinputfields,
                                     opinputfields, invec, impl, request);
  CeedChkBackend(ierr);

  // Output Lvecs, Evecs, and Qvecs
  for (CeedInt i=0; i<numoutputfields; i++) {
    // Set Qvec if needed
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_NONE) {
      // Set qvec to single block evec
      ierr = CeedVectorGetArray(impl->evecsout[i], CEED_MEM_HOST,
                                &impl->edata[i + numinputfields]);
      CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->qvecsout[i], CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                impl->edata[i + numinputfields]); CeedChkBackend(ierr);
      ierr = CeedVectorRestoreArray(impl->evecsout[i],
                                    &impl->edata[i + numinputfields]);
      CeedChkBackend(ierr);
    }
  }

  // Loop through elements
  for (CeedInt e=0; e<nblks*blksize; e+=blksize) {
    // Input basis apply
    ierr = CeedOperatorInputBasis_Opt(e, Q, qfinputfields, opinputfields,
                                      numinputfields, blksize, invec, false,
                                      impl, request); CeedChkBackend(ierr);

    // Q function
    if (!impl->identityqf) {
      ierr = CeedQFunctionApply(qf, Q*blksize, impl->qvecsin, impl->qvecsout);
      CeedChkBackend(ierr);
    }

    // Output basis apply and restrict
    ierr = CeedOperatorOutputBasis_Opt(e, Q, qfoutputfields, opoutputfields,
                                       blksize, numinputfields, numoutputfields,
                                       op, outvec, impl, request);
    CeedChkBackend(ierr);
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Opt(numinputfields, qfinputfields,
                                       opinputfields, impl);
  CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Opt(CeedOperator op,
    CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  Ceed_Opt *ceedimpl;
  ierr = CeedGetData(ceed, &ceedimpl); CeedChkBackend(ierr);
  const CeedInt blksize = ceedimpl->blksize;
  CeedOperator_Opt *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedInt Q, numinputfields, numoutputfields, numelements, size;
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChkBackend(ierr);
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  CeedInt nblks = (numelements/blksize) + !!(numelements%blksize);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChkBackend(ierr);
  CeedVector vec, lvec;
  CeedInt numactivein = 0, numactiveout = 0;
  CeedVector *activein = NULL;
  CeedScalar *a, *tmp;

  // Setup
  ierr = CeedOperatorSetup_Opt(op); CeedChkBackend(ierr);

  // Check for identity
  if (impl->identityqf)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Assembling identity qfunctions not supported");
  // LCOV_EXCL_STOP

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Opt(numinputfields, qfinputfields,
                                     opinputfields, NULL, impl, request);
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
        ierr = CeedVectorCreate(ceed, Q*blksize, &activein[numactivein+field]);
        CeedChkBackend(ierr);
        ierr = CeedVectorSetArray(activein[numactivein+field], CEED_MEM_HOST,
                                  CEED_USE_POINTER, &tmp[field*Q*blksize]);
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

  // Setup lvec
  ierr = CeedVectorCreate(ceed, nblks*blksize*Q*numactivein*numactiveout,
                          &lvec); CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(lvec, CEED_MEM_HOST, &a); CeedChkBackend(ierr);

  // Create output restriction
  CeedInt strides[3] = {1, Q, numactivein *numactiveout*Q};
  ierr = CeedElemRestrictionCreateStrided(ceed, numelements, Q,
                                          numactivein*numactiveout,
                                          numactivein*numactiveout*numelements*Q,
                                          strides, rstr); CeedChkBackend(ierr);
  // Create assembled vector
  ierr = CeedVectorCreate(ceed, numelements*Q*numactivein*numactiveout,
                          assembled); CeedChkBackend(ierr);

  // Loop through elements
  for (CeedInt e=0; e<nblks*blksize; e+=blksize) {
    // Input basis apply
    ierr = CeedOperatorInputBasis_Opt(e, Q, qfinputfields, opinputfields,
                                      numinputfields, blksize, NULL, true,
                                      impl, request); CeedChkBackend(ierr);

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
          a += size*Q*blksize; // Advance the pointer by the size of the output
        }
      }
      // Apply QFunction
      ierr = CeedQFunctionApply(qf, Q*blksize, impl->qvecsin, impl->qvecsout);
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
      CeedVectorSetArray(impl->qvecsout[out], CEED_MEM_HOST, CEED_COPY_VALUES,
                         NULL); CeedChkBackend(ierr);
    }
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Opt(numinputfields, qfinputfields,
                                       opinputfields, impl);
  CeedChkBackend(ierr);

  // Output blocked restriction
  ierr = CeedVectorRestoreArray(lvec, &a); CeedChkBackend(ierr);
  ierr = CeedVectorSetValue(*assembled, 0.0); CeedChkBackend(ierr);
  CeedElemRestriction blkrstr;
  ierr = CeedElemRestrictionCreateBlockedStrided(ceed, numelements, Q, blksize,
         numactivein*numactiveout, numactivein*numactiveout*numelements*Q,
         strides, &blkrstr); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionApply(blkrstr, CEED_TRANSPOSE, lvec, *assembled,
                                  request); CeedChkBackend(ierr);

  // Cleanup
  for (CeedInt i=0; i<numactivein; i++) {
    ierr = CeedVectorDestroy(&activein[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&activein); CeedChkBackend(ierr);
  ierr = CeedVectorDestroy(&lvec); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionDestroy(&blkrstr); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Destroy
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Opt(CeedOperator op) {
  int ierr;
  CeedOperator_Opt *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);

  for (CeedInt i=0; i<impl->numein+impl->numeout; i++) {
    ierr = CeedElemRestrictionDestroy(&impl->blkrestr[i]); CeedChkBackend(ierr);
    ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->blkrestr); CeedChkBackend(ierr);
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
int CeedOperatorCreate_Opt(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  Ceed_Opt *ceedimpl;
  ierr = CeedGetData(ceed, &ceedimpl); CeedChkBackend(ierr);
  CeedInt blksize = ceedimpl->blksize;
  CeedOperator_Opt *impl;

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedOperatorSetData(op, impl); CeedChkBackend(ierr);

  if (blksize != 1 && blksize != 8)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Opt backend cannot use blocksize: %d", blksize);
  // LCOV_EXCL_STOP

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction",
                                CeedOperatorLinearAssembleQFunction_Opt);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd",
                                CeedOperatorApplyAdd_Opt); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Opt); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
