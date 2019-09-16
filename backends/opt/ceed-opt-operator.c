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

#include <string.h>
#include "ceed-opt.h"
#include "../ref/ceed-ref.h"

static int CeedOperatorDestroy_Opt(CeedOperator op) {
  int ierr;
  CeedOperator_Opt *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numein+impl->numeout; i++) {
    ierr = CeedElemRestrictionDestroy(&impl->blkrestr[i]); CeedChk(ierr);
    ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->blkrestr); CeedChk(ierr);
  ierr = CeedFree(&impl->evecs); CeedChk(ierr);
  ierr = CeedFree(&impl->edata); CeedChk(ierr);
  ierr = CeedFree(&impl->inputstate); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numein; i++) {
    ierr = CeedVectorDestroy(&impl->evecsin[i]); CeedChk(ierr);
    ierr = CeedVectorDestroy(&impl->qvecsin[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->evecsin); CeedChk(ierr);
  ierr = CeedFree(&impl->qvecsin); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->evecsout[i]); CeedChk(ierr);
    ierr = CeedVectorDestroy(&impl->qvecsout[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->evecsout); CeedChk(ierr);
  ierr = CeedFree(&impl->qvecsout); CeedChk(ierr);

  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

/*
  Setup infields or outfields
 */
static int CeedOperatorSetupFields_Opt(CeedQFunction qf, CeedOperator op,
                                       bool inOrOut, const CeedInt blksize,
                                       CeedElemRestriction *blkrestr,
                                       CeedVector *fullevecs, CeedVector *evecs,
                                       CeedVector *qvecs, CeedInt starte,
                                       CeedInt numfields, CeedInt Q) {
  CeedInt dim, ierr, ncomp, size, P;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedBasis basis;
  CeedElemRestriction r;
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
  for (CeedInt i=0; i<numfields; i++) {
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChk(ierr);

    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &r);
      CeedChk(ierr);
      CeedElemRestriction_Ref *data;
      ierr = CeedElemRestrictionGetData(r, (void *)&data); CeedChk(ierr);
      Ceed ceed;
      ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
      CeedInt nelem, elemsize, nnodes;
      ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
      ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
      ierr = CeedElemRestrictionGetNumNodes(r, &nnodes); CeedChk(ierr);
      ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
      ierr = CeedElemRestrictionCreateBlocked(ceed, nelem, elemsize,
                                              blksize, nnodes, ncomp,
                                              CEED_MEM_HOST, CEED_COPY_VALUES,
                                              data->indices, &blkrestr[i+starte]);
      CeedChk(ierr);
      ierr = CeedElemRestrictionCreateVector(blkrestr[i+starte], NULL,
                                             &fullevecs[i+starte]);
      CeedChk(ierr);
    }

    switch(emode) {
    case CEED_EVAL_NONE:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blksize, &evecs[i]); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blksize, &qvecs[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChk(ierr);
      ierr = CeedElemRestrictionGetElementSize(r, &P);
      CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, P*size*blksize, &evecs[i]); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blksize, &qvecs[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedQFunctionFieldGetSize(qffields[i], &size); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
      ierr = CeedElemRestrictionGetElementSize(r, &P);
      CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, P*size/dim*blksize, &evecs[i]); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blksize, &qvecs[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*blksize, &qvecs[i]); CeedChk(ierr);
      ierr = CeedBasisApply(basis, blksize, CEED_NOTRANSPOSE,
                            CEED_EVAL_WEIGHT, NULL, qvecs[i]); CeedChk(ierr);

      break;
    case CEED_EVAL_DIV:
      break; // Not implimented
    case CEED_EVAL_CURL:
      break; // Not implimented
    }
  }
  return 0;
}

/*
  CeedOperator needs to connect all the named fields (be they active or passive)
  to the named inputs and outputs of its CeedQFunction.
 */
static int CeedOperatorSetup_Opt(CeedOperator op) {
  int ierr;
  bool setupdone;
  ierr = CeedOperatorGetSetupStatus(op, &setupdone); CeedChk(ierr);
  if (setupdone) return 0;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  Ceed_Opt *ceedimpl;
  ierr = CeedGetData(ceed, (void *)&ceedimpl); CeedChk(ierr);
  const CeedInt blksize = ceedimpl->blksize;
  CeedOperator_Opt *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, numinputfields, numoutputfields;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);

  // Allocate
  ierr = CeedCalloc(numinputfields + numoutputfields, &impl->blkrestr);
  CeedChk(ierr);
  ierr = CeedCalloc(numinputfields + numoutputfields, &impl->evecs);
  CeedChk(ierr);
  ierr = CeedCalloc(numinputfields + numoutputfields, &impl->edata);
  CeedChk(ierr);

  ierr = CeedCalloc(16, &impl->inputstate); CeedChk(ierr);
  ierr = CeedCalloc(16, &impl->evecsin); CeedChk(ierr);
  ierr = CeedCalloc(16, &impl->evecsout); CeedChk(ierr);
  ierr = CeedCalloc(16, &impl->qvecsin); CeedChk(ierr);
  ierr = CeedCalloc(16, &impl->qvecsout); CeedChk(ierr);

  impl->numein = numinputfields; impl->numeout = numoutputfields;

  // Set up infield and outfield pointer arrays
  // Infields
  ierr = CeedOperatorSetupFields_Opt(qf, op, 0, blksize, impl->blkrestr,
                                     impl->evecs, impl->evecsin,
                                     impl->qvecsin, 0,
                                     numinputfields, Q);
  CeedChk(ierr);
  // Outfields
  ierr = CeedOperatorSetupFields_Opt(qf, op, 1, blksize, impl->blkrestr,
                                     impl->evecs, impl->evecsout,
                                     impl->qvecsout, numinputfields,
                                     numoutputfields, Q);
  CeedChk(ierr);

  ierr = CeedOperatorSetSetupDone(op); CeedChk(ierr);

  return 0;
}

static inline int CeedOperatorApply_Opt(CeedOperator op,
                                        const CeedInt blksize, CeedVector invec,
                                        CeedVector outvec,
                                        CeedRequest *request) {
  int ierr;
  CeedOperator_Opt *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);
  CeedInt Q, elemsize, numinputfields, numoutputfields, numelements, size, dim;
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  CeedInt nblks = (numelements/blksize) + !!(numelements%blksize);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
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
  uint64_t state;

  // Setup
  ierr = CeedOperatorSetup_Opt(op); CeedChk(ierr);

  // Input Evecs and Restriction
  for (CeedInt i=0; i<numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      if (vec != CEED_VECTOR_ACTIVE) {
        // Restrict
        ierr = CeedVectorGetState(vec, &state); CeedChk(ierr);
        if (state != impl->inputstate[i]) {
          ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode);
          CeedChk(ierr);
          ierr = CeedElemRestrictionApply(impl->blkrestr[i], CEED_NOTRANSPOSE,
                                          lmode, vec, impl->evecs[i], request);
          CeedChk(ierr);
          impl->inputstate[i] = state;
        }
      } else {
        // Set Qvec for CEED_EVAL_NONE
        if (emode == CEED_EVAL_NONE) {
          ierr = CeedVectorGetArray(impl->evecsin[i], CEED_MEM_HOST,
                                    &impl->edata[i]); CeedChk(ierr);
          ierr = CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_HOST,
                                    CEED_USE_POINTER,
                                    impl->edata[i]); CeedChk(ierr);
          ierr = CeedVectorRestoreArray(impl->evecsin[i],
                                        &impl->edata[i]); CeedChk(ierr);
        }
      }
      // Get evec
      ierr = CeedVectorGetArrayRead(impl->evecs[i], CEED_MEM_HOST,
                                    (const CeedScalar **) &impl->edata[i]);
      CeedChk(ierr);
    }
  }

  // Output Lvecs, Evecs, and Qvecs
  for (CeedInt i=0; i<numoutputfields; i++) {
    // Zero Lvecs
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      if (!impl->add) {
        vec = outvec;
        ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
      }
    } else {
      ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
    }
    // Set Qvec if needed
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChk(ierr);
    if (emode == CEED_EVAL_NONE) {
      // Set qvec to single block evec
      ierr = CeedVectorGetArray(impl->evecsout[i], CEED_MEM_HOST,
                                &impl->edata[i + numinputfields]);
      CeedChk(ierr);
      ierr = CeedVectorSetArray(impl->qvecsout[i], CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                impl->edata[i + numinputfields]); CeedChk(ierr);
      ierr = CeedVectorRestoreArray(impl->evecsout[i],
                                    &impl->edata[i + numinputfields]);
      CeedChk(ierr);
    }
  }
  impl->add = false;

  // Loop through elements
  for (CeedInt e=0; e<nblks*blksize; e+=blksize) {
    // Input basis apply if needed
    for (CeedInt i=0; i<numinputfields; i++) {
      CeedInt activein = 0;
      // Get elemsize, emode, size
      ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
      CeedChk(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
      CeedChk(ierr);
      ierr = CeedQFunctionFieldGetSize(qfinputfields[i], &size); CeedChk(ierr);
      // Restrict block active input
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE) {
        ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode);
        CeedChk(ierr);
        ierr = CeedElemRestrictionApplyBlock(impl->blkrestr[i], e/blksize,
                                             CEED_NOTRANSPOSE, lmode, invec,
                                             impl->evecsin[i], request);
        CeedChk(ierr);
        activein = 1;
      }
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        if (!activein) {
          ierr = CeedVectorSetArray(impl->qvecsin[i], CEED_MEM_HOST,
                                    CEED_USE_POINTER,
                                    &impl->edata[i][e*Q*size]); CeedChk(ierr);
        }
        break;
      case CEED_EVAL_INTERP:
        ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis);
        CeedChk(ierr);
        if (!activein) {
          ierr = CeedVectorSetArray(impl->evecsin[i], CEED_MEM_HOST,
                                    CEED_USE_POINTER,
                                    &impl->edata[i][e*elemsize*size]);
          CeedChk(ierr);
        }
        ierr = CeedBasisApply(basis, blksize, CEED_NOTRANSPOSE,
                              CEED_EVAL_INTERP, impl->evecsin[i],
                              impl->qvecsin[i]); CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis);
        CeedChk(ierr);
        if (!activein) {
          ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
          ierr = CeedVectorSetArray(impl->evecsin[i], CEED_MEM_HOST,
                                    CEED_USE_POINTER,
                                    &impl->edata[i][e*elemsize*size/dim]);
          CeedChk(ierr);
        }
        ierr = CeedBasisApply(basis, blksize, CEED_NOTRANSPOSE,
                              CEED_EVAL_GRAD, impl->evecsin[i],
                              impl->qvecsin[i]); CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
    }

    // Q function
    ierr = CeedQFunctionApply(qf, Q*blksize, impl->qvecsin, impl->qvecsout);
    CeedChk(ierr);

    // Output basis apply and restrict
    for (CeedInt i=0; i<numoutputfields; i++) {
      // Get elemsize, emode, size
      ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChk(ierr);
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        break; // No action
      case CEED_EVAL_INTERP:
        ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
        CeedChk(ierr);
        ierr = CeedBasisApply(basis, blksize, CEED_TRANSPOSE,
                              CEED_EVAL_INTERP, impl->qvecsout[i],
                              impl->evecsout[i]); CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
        CeedChk(ierr);
        ierr = CeedBasisApply(basis, blksize, CEED_TRANSPOSE,
                              CEED_EVAL_GRAD, impl->qvecsout[i],
                              impl->evecsout[i]); CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT: {
        Ceed ceed;
        ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
        return CeedError(ceed, 1,
                         "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        break; // Should not occur
      }
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
      // Restrict output block
      // Get output vector
      ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        vec = outvec;
      // Restrict
      ierr = CeedOperatorFieldGetLMode(opoutputfields[i], &lmode);
      CeedChk(ierr);
      ierr = CeedElemRestrictionApplyBlock(impl->blkrestr[i+impl->numein],
                                           e/blksize, CEED_TRANSPOSE,
                                           lmode, impl->evecsout[i],
                                           vec, request); CeedChk(ierr);
    }
  }

  // Restore input arrays
  for (CeedInt i=0; i<numinputfields; i++) {
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

int CeedOperatorApply_Opt_1(CeedOperator op, CeedVector invec,
                            CeedVector outvec, CeedRequest *request) {
  return CeedOperatorApply_Opt(op, 1, invec, outvec, request);
}

int CeedOperatorApply_Opt_8(CeedOperator op, CeedVector invec,
                            CeedVector outvec, CeedRequest *request) {
  return CeedOperatorApply_Opt(op, 8, invec, outvec, request);
}

int CeedOperatorCreate_Opt(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  Ceed_Opt *ceedimpl;
  ierr = CeedGetData(ceed, (void *)&ceedimpl); CeedChk(ierr);
  CeedInt blksize = ceedimpl->blksize;
  CeedOperator_Opt *impl;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedOperatorSetData(op, (void *)&impl); CeedChk(ierr);

  if (blksize == 1) {
    ierr = CeedSetBackendFunction(ceed, "Operator", op, "Apply",
                                  CeedOperatorApply_Opt_1); CeedChk(ierr);
  } else if (blksize == 8) {
    ierr = CeedSetBackendFunction(ceed, "Operator", op, "Apply",
                                  CeedOperatorApply_Opt_8); CeedChk(ierr);
  } else {
    return CeedError(ceed, 1, "Opt backend cannot use blocksize: %d", blksize);
  }

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Opt); CeedChk(ierr);
  return 0;
}
