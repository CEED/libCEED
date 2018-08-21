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

#include <ceed-impl.h>
#include <string.h>
#include "ceed-ref.h"

static int CeedOperatorDestroy_Ref(CeedOperator op) {
  CeedOperator_Ref *impl = op->data;
  int ierr;

  for (CeedInt i=0; i<impl->numein+impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->evecs); CeedChk(ierr);
  ierr = CeedFree(&impl->edata); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numqin+impl->numqout; i++) {
    ierr = CeedFree(&impl->qdata_alloc[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->qdata_alloc); CeedChk(ierr);
  ierr = CeedFree(&impl->qdata); CeedChk(ierr);

  ierr = CeedFree(&impl->indata); CeedChk(ierr);
  ierr = CeedFree(&impl->outdata); CeedChk(ierr);

  ierr = CeedFree(&op->data); CeedChk(ierr);
  return 0;
}

/*
  Setup infields or outfields
 */
static int CeedOperatorSetupFields_Ref(struct CeedQFunctionField qfields[16],
                                       struct CeedOperatorField ofields[16],
                                       CeedVector *evecs, CeedScalar **qdata,
                                       CeedScalar **qdata_alloc, CeedScalar **indata,
                                       CeedInt starti, CeedInt startq,
                                       CeedInt numfields, CeedInt Q) {
  CeedInt dim, ierr, iq=startq, ncomp;

  // Loop over fields
  for (CeedInt i=0; i<numfields; i++) {
    CeedEvalMode emode = qfields[i].emode;

    if (emode != CEED_EVAL_WEIGHT) {
      ierr = CeedElemRestrictionCreateVector(ofields[i].Erestrict, NULL,
                                             &evecs[i+starti]);
      CeedChk(ierr);
    }

    switch(emode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ncomp = qfields[i].ncomp;
      ierr = CeedMalloc(Q*ncomp, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_GRAD:
      ncomp = qfields[i].ncomp;
      dim = ofields[i].basis->dim;
      ierr = CeedMalloc(Q*ncomp*dim, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedMalloc(Q, &qdata_alloc[iq]); CeedChk(ierr);
      ierr = CeedBasisApply(ofields[iq].basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                            NULL, qdata_alloc[iq]); CeedChk(ierr);
      qdata[i] = qdata_alloc[iq];
      indata[i] = qdata[i];
      iq++;
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
static int CeedOperatorSetup_Ref(CeedOperator op) {
  if (op->setupdone) return 0;
  CeedOperator_Ref *impl = op->data;
  CeedQFunction qf = op->qf;
  CeedInt Q = op->numqpoints;
  int ierr;

  // Count infield and outfield array sizes and evectors
  impl->numein = qf->numinputfields;
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    impl->numqin += !!(emode & CEED_EVAL_INTERP) + !!(emode & CEED_EVAL_GRAD) +
                    !!(emode & CEED_EVAL_WEIGHT);
  }
  impl->numeout = qf->numoutputfields;
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    CeedEvalMode emode = qf->outputfields[i].emode;
    impl->numqout += !!(emode & CEED_EVAL_INTERP) + !!(emode & CEED_EVAL_GRAD);
  }

  // Allocate
  ierr = CeedCalloc(impl->numein + impl->numeout, &impl->evecs); CeedChk(ierr);
  ierr = CeedCalloc(impl->numein + impl->numeout, &impl->edata);
  CeedChk(ierr);

  ierr = CeedCalloc(impl->numqin + impl->numqout, &impl->qdata_alloc);
  CeedChk(ierr);
  ierr = CeedCalloc(qf->numinputfields + qf->numoutputfields, &impl->qdata);
  CeedChk(ierr);

  ierr = CeedCalloc(16, &impl->indata); CeedChk(ierr);
  ierr = CeedCalloc(16, &impl->outdata); CeedChk(ierr);

  // Set up infield and outfield pointer arrays
  // Infields
  ierr = CeedOperatorSetupFields_Ref(qf->inputfields, op->inputfields,
                                     impl->evecs, impl->qdata, impl->qdata_alloc,
                                     impl->indata, 0, 0,
                                     qf->numinputfields, Q); CeedChk(ierr);

  // Outfields
  ierr = CeedOperatorSetupFields_Ref(qf->outputfields, op->outputfields,
                                     impl->evecs, impl->qdata, impl->qdata_alloc,
                                     impl->indata, qf->numinputfields,
                                     impl->numqin, qf->numoutputfields, Q); CeedChk(ierr);

  // Input Qvecs
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    if ((emode != CEED_EVAL_NONE) && (emode != CEED_EVAL_WEIGHT))
      impl->indata[i] =  impl->qdata[i];
  }
  // Output Qvecs
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    CeedEvalMode emode = qf->outputfields[i].emode;
    if (emode != CEED_EVAL_NONE)
      impl->outdata[i] =  impl->qdata[i + qf->numinputfields];
  }

  op->setupdone = 1;

  return 0;
}

static int CeedOperatorApply_Ref(CeedOperator op, CeedVector invec,
                                 CeedVector outvec, CeedRequest *request) {
  CeedOperator_Ref *impl = op->data;
  CeedInt Q = op->numqpoints, elemsize;
  int ierr;
  CeedQFunction qf = op->qf;
  CeedTransposeMode lmode = CEED_NOTRANSPOSE;

  // Setup
  ierr = CeedOperatorSetup_Ref(op); CeedChk(ierr);

  // Input Evecs and Restriction
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Active
      if (op->inputfields[i].vec == CEED_VECTOR_ACTIVE) {
        // Restrict
        ierr = CeedElemRestrictionApply(op->inputfields[i].Erestrict, CEED_NOTRANSPOSE,
                                        lmode, invec, impl->evecs[i],
                                        request); CeedChk(ierr);
        // Get evec
        ierr = CeedVectorGetArrayRead(impl->evecs[i], CEED_MEM_HOST,
                                      (const CeedScalar **) &impl->edata[i]); CeedChk(ierr);
      } else {
        // Passive
        // Restrict
        ierr = CeedElemRestrictionApply(op->inputfields[i].Erestrict, CEED_NOTRANSPOSE,
                                        lmode, op->inputfields[i].vec, impl->evecs[i],
                                        request); CeedChk(ierr);
        // Get evec
        ierr = CeedVectorGetArrayRead(impl->evecs[i], CEED_MEM_HOST,
                                      (const CeedScalar **) &impl->edata[i]); CeedChk(ierr);
      }
    }
  }

  // Output Evecs
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    ierr = CeedVectorGetArray(impl->evecs[i+impl->numein], CEED_MEM_HOST,
                              &impl->edata[i + qf->numinputfields]); CeedChk(ierr);
  }

  // Loop through elements
  for (CeedInt e=0; e<op->numelements; e++) {
    // Input basis apply if needed
    for (CeedInt i=0; i<qf->numinputfields; i++) {
      // Get elemsize, emode, ncomp
      elemsize = op->inputfields[i].Erestrict->elemsize;
      CeedEvalMode emode = qf->inputfields[i].emode;
      CeedInt ncomp = qf->inputfields[i].ncomp;
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        impl->indata[i] = &impl->edata[i][e*Q*ncomp];
        break;
      case CEED_EVAL_INTERP:
        ierr = CeedBasisApply(op->inputfields[i].basis, 1, CEED_NOTRANSPOSE,
                              CEED_EVAL_INTERP, &impl->edata[i][e*elemsize*ncomp], impl->qdata[i]);
        CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedBasisApply(op->inputfields[i].basis, 1, CEED_NOTRANSPOSE,
                              CEED_EVAL_GRAD, &impl->edata[i][e*elemsize*ncomp], impl->qdata[i]);
        CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
    }
    // Output pointers
    for (CeedInt i=0; i<qf->numoutputfields; i++) {
      CeedEvalMode emode = qf->outputfields[i].emode;
      if (emode == CEED_EVAL_NONE) {
        CeedInt ncomp = qf->outputfields[i].ncomp;
        impl->outdata[i] = &impl->edata[i + qf->numinputfields][e*Q*ncomp];
      }
    }
    // Q function
    ierr = CeedQFunctionApply(op->qf, Q, (const CeedScalar * const*) impl->indata,
                              impl->outdata); CeedChk(ierr);

    // Output basis apply if needed
    for (CeedInt i=0; i<qf->numoutputfields; i++) {
      // Get elemsize, emode, ncomp
      elemsize = op->outputfields[i].Erestrict->elemsize;
      CeedInt ncomp = qf->outputfields[i].ncomp;
      CeedEvalMode emode = qf->outputfields[i].emode;
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        break; // No action
      case CEED_EVAL_INTERP:
        ierr = CeedBasisApply(op->outputfields[i].basis, 1, CEED_TRANSPOSE,
                              CEED_EVAL_INTERP, impl->outdata[i],
                              &impl->edata[i + qf->numinputfields][e*elemsize*ncomp]);
        CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedBasisApply(op->outputfields[i].basis, 1, CEED_TRANSPOSE,
                              CEED_EVAL_GRAD,
                              impl->outdata[i], &impl->edata[i + qf->numinputfields][e*elemsize*ncomp]);
        CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT:
        return CeedError(op->ceed, 1,
                         "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        break; // Should not occur
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
    }
  }

  // Zero lvecs
  ierr = CeedVectorSetValue(outvec, 0.0); CeedChk(ierr);
  for (CeedInt i=0; i<qf->numoutputfields; i++)
    if (op->outputfields[i].vec != CEED_VECTOR_ACTIVE) {
      ierr = CeedVectorSetValue(op->outputfields[i].vec, 0.0); CeedChk(ierr);
    }

  // Output restriction
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    // Restore evec
    ierr = CeedVectorRestoreArray(impl->evecs[i+impl->numein],
                                  &impl->edata[i + qf->numinputfields]); CeedChk(ierr);
    // Active
    if (op->outputfields[i].vec == CEED_VECTOR_ACTIVE) {
      // Restrict
      ierr = CeedElemRestrictionApply(op->outputfields[i].Erestrict, CEED_TRANSPOSE,
                                      lmode, impl->evecs[i+impl->numein], outvec, request); CeedChk(ierr);
    } else {
      // Passive
      // Restrict
      ierr = CeedElemRestrictionApply(op->outputfields[i].Erestrict, CEED_TRANSPOSE,
                                      lmode, impl->evecs[i+impl->numein], op->outputfields[i].vec,
                                      request); CeedChk(ierr);
    }
  }

  // Restore input arrays
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedVectorRestoreArrayRead(impl->evecs[i],
                                        (const CeedScalar **) &impl->edata[i]); CeedChk(ierr);
    }
  }

  return 0;
}

int CeedOperatorCreate_Ref(CeedOperator op) {
  CeedOperator_Ref *impl;
  int ierr;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  op->data = impl;
  op->Destroy = CeedOperatorDestroy_Ref;
  op->Apply = CeedOperatorApply_Ref;
  return 0;
}
