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

CeedElemRestriction CEED_RESTRICTION_IDENTITY = NULL;
CeedBasis CEED_BASIS_COLOCATED = NULL;
CeedVector CEED_VECTOR_ACTIVE = NULL;
CeedVector CEED_VECTOR_NONE = NULL;

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
                                       CeedInt starti, CeedInt starte,
                                       CeedInt startq, CeedInt numfields, CeedInt Q) {
  CeedInt dim, ierr, ie=starte, iq=startq, ncomp;

  // Loop over fields
  for (CeedInt i=0; i<numfields; i++) {
    if (ofields[i].Erestrict) {
      ierr = CeedElemRestrictionCreateVector(ofields[i].Erestrict, NULL, &evecs[ie]);
      CeedChk(ierr);
      ie++;
    }
    CeedEvalMode emode = qfields[i].emode;
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
      ierr = CeedBasisApply(ofields[iq].basis, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
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
  CeedOperator_Ref *opref = op->data;
  CeedQFunction qf = op->qf;
  CeedInt Q = op->numqpoints;
  int ierr;

  // Count infield and outfield array sizes and evectors
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    opref->numqin += !!(emode & CEED_EVAL_INTERP) + !!(emode & CEED_EVAL_GRAD) + !!
                     (emode & CEED_EVAL_WEIGHT);
    opref->numein +=
      !!op->inputfields[i].Erestrict; // Need E-vector when restriction exists
  }
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    CeedEvalMode emode = qf->outputfields[i].emode;
    opref->numqout += !!(emode & CEED_EVAL_INTERP) + !!(emode & CEED_EVAL_GRAD);
    opref->numeout += !!op->outputfields[i].Erestrict;
  }

  // Allocate
  ierr = CeedCalloc(opref->numein + opref->numeout, &opref->evecs); CeedChk(ierr);
  ierr = CeedCalloc(qf->numinputfields + qf->numoutputfields, &opref->edata);
  CeedChk(ierr);

  ierr = CeedCalloc(opref->numqin + opref->numqout, &opref->qdata_alloc);
  CeedChk(ierr);
  ierr = CeedCalloc(qf->numinputfields + qf->numoutputfields, &opref->qdata);
  CeedChk(ierr);

  ierr = CeedCalloc(16, &opref->indata); CeedChk(ierr);
  ierr = CeedCalloc(16, &opref->outdata); CeedChk(ierr);

  // Set up infield and outfield pointer arrays
  // Infields
  ierr = CeedOperatorSetupFields_Ref(qf->inputfields, op->inputfields,
                                     opref->evecs, opref->qdata, opref->qdata_alloc,
                                     opref->indata, 0, 0, 0,
                                     qf->numinputfields, Q); CeedChk(ierr);

  // Outfields
  ierr = CeedOperatorSetupFields_Ref(qf->outputfields, op->outputfields,
                                     opref->evecs, opref->qdata, opref->qdata_alloc,
                                     opref->indata, qf->numinputfields, opref->numein,
                                     opref->numqin, qf->numoutputfields, Q); CeedChk(ierr);

  op->setupdone = 1;

  return 0;
}

static int CeedOperatorApply_Ref(CeedOperator op, CeedVector invec,
                                 CeedVector outvec, CeedRequest *request) {
  CeedOperator_Ref *opref = op->data;
  CeedInt Q = op->numqpoints, elemsize;
  int ierr;
  CeedQFunction qf = op->qf;
  CeedTransposeMode lmode = CEED_NOTRANSPOSE;

  // Setup
  ierr = CeedOperatorSetup_Ref(op); CeedChk(ierr);

  // Input Evecs and Restriction
  for (CeedInt i=0,iein=0; i<qf->numinputfields; i++) {
    // Restriction
    if (op->inputfields[i].Erestrict) {
      // Passive
      if (op->inputfields[i].vec) {
        ierr = CeedElemRestrictionApply(op->inputfields[i].Erestrict, CEED_NOTRANSPOSE,
                                        lmode, op->inputfields[i].vec, opref->evecs[iein],
                                        request); CeedChk(ierr);
        ierr = CeedVectorGetArrayRead(opref->evecs[iein], CEED_MEM_HOST,
                                      (const CeedScalar **) &opref->edata[i]); CeedChk(ierr);
        iein++;
      } else {
        // Active
        ierr = CeedElemRestrictionApply(op->inputfields[i].Erestrict, CEED_NOTRANSPOSE,
                                        lmode, invec, opref->evecs[iein], request); CeedChk(ierr);
        ierr = CeedVectorGetArrayRead(opref->evecs[iein], CEED_MEM_HOST,
                                      (const CeedScalar **) &opref->edata[i]); CeedChk(ierr);
        iein++;
      }
    } else {
      // No restriction
      CeedEvalMode emode = qf->inputfields[i].emode;
      if (emode & CEED_EVAL_WEIGHT) {
      } else {
        ierr = CeedVectorGetArrayRead(op->inputfields[i].vec, CEED_MEM_HOST,
                                      (const CeedScalar **) &opref->edata[i]); CeedChk(ierr);
      }
    }
  }

  // Output Evecs
  for (CeedInt i=0,ieout=opref->numein; i<qf->numoutputfields; i++) {
    // Restriction
    if (op->outputfields[i].Erestrict) {
      ierr = CeedVectorGetArray(opref->evecs[ieout], CEED_MEM_HOST,
                                &opref->edata[i + qf->numinputfields]); CeedChk(ierr);
      ieout++;
    } else {
      // No restriction
      // Passive
      if (op->inputfields[i].vec) {
        ierr = CeedVectorGetArray(op->inputfields[i].vec, CEED_MEM_HOST,
                                  &opref->edata[i + qf->numinputfields]); CeedChk(ierr);
      } else {
        // Active
        ierr = CeedVectorGetArray(outvec, CEED_MEM_HOST,
                                  &opref->edata[i + qf->numinputfields]); CeedChk(ierr);
      }
    }
  }

  // Output Qvecs
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    CeedEvalMode emode = qf->outputfields[i].emode;
    if (emode != CEED_EVAL_NONE) {
      opref->outdata[i] =  opref->qdata[i + qf->numinputfields];
    }
  }

  // Loop through elements
  for (CeedInt e=0; e<op->numelements; e++) {
    // Input basis apply if needed
    for (CeedInt i=0; i<qf->numinputfields; i++) {
      // Get elemsize
      if (op->inputfields[i].Erestrict) {
        elemsize = op->inputfields[i].Erestrict->elemsize;
      } else {
        elemsize = Q;
      }
      // Get emode, ncomp
      CeedEvalMode emode = qf->inputfields[i].emode;
      CeedInt ncomp = qf->inputfields[i].ncomp;
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        opref->indata[i] = &opref->edata[i][e*Q*ncomp];
        break;
      case CEED_EVAL_INTERP:
        ierr = CeedBasisApply(op->inputfields[i].basis, CEED_NOTRANSPOSE,
                              CEED_EVAL_INTERP, &opref->edata[i][e*elemsize*ncomp], opref->qdata[i]);
        CeedChk(ierr);
        opref->indata[i] = opref->qdata[i];
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedBasisApply(op->inputfields[i].basis, CEED_NOTRANSPOSE,
                              CEED_EVAL_GRAD, &opref->edata[i][e*elemsize*ncomp], opref->qdata[i]);
        CeedChk(ierr);
        opref->indata[i] = opref->qdata[i];
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
        opref->outdata[i] = &opref->edata[i + qf->numinputfields][e*Q*ncomp];
      }
    }
    // Q function
    ierr = CeedQFunctionApply(op->qf, Q, (const CeedScalar * const*) opref->indata,
                              opref->outdata); CeedChk(ierr);

    // Output basis apply if needed
    for (CeedInt i=0; i<qf->numoutputfields; i++) {
      // Get elemsize
      if (op->outputfields[i].Erestrict) {
        elemsize = op->outputfields[i].Erestrict->elemsize;
      } else {
        elemsize = Q;
      }
      // Get emode, ncomp
      CeedInt ncomp = qf->outputfields[i].ncomp;
      CeedEvalMode emode = qf->outputfields[i].emode;
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        break; // No action
      case CEED_EVAL_INTERP:
        ierr = CeedBasisApply(op->outputfields[i].basis, CEED_TRANSPOSE,
                              CEED_EVAL_INTERP, opref->outdata[i],
                              &opref->edata[i + qf->numinputfields][e*elemsize*ncomp]); CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedBasisApply(op->outputfields[i].basis, CEED_TRANSPOSE, CEED_EVAL_GRAD,
                              opref->outdata[i], &opref->edata[i + qf->numinputfields][e*elemsize*ncomp]);
        CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT:
        break; // Should not occur
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
    }
  }

  // Output restriction
  for (CeedInt i=0,ieout=opref->numein; i<qf->numoutputfields; i++) {
    // Restriction
    if (op->outputfields[i].Erestrict) {
      // Passive
      if (op->outputfields[i].vec) {
        ierr = CeedVectorRestoreArray(opref->evecs[ieout],
                                      &opref->edata[i + qf->numinputfields]); CeedChk(ierr);
        ierr = CeedElemRestrictionApply(op->outputfields[i].Erestrict, CEED_TRANSPOSE,
                                        lmode, opref->evecs[ieout], op->outputfields[i].vec, request); CeedChk(ierr);
        ieout++;
      } else {
        // Active
        ierr = CeedVectorRestoreArray(opref->evecs[ieout],
                                      &opref->edata[i + qf->numinputfields]); CeedChk(ierr);
        ierr = CeedElemRestrictionApply(op->outputfields[i].Erestrict, CEED_TRANSPOSE,
                                        lmode, opref->evecs[ieout], outvec, request); CeedChk(ierr);
        ieout++;
      }
    } else {
      // No Restriction
      // Passive
      if (op->outputfields[i].vec) {
        ierr = CeedVectorRestoreArray(op->outputfields[i].vec,
                                      &opref->edata[i + qf->numinputfields]); CeedChk(ierr);
      } else {
        // Active
        ierr = CeedVectorRestoreArray(outvec, &opref->edata[i + qf->numinputfields]);
        CeedChk(ierr);
      }
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
