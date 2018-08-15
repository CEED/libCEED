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
#define CEED_DEBUG_COLOR 198
#include "ceed-occa.h"

// *****************************************************************************
// * Destroy the CeedOperator_Occa
// *****************************************************************************
static int CeedOperatorDestroy_Occa(CeedOperator op) {
  CeedOperator_Occa *impl = op->data;
  int ierr;

  for (CeedInt i=0; i<impl->numein+impl->numeout; i++) {
    if (impl->Evecs[i]) {
      ierr = CeedVectorDestroy(&impl->Evecs[i]); CeedChk(ierr);
    }
  }
  ierr = CeedFree(&impl->Evecs); CeedChk(ierr);
  ierr = CeedFree(&impl->Edata); CeedChk(ierr);

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
// *****************************************************************************
// * Apply Basis action to an input
// *****************************************************************************
/*static int CeedOperatorBasisAction_Occa(CeedVector *evec, CeedVector *qvec,
                                        CeedBasis basis,
                                        CeedEvalMode emode, CeedInt i, CeedInt Q,
                                        CeedScalar **qdata, CeedScalar **indata) {
  CeedInt ierr;
  const Ceed ceed = basis->ceed;
  dbg("\t[CeedOperator][BasisAction]");

  switch(emode) {
  case CEED_EVAL_NONE: // No basis action, evec = qvec
    dbg("\t[CeedOperator][BasisAction] CEED_EVAL_NONE");
    ierr = CeedVectorGetArray(*evec, CEED_MEM_HOST, &qdata[i]);
    CeedChk(ierr);
    indata[i] = qdata[i];
    break;
  case CEED_EVAL_INTERP:
    dbg("\t[CeedOperator][BasisAction] CEED_EVAL_INTERP");
    ierr = CeedBasisApplyElems_Occa(basis, Q, CEED_NOTRANSPOSE,
                                    CEED_EVAL_INTERP, *evec, *qvec);
    CeedChk(ierr);
    ierr = CeedVectorGetArray(*qvec, CEED_MEM_HOST, &qdata[i]);
    CeedChk(ierr);
    indata[i] = qdata[i];
    break;
  case CEED_EVAL_GRAD:
    dbg("\t[CeedOperator][BasisAction] CEED_EVAL_GRAD");
    ierr = CeedBasisApplyElems_Occa(basis, Q, CEED_NOTRANSPOSE,
                                    CEED_EVAL_GRAD, *evec, *qvec);
    CeedChk(ierr);
    ierr = CeedVectorGetArray(*qvec, CEED_MEM_HOST, &qdata[i]);
    CeedChk(ierr);
    indata[i] = qdata[i];
    break;
  case CEED_EVAL_WEIGHT:
    dbg("\t[CeedOperator][BasisAction] CEED_EVAL_WEIGHT");
    ierr = CeedBasisApplyElems_Occa(basis, Q, CEED_NOTRANSPOSE,
                                    CEED_EVAL_WEIGHT, *evec, *qvec);
    CeedChk(ierr);
    ierr = CeedVectorGetArray(*qvec, CEED_MEM_HOST, &qdata[i]);
    CeedChk(ierr);
    indata[i] = qdata[i];
    break;
  case CEED_EVAL_DIV: break; // Not implemented
  case CEED_EVAL_CURL: break; // Not implemented
  }
  dbg("\t[CeedOperator][BasisAction] done");
  return 0;
  }*/

// *****************************************************************************
// * Dump data
// *****************************************************************************
static int CeedOperatorDump_Occa(CeedOperator op) {
  const Ceed ceed = op->ceed;
  const CeedQFunction qf = op->qf;
  CeedOperator_Occa *data = op->data;

  const CeedInt numE = data->numein + data->numeout;
  const CeedInt numQ = data->numqin + data->numqout;
  const CeedInt numIO = qf->numinputfields + qf->numoutputfields;
  for (CeedInt i=0; i<numE; i++) {
    if (data->Evecs[i]) {
      dbg("[CeedOperator][Dump] \033[7mdata->Evecs[%d]",i);
    }
  }
  for (CeedInt i=0; i<numIO; i++) {
    if (data->Edata[i]) {
      dbg("[CeedOperator][Dump] \033[7mdata->Edata[%d]",i);
    }
  }
  for (CeedInt i=0; i<numQ; i++) {
    if (data->qdata_alloc[i]) {
      dbg("[CeedOperator][Dump] \033[7mdata->qdata_alloc[%d]",i);
    }
  }
  for (CeedInt i=0; i<numIO; i++) {
    if (data->qdata[i]) {
      dbg("[CeedOperator][Dump] \033[7mdata->qdata[%d]",i);
    }
  }
  for (CeedInt i=0; i<16; i++) {
    if (data->indata[i]) {
      dbg("[CeedOperator][Dump] \033[7mdata->INdata[%d]",i);
    }
  }
  for (CeedInt i=0; i<16; i++) {
    if (data->outdata[i]) {
      dbg("[CeedOperator][Dump] \033[7mdata->OUTdata[%d]",i);
    }
  }
  return 0;
}

// *****************************************************************************
// * Setup infields or outfields
// *****************************************************************************
static int CeedOperatorSetupFields_Occa(CeedOperator op,
                                        struct CeedQFunctionField qfields[16],
                                        struct CeedOperatorField ofields[16],
                                        CeedVector *evecs, CeedScalar **qdata, CeedScalar **qdata_alloc,
                                        CeedScalar **indata,
                                        const CeedInt starti,
                                        CeedInt startq,
                                        const CeedInt numfields,
                                        const CeedInt Q) {
  CeedQFunction qf = op->qf;
  CeedQFunction_Occa *qf_data = qf->data;
  const Ceed ceed = op->ceed;
  CeedInt ierr, ncomp;
  CeedInt dim = 1;
  CeedInt iq=startq;
  // Loop over fields
  for (CeedInt i=0; i<numfields; i++) {
    dbg("\t\t[CeedOperator][SetupFields] # %d/%d, \033[7m%s",i,numfields-1,
        qfields[i].fieldname);
    CeedEvalMode emode = qfields[i].emode;
    if (emode != CEED_EVAL_WEIGHT) {
      dbg("\t\t[CeedOperator][SetupFields] restriction");
      ierr = CeedElemRestrictionCreateVector(ofields[i].Erestrict, NULL, &evecs[i+starti]);
      CeedChk(ierr);
    } else {
      dbg("\t\t[CeedOperator][SetupFields] no restriction");
    }
    switch(emode) {
    case CEED_EVAL_NONE:
      dbg("\t\t[CeedOperator][SetupFields] NONE, Q==");
      break; // No action
    case CEED_EVAL_INTERP:
      dbg("\t\t[CeedOperator][SetupFields] INTERP, Q++, qdata[%d]=qdata_alloc[%d]",
          i + starti,iq);
      ncomp = qfields[i].ncomp;
      dim = ofields[i].basis->dim;
      ierr = CeedMalloc(Q*ncomp, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_GRAD:
      dbg("\t\t[CeedOperator][SetupFields] GRAD, Q++, qdata[%d]=qdata_alloc[%d]",
          i + starti,iq);
      ncomp = qfields[i].ncomp;
      dim = ofields[i].basis->dim;
      ierr = CeedMalloc(Q*ncomp*dim, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      dbg("\t\t[CeedOperator][SetupFields] WEIGHT, Q== & qdata[%d]=indata[%d]=qdata_alloc[%d]",
          i + starti,i,iq);
      ierr = CeedMalloc(Q, &qdata_alloc[iq]); CeedChk(ierr);
      ierr = CeedBasisApply(ofields[iq].basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                            NULL, qdata_alloc[iq]); CeedChk(ierr);
      assert(starti==0);
      dim = ofields[i].basis->dim;
      qdata[i + starti] = qdata_alloc[iq];
      indata[i] = qdata[i];
      break;
    case CEED_EVAL_DIV: break; // Not implemented
    case CEED_EVAL_CURL: break; // Not implemented
    }
    qf_data->dim = dim;
  }
  return 0;
}

// *****************************************************************************
// * CeedOperator needs to connect all the named fields (be they active or
// * passive) to the named inputs and outputs of its CeedQFunction.
// *****************************************************************************
static int CeedOperatorSetup_Occa(CeedOperator op) {
  if (op->setupdone) return 0;
  const Ceed ceed = op->ceed;
  CeedOperator_Occa *data = op->data;
  CeedQFunction qf = op->qf;
  CeedInt Q = op->numqpoints;
  int ierr;

  // ***************************************************************************
  // Count infield and outfield array sizes and evectors
  data->numein = qf->numinputfields;
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    data->numqin +=
      !! (emode & CEED_EVAL_INTERP) +
      !! (emode & CEED_EVAL_GRAD) +
      !! (emode & CEED_EVAL_WEIGHT);
  }
  dbg("\t[CeedOperator][Setup] numqin=%d, numein=%d",
      data->numqin, data->numein);

  // ***************************************************************************
  data->numeout = qf->numoutputfields;
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    CeedEvalMode emode = qf->outputfields[i].emode;
    data->numqout +=
      !! (emode & CEED_EVAL_INTERP) +
      !! (emode & CEED_EVAL_GRAD);
  }
  dbg("\t[CeedOperator][Setup] numqout=%d, numeout=%d",
      data->numqout, data->numeout);

  // Allocate ******************************************************************
  const CeedInt numE = data->numein + data->numeout;
  const CeedInt numQ = data->numqin + data->numqout;
  const CeedInt numIO = qf->numinputfields + qf->numoutputfields;
  dbg("\t[CeedOperator][Setup] numE=%d",numE);
  dbg("\t[CeedOperator][Setup] numQ=%d",numQ);
  dbg("\t[CeedOperator][Setup] numIO=%d (%d in, %d out)",numIO,
      qf->numinputfields, qf->numoutputfields);

  dbg("\t[CeedOperator][Setup] %d Evecs",numE);
  ierr = CeedCalloc(numE, &data->Evecs); CeedChk(ierr);
  dbg("\t[CeedOperator][Setup] %d Edata",numE);
  ierr = CeedCalloc(numE, &data->Edata); CeedChk(ierr);

  dbg("\t[CeedOperator][Setup] %d qdata_alloc",numQ);
  ierr = CeedCalloc(numQ, &data->qdata_alloc); CeedChk(ierr);
  dbg("\t[CeedOperator][Setup] %d qdata",numIO);
  ierr = CeedCalloc(numIO, &data->qdata); CeedChk(ierr);

  dbg("\t[CeedOperator][Setup] %d indata",16);
  ierr = CeedCalloc(16, &data->indata); CeedChk(ierr);
  dbg("\t[CeedOperator][Setup] %d outdata",16);
  ierr = CeedCalloc(16, &data->outdata); CeedChk(ierr);

  // Dump data before setting fields
  //dbg("\t[CeedOperator][Setup] Dump data before setting fields: (should be void)");
  //CeedOperatorDump_Occa(op);

  // Set up infield and outfield pointer arrays
  dbg("\t[CeedOperator][Setup] Set up IN fields:");
  // Infields
  ierr = CeedOperatorSetupFields_Occa(op,qf->inputfields, op->inputfields,
                                      data->Evecs, data->qdata, data->qdata_alloc, data->indata,
                                      0, 0,
                                      qf->numinputfields, Q);
  CeedChk(ierr);
  dbg("\t[CeedOperator][Setup] Set up OUT fields:");
  // Outfields
  ierr = CeedOperatorSetupFields_Occa(op,qf->outputfields, op->outputfields,
                                      data->Evecs, data->qdata, data->qdata_alloc, data->indata,
                                      qf->numinputfields, data->numqin,
                                      qf->numoutputfields, Q); CeedChk(ierr);
  op->setupdone = true;
  dbg("\t[CeedOperator][Setup] done");
  return 0;
}

static int SyncToHostPointer(CeedVector vec) {
  // The device copy is not updated in the host array by default.  We may need
  // to rethink memory management in this example, but this provides the
  // expected semantics when using CeedVectorSetArray for the vector that will
  // hold an output quantity.  This should at least be lazy instead of eager
  // and we should do better about avoiding copies.
  const CeedVector_Occa *outvdata = (CeedVector_Occa*)vec->data;
  if (outvdata->used_pointer) {
    occaCopyMemToPtr(outvdata->used_pointer, outvdata->d_array,
                     vec->length * sizeof(CeedScalar), NO_OFFSET, NO_PROPS);
  }
  return 0;
}

// *****************************************************************************
// * Apply CeedOperator to a vector
// *****************************************************************************
static int CeedOperatorApply_Occa(CeedOperator op,
                                  CeedVector invec,
                                  CeedVector outvec,
                                  CeedRequest *request) {
  const Ceed ceed = op->ceed;
  dbg("[CeedOperator][Apply]");
  CeedOperator_Occa *data = op->data;
  //CeedVector *E = data->Evecs, *D = data->D, outvec;
  CeedInt Q = op->numqpoints, elemsize;
  int ierr;
  CeedQFunction qf = op->qf;
  CeedTransposeMode lmode = CEED_NOTRANSPOSE;
  // ***************************************************************************
  dbg("[CeedOperator][Dump] Setup?");
  ierr = CeedOperatorSetup_Occa(op); CeedChk(ierr);
  //CeedOperatorDump_Occa(op);

  // Tell CeedQFunction_Occa's structure we are coming from an operator ********
  CeedQFunction_Occa *qfd = op->qf->data;
  qfd->op = op;

  // Input Evecs and Restriction
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    dbg("\n[CeedOperator][Apply] %d/%d Input Evecs:",i,qf->numinputfields-1);
    CeedEvalMode emode = qf->inputfields[i].emode;
    if (emode & CEED_EVAL_WEIGHT) {
      dbg("[CeedOperator][Apply] No restriction, WEIGHT");
    } else { // Restriction ****************************************************
      if (op->inputfields[i].vec == CEED_VECTOR_ACTIVE) { // Active
        dbg("[CeedOperator][Apply] Restriction/Active: data->Evecs[%d] = Edata[%d]",
            i,i);
        ierr = CeedElemRestrictionApply(op->inputfields[i].Erestrict, CEED_NOTRANSPOSE,
                                        lmode, invec, data->Evecs[i],
                                        request); CeedChk(ierr);
        ierr = CeedVectorGetArrayRead(data->Evecs[i], CEED_MEM_HOST,
                                      (const CeedScalar **) &data->Edata[i]); CeedChk(ierr);
      } else { // Passive
        dbg("[CeedOperator][Apply] Restriction/Passive: data->Evecs[%d] = Edata[i]",
            i,i);
        ierr = CeedElemRestrictionApply(op->inputfields[i].Erestrict, CEED_NOTRANSPOSE,
                                        lmode, op->inputfields[i].vec, data->Evecs[i],
                                        request); CeedChk(ierr);
        ierr = CeedVectorGetArrayRead(data->Evecs[i], CEED_MEM_HOST,
                                      (const CeedScalar **) &data->Edata[i]); CeedChk(ierr);
      }
    }
  }
  //dbg("\n[CeedOperator][Apply] Input Evecs and Restriction done, debug:");
  //CeedOperatorDump_Occa(op);

  // Output Evecs
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    dbg("\n[CeedOperator][Apply] %d/%d Output Evecs:",i,qf->numoutputfields-1);
    dbg("[CeedOperator][Apply] Restriction");
    ierr = CeedVectorGetArray(data->Evecs[i+data->numein], CEED_MEM_HOST,
                              &data->Edata[i + qf->numinputfields]); CeedChk(ierr);
  }
  //dbg("\n[CeedOperator][Apply] Output Evecs done, debug:");
  //CeedOperatorDump_Occa(op);

  // Output Qvecs **************************************************************
  dbg("\n[CeedOperator][Apply] Output Qvecs!");
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    CeedEvalMode emode = qf->outputfields[i].emode;
    if (emode != CEED_EVAL_NONE) {
      dbg("\n[CeedOperator][Apply] NONE, outdata++");
      data->outdata[i] =  data->qdata[i + qf->numinputfields];
    } else {
      dbg("\n[CeedOperator][Apply] else NONE");
    }
  }
  //dbg("\n[CeedOperator][Apply] Output Qvecs done, debug:");
  //CeedOperatorDump_Occa(op);

  // Loop through elements *****************************************************
  dbg("\n[CeedOperator][Apply] Loop through elements");
  for (CeedInt e=0; e<op->numelements; e++) {
    dbg("\n\t[CeedOperator][Apply] e # %d/%d",e,op->numelements-1);
    // Input basis apply if needed
    dbg("\t[CeedOperator][Apply] Input basis apply if needed");
    dbg("\t[CeedOperator][Apply] num input fields");
    for (CeedInt i=0; i<qf->numinputfields; i++) {
      const char *name = qf->inputfields[i].fieldname;
      dbg("\t\t[CeedOperator][Apply] IN \033[7m%s",name);
      // Get elemsize, emode, ncomp
      elemsize = op->inputfields[i].Erestrict->elemsize;
      const CeedEvalMode emode = qf->inputfields[i].emode;
      const CeedInt ncomp = qf->inputfields[i].ncomp;
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        dbg("\t\t[CeedOperator][Apply] in NONE, indata[%d] = Edata[%d]",i,i);
        data->indata[i] = &data->Edata[i][e*Q*ncomp];
        break;
      case CEED_EVAL_INTERP:
        dbg("\t\t[CeedOperator][Apply] in INTERP, basis, Edata[%d] => qdata[%d] => indata[%d]",
            i,i,i);
        ierr = CeedBasisApply(op->inputfields[i].basis, 1, CEED_NOTRANSPOSE,
                              CEED_EVAL_INTERP, &data->Edata[i][e*elemsize*ncomp],
                              data->qdata[i]);
        CeedChk(ierr);
        data->indata[i] = data->qdata[i];
        break;
      case CEED_EVAL_GRAD:
        dbg("\t\t[CeedOperator][Apply] in GRAD, basis, Edata[%d] => qdata[%d] => indata[%d]",
            i,i,i);
        ierr = CeedBasisApply(op->inputfields[i].basis, 1, CEED_NOTRANSPOSE,
                              CEED_EVAL_GRAD, &data->Edata[i][e*elemsize*ncomp],
                              data->qdata[i]);
        CeedChk(ierr);
        data->indata[i] = data->qdata[i];
        break;
      case CEED_EVAL_WEIGHT:
        dbg("\t\t[CeedOperator][Apply] in WEIGHT");
        break;  // No action
      case CEED_EVAL_DIV:
        dbg("\t\t[CeedOperator][Apply] in DIV");
        break; // Not implemented
      case CEED_EVAL_CURL:
        dbg("\t\t[CeedOperator][Apply] in CURL");
        break; // Not implemented
      }
    }
    // Output pointers
    dbg("\t[CeedOperator][Apply] num output fields");
    for (CeedInt i=0; i<qf->numoutputfields; i++) {
      CeedEvalMode emode = qf->outputfields[i].emode;
      const char *name = qf->inputfields[i].fieldname;
      dbg("\t\t[CeedOperator][Apply] OUT %s",name);
      if (emode == CEED_EVAL_NONE) {
        dbg("\t\t[CeedOperator][Apply] out NONE, Edata[%d] => outdata[%d]",
            i + qf->numinputfields,i);
        CeedInt ncomp = qf->outputfields[i].ncomp;
        data->outdata[i] = &data->Edata[i + qf->numinputfields][e*Q*ncomp];
      }
      if (emode == CEED_EVAL_INTERP) {
        dbg("\t\t[CeedOperator][Apply] out INTERP");
      }
      if (emode == CEED_EVAL_GRAD) {
        dbg("\t\t[CeedOperator][Apply] out GRAD");
      }
      if (emode == CEED_EVAL_WEIGHT) {
        dbg("\t\t[CeedOperator][Apply] out WEIGHT");
      }
    }

    dbg("\n[CeedOperator][Apply] before Q function debug:");
    CeedOperatorDump_Occa(op);

    // Q function
    dbg("\t[CeedOperator][Apply] Q function apply");
    ierr = CeedQFunctionApply(op->qf, Q, (const CeedScalar * const*) data->indata,
                              data->outdata); CeedChk(ierr);

    // Output basis apply if needed
    //dbg("\t[CeedOperator][Apply] Output basis apply if needed");
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
                              CEED_EVAL_INTERP, data->outdata[i],
                              &data->Edata[i + qf->numinputfields][e*elemsize*ncomp]); CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedBasisApply(op->outputfields[i].basis, 1, CEED_TRANSPOSE,
                              CEED_EVAL_GRAD,
                              data->outdata[i], &data->Edata[i + qf->numinputfields][e*elemsize*ncomp]);
        CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT: break; // Should not occur
      case CEED_EVAL_DIV: break; // Not implemented
      case CEED_EVAL_CURL: break; // Not implemented
      }
    }
  } // numelements

  // Output restriction
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    // Restore evec
    ierr = CeedVectorRestoreArray(data->Evecs[i+data->numein],
                                    &data->Edata[i + qf->numinputfields]); CeedChk(ierr);
    // Active
    if (op->outputfields[i].vec == CEED_VECTOR_ACTIVE) {
      // Zero lvec
      ierr = CeedVectorSetValue(outvec, 0.0); CeedChk(ierr);
      // Restrict
      ierr = CeedElemRestrictionApply(op->outputfields[i].Erestrict, CEED_TRANSPOSE,
                                      lmode, data->Evecs[i+data->numein], outvec, request); CeedChk(ierr);
      ierr = SyncToHostPointer(outvec); CeedChk(ierr);
    } else {
      // Passive
      // Zero lvec
      ierr = CeedVectorSetValue(op->outputfields[i].vec, 0.0); CeedChk(ierr);
      // Restrict
      ierr = CeedElemRestrictionApply(op->outputfields[i].Erestrict, CEED_TRANSPOSE,
                                      lmode, data->Evecs[i+data->numein], op->outputfields[i].vec, request); CeedChk(ierr);
      ierr = SyncToHostPointer(op->outputfields[i].vec); CeedChk(ierr);
    }
  }

  // Restore input arrays
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    if (emode & CEED_EVAL_WEIGHT) {
    } else {
      // Restriction
      ierr = CeedVectorRestoreArrayRead(data->Evecs[i],
                                        (const CeedScalar **) &data->Edata[i]); CeedChk(ierr);
    }
  }
  return 0;
}
/*
  // Fill CeedBasis_Occa's structure with CeedElemRestriction ******************
  CeedBasis_Occa *basis = op->basis->data;
  basis->er = op->Erestrict;
  // Fill CeedQFunction_Occa's structure with nc, dim & qdata ******************
  CeedQFunction_Occa *qfd = op->qf->data;
  qfd->op = true;
  qfd->nc = nc;
  qfd->dim = dim;
  qfd->nelem = nelem;
  qfd->elemsize = elemsize;
  qfd->d_q = ((CeedVector_Occa *)qdata->data)->d_array;
  // ***************************************************************************
  if (!data->etmp) {
    const int n = nc*nelem*elemsize;
    const int bn = Q*nc*(dim+2)*nelem;
    dbg("[CeedOperator][Apply] Setup, n=%d & bn=%d",n,bn);
    ierr = CeedVectorCreate(op->ceed,n,&data->etmp); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed,bn,&data->BEu); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed,bn,&data->BEv); CeedChk(ierr);
    // etmp is allocated when CeedVectorGetArray is called below
  }
  // Push the memory to the QFunction that will be used
  qfd->b_u = ((CeedVector_Occa *)data->BEu->data)->d_array;
  qfd->b_v = ((CeedVector_Occa *)data->BEv->data)->d_array;
  etmp = data->etmp;
  if (op->qf->inmode & ~CEED_EVAL_WEIGHT) {
    dbg("[CeedOperator][Apply] Apply Restriction");
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_NOTRANSPOSE,
                                    nc, lmode, ustate, etmp,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  }
  // We want to avoid Get/Restore
  ierr = CeedVectorGetArray(etmp, CEED_MEM_HOST, &Eu); CeedChk(ierr);
  // Fetching back data from device memory
  ierr = CeedVectorGetArray(qdata, CEED_MEM_HOST, (CeedScalar**)&qd);
  CeedChk(ierr);
  // Local arrays, sizes & pointers ********************************************
  CeedScalar BEu[Q*nc*(dim+2)], BEv[Q*nc*(dim+2)], *out[5] = {0,0,0,0,0};
  const CeedScalar *in[5] = {0,0,0,0,0};
  const size_t qbytes = op->qf->qdatasize;
  // ***************************************************************************
  ierr = CeedBasisApplyElems_Occa(op->basis,Q,CEED_NOTRANSPOSE,op->qf->inmode,
                                  data->etmp,data->BEu); CeedChk(ierr);
  // ***************************************************************************
  dbg("[CeedOperator][Apply] Q for-loop");
  for (CeedInt e=0; e<nelem; e++) {
    for(CeedInt k=0; k<(Q*nc*(dim+2)); k++) BEu[k]=0.0;
    ierr = CeedBasisApply(op->basis, CEED_NOTRANSPOSE,op->qf->inmode,
                          &Eu[e*nc*elemsize], BEu); CeedChk(ierr);
    CeedScalar *u_ptr = BEu, *v_ptr = BEv;
    if (op->qf->inmode & CEED_EVAL_INTERP) { in[0] = u_ptr; u_ptr += Q*nc; }
    if (op->qf->inmode & CEED_EVAL_GRAD) { in[1] = u_ptr; u_ptr += Q*nc*dim; }
    if (op->qf->inmode & CEED_EVAL_WEIGHT) { in[4] = u_ptr; u_ptr += Q; }
    if (op->qf->outmode & CEED_EVAL_INTERP) { out[0] = v_ptr; v_ptr += Q*nc; }
    if (op->qf->outmode & CEED_EVAL_GRAD) { out[1] = v_ptr; v_ptr += Q*nc*dim; }
    qfd->e = e;
    ierr = CeedQFunctionApply(op->qf, &qd[e*Q*qbytes], Q, in, out); CeedChk(ierr);
    ierr = CeedBasisApply(op->basis, CEED_TRANSPOSE,op->qf->outmode, BEv,
                          &Eu[e*nc*elemsize]); CeedChk(ierr);
  }
  // ***************************************************************************
  ierr = CeedBasisApplyElems_Occa(op->basis,Q,CEED_TRANSPOSE,op->qf->outmode,
                                  data->BEv,data->etmp); CeedChk(ierr);
  // *************************************************************************
  ierr = CeedVectorRestoreArray(etmp, &Eu); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(qdata, (CeedScalar**)&qd); CeedChk(ierr);
  // ***************************************************************************
  if (residual) {
    dbg("[CeedOperator][Apply] residual");
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_TRANSPOSE,
                                    nc, lmode, etmp, residual,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
    // Restore used pointer if one was provided ********************************
    const CeedVector_Occa *data = residual->data;
    if (data->used_pointer)
      occaCopyMemToPtr(data->used_pointer,data->d_array,
                       residual->length*sizeof(CeedScalar),
                       NO_OFFSET, NO_PROPS);
  }
  // ***************************************************************************
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return 0;
}
*/

// *****************************************************************************
// * Create an operator
// *****************************************************************************
int CeedOperatorCreate_Occa(CeedOperator op) {
  const Ceed ceed = op->ceed;
  CeedOperator_Occa *impl;
  int ierr;

  dbg("[CeedOperator][Create]");
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  op->data = impl;
  op->Destroy = CeedOperatorDestroy_Occa;
  op->Apply = CeedOperatorApply_Occa;
  return 0;
}
