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
  int ierr;
  CeedOperator_Occa *impl;
  ierr = CeedOperatorGetData(op, (void *)&impl); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numein+impl->numeout; i++) {
    if (impl->Evecs[i]) {
      ierr = CeedVectorDestroy(&impl->Evecs[i]); CeedChk(ierr);
    }
  }
  ierr = CeedFree(&impl->Evecs); CeedChk(ierr);
  ierr = CeedFree(&impl->Edata); CeedChk(ierr);

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

// *****************************************************************************
// * Dump data
// *****************************************************************************
/*static int CeedOperatorDump_Occa(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedOperator_Occa *data;
  ierr = CeedOperatorGetData(op, (void*)&data); CeedChk(ierr);

  const CeedInt numE = data->numein + data->numeout;
  CeedInt numin, numout, numIO;
  ierr = CeedQFunctionGetNumArgs(qf, &numin, &numout); CeedChk(ierr);
  numIO = numin + numout;
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
  return 0;
}
*/

// *****************************************************************************
// * Setup infields or outfields
// *****************************************************************************
static int CeedOperatorSetupFields_Occa(CeedQFunction qf, CeedOperator op,
                                        bool inOrOut,
                                        CeedVector *fullevecs, CeedVector *evecs,
                                        CeedVector *qvecs, CeedInt starte,
                                        CeedInt numfields, CeedInt Q) {
  CeedInt dim = 1, ierr, ncomp, P;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedQFunction_Occa *qf_data;
  ierr = CeedQFunctionGetData(qf, (void *)&qf_data); CeedChk(ierr);
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
  for (CeedInt i=0; i<numfields; i++) {
    dbg("\t\t[CeedOperator][SetupFields] # %d/%d, \033[7m %d",i,numfields-1, i);
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode); CeedChk(ierr);
    if (emode != CEED_EVAL_WEIGHT) {
      dbg("\t\t[CeedOperator][SetupFields] restriction");
      ierr = CeedOperatorFieldGetElemRestriction(opfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedElemRestrictionCreateVector(Erestrict, NULL, &fullevecs[i+starte]);
      CeedChk(ierr);
    } else {
      dbg("\t\t[CeedOperator][SetupFields] no restriction");
    }
    switch(emode) {
    case CEED_EVAL_NONE:
      dbg("\t\t[CeedOperator][SetupFields] NONE, Q==");
      ierr = CeedQFunctionFieldGetSize(qffields[i], &ncomp);
      CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*ncomp, &qvecs[i]); CeedChk(ierr);
      break; // No action
    case CEED_EVAL_INTERP:
      dbg("\t\t[CeedOperator][SetupFields] INTERP, Q++, qvec[%d]",
          i + starte);
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedQFunctionFieldGetSize(qffields[i], &ncomp);
      CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
      ierr = CeedElemRestrictionGetElementSize(Erestrict, &P);
      CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, P*ncomp, &evecs[i]); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*ncomp, &qvecs[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      dbg("\t\t[CeedOperator][SetupFields] GRAD, Q++, qvec[%d]",
          i + starte);
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedQFunctionFieldGetSize(qffields[i], &ncomp); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
      ncomp /= dim;
      ierr = CeedElemRestrictionGetElementSize(Erestrict, &P);
      CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, P*ncomp, &evecs[i]); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q*ncomp*dim, &qvecs[i]); CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      dbg("\t\t[CeedOperator][SetupFields] WEIGHT, Q== & qvec[%d]",
          i + starte);
      ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
      ierr = CeedVectorCreate(ceed, Q, &qvecs[i]); CeedChk(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                            CEED_VECTOR_NONE, qvecs[i]); CeedChk(ierr);
      assert(starte==0);
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
  int ierr;
  bool setupdone;
  ierr = CeedOperatorGetSetupStatus(op, &setupdone); CeedChk(ierr);
  if (setupdone) return 0;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  CeedOperator_Occa *data;
  ierr = CeedOperatorGetData(op, (void *)&data); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt Q, numinputfields, numoutputfields;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChk(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);

  // ***************************************************************************
  // Count infield and outfield array sizes and evectors CeedChk(ierr);
  data->numein = numinputfields;
  data->numeout = numoutputfields;

  // Allocate ******************************************************************
  const CeedInt numIO = numinputfields + numoutputfields;
  dbg("\t[CeedOperator][Setup] numIO=%d (%d in, %d out)",numIO,
      numinputfields, numoutputfields);

  ierr = CeedCalloc(numinputfields + numoutputfields, &data->Evecs);
  CeedChk(ierr);
  ierr = CeedCalloc(numinputfields + numoutputfields, &data->Edata);
  CeedChk(ierr);

  ierr = CeedCalloc(16, &data->evecsin); CeedChk(ierr);
  ierr = CeedCalloc(16, &data->evecsout); CeedChk(ierr);
  ierr = CeedCalloc(16, &data->qvecsin); CeedChk(ierr);
  ierr = CeedCalloc(16, &data->qvecsout); CeedChk(ierr);

  // Dump data before setting fields
  //dbg("\t[CeedOperator][Setup] Dump data before setting fields: (should be void)");
  //CeedOperatorDump_Occa(op);

  // Set up infield and outfield pointer arrays
  dbg("\t[CeedOperator][Setup] Set up IN fields:");
  // Infields
  ierr = CeedOperatorSetupFields_Occa(qf, op, 0, data->Evecs,
                                      data->evecsin, data->qvecsin, 0,
                                      numinputfields, Q);
  CeedChk(ierr);
  dbg("\t[CeedOperator][Setup] Set up OUT fields:");
  // Outfields
  ierr = CeedOperatorSetupFields_Occa(qf, op, 1, data->Evecs,
                                      data->evecsout, data->qvecsout,
                                      numinputfields, numoutputfields, Q);
  CeedChk(ierr);
  ierr = CeedOperatorSetSetupDone(op); CeedChk(ierr);
  dbg("\t[CeedOperator][Setup] done");
  return 0;
}

// *****************************************************************************
// * Sync CeedVector to Host
// *****************************************************************************
static int SyncToHostPointer(CeedVector vec) {
  // The device copy is not updated in the host array by default.  We may need
  // to rethink memory management in this example, but this provides the
  // expected semantics when using CeedVectorSetArray for the vector that will
  // hold an output quantity.  This should at least be lazy instead of eager
  // and we should do better about avoiding copies.
  int ierr;
  CeedVector_Occa *outvdata;
  ierr = CeedVectorGetData(vec, (void *)&outvdata); CeedChk(ierr);
  if (outvdata->h_array) {
    CeedInt length;
    ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
    occaCopyMemToPtr(outvdata->h_array, outvdata->d_array,
                     length * sizeof(CeedScalar), NO_OFFSET, NO_PROPS);
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
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  dbg("[CeedOperator][Apply]");
  CeedOperator_Occa *data;
  ierr = CeedOperatorGetData(op, (void *)&data); CeedChk(ierr);
  //CeedVector *E = data->Evecs, *D = data->D, outvec;
  CeedInt Q, elemsize, numelements, numinputfields, numoutputfields, ncomp, dim;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChk(ierr);
  ierr = CeedOperatorGetNumElements(op, &numelements); CeedChk(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
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
  // ***************************************************************************
  //dbg("[CeedOperator][Dump] Setup?");
  ierr = CeedOperatorSetup_Occa(op); CeedChk(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  //CeedOperatorDump_Occa(op);

  // Tell CeedQFunction_Occa's structure we are coming from an operator ********
  CeedQFunction_Occa *qfd;
  ierr = CeedQFunctionGetData(qf, (void *)&qfd); CeedChk(ierr);
  qfd->op = op;

  // Input Evecs and Restriction
  for (CeedInt i=0; i<numinputfields; i++) {
    dbg("\n[CeedOperator][Apply] %d/%d Input Evecs:",i,numinputfields-1);
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode & CEED_EVAL_WEIGHT) {
      dbg("[CeedOperator][Apply] No restriction, WEIGHT");
    } else { // Restriction ****************************************************
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChk(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        vec = invec;
      dbg("[CeedOperator][Apply] Restriction: data->Evecs[%d] = Edata[%d]",
          i,i);
      // Restrict
      ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedOperatorFieldGetLMode(opinputfields[i], &lmode); CeedChk(ierr);
      ierr = CeedElemRestrictionApply(Erestrict, CEED_NOTRANSPOSE,
                                      lmode, vec, data->Evecs[i],
                                      request); CeedChk(ierr);
      // Get evec
      ierr = CeedVectorGetArrayRead(data->Evecs[i], CEED_MEM_HOST,
                                    (const CeedScalar **) &data->Edata[i]);
      CeedChk(ierr);
    }
  }
  //dbg("\n[CeedOperator][Apply] Input Evecs and Restriction done, debug:");
  //CeedOperatorDump_Occa(op);

  // Output Evecs
  for (CeedInt i=0; i<numoutputfields; i++) {
    dbg("\n[CeedOperator][Apply] %d/%d Output Evecs:",i,numoutputfields-1);
    dbg("[CeedOperator][Apply] Restriction");
    ierr = CeedVectorGetArray(data->Evecs[i+data->numein], CEED_MEM_HOST,
                              &data->Edata[i + numinputfields]); CeedChk(ierr);
  }
  //dbg("\n[CeedOperator][Apply] Output Evecs done, debug:");
  //CeedOperatorDump_Occa(op);

  // Loop through elements *****************************************************
  dbg("\n[CeedOperator][Apply] Loop through elements");
  for (CeedInt e=0; e<numelements; e++) {
    dbg("\n\t[CeedOperator][Apply] e # %d/%d",e,numelements-1);
    // Input basis apply if needed
    dbg("\t[CeedOperator][Apply] Input basis apply if needed");
    dbg("\t[CeedOperator][Apply] num input fields");
    for (CeedInt i=0; i<numinputfields; i++) {
      dbg("\t\t[CeedOperator][Apply] IN \033[7m %d", i);
      // Get elemsize, emode, ncomp
      ierr = CeedOperatorFieldGetElemRestriction(opinputfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
      CeedChk(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
      CeedChk(ierr);
      ierr = CeedQFunctionFieldGetSize(qfinputfields[i], &ncomp); CeedChk(ierr);
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        dbg("\t\t[CeedOperator][Apply] in NONE, indata[%d] = Edata[%d]",i,i);
        ierr = CeedVectorSetArray(data->qvecsin[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i][e*Q*ncomp]); CeedChk(ierr);
        break;
      case CEED_EVAL_INTERP:
        dbg("\t\t[CeedOperator][Apply] in INTERP, basis, Edata[%d] => qdata[%d] => indata[%d]",
            i,i,i);
        ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
        ierr = CeedVectorSetArray(data->evecsin[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i][e*elemsize*ncomp]);
        CeedChk(ierr);
        ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE,
                              CEED_EVAL_INTERP, data->evecsin[i],
                              data->qvecsin[i]); CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        dbg("\t\t[CeedOperator][Apply] in GRAD, basis, Edata[%d] => qdata[%d] => indata[%d]",
            i,i,i);
        ierr = CeedOperatorFieldGetBasis(opinputfields[i], &basis); CeedChk(ierr);
        ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
        ncomp /= dim;
        ierr = CeedVectorSetArray(data->evecsin[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i][e*elemsize*ncomp]);
        CeedChk(ierr);
        ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE,
                              CEED_EVAL_GRAD, data->evecsin[i],
                              data->qvecsin[i]); CeedChk(ierr);
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
    for (CeedInt i=0; i<numoutputfields; i++) {
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChk(ierr);
      dbg("\t\t[CeedOperator][Apply] OUT %d",i);
      if (emode == CEED_EVAL_NONE) {
        dbg("\t\t[CeedOperator][Apply] out NONE, Edata[%d] => outdata[%d]",
            i + numinputfields,i);
        ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &ncomp);
        CeedChk(ierr);
        ierr = CeedVectorSetArray(data->qvecsout[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i + numinputfields][e*Q*ncomp]);
        CeedChk(ierr);
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
    //CeedOperatorDump_Occa(op);

    // Q function
    dbg("\t[CeedOperator][Apply] Q function apply");
    ierr = CeedQFunctionApply(qf, Q, data->qvecsin, data->qvecsout); CeedChk(ierr);

    // Output basis apply if needed
    dbg("\t[CeedOperator][Apply] Output basis apply if needed");
    for (CeedInt i=0; i<numoutputfields; i++) {
      // Get elemsize, emode, ncomp
      ierr = CeedOperatorFieldGetElemRestriction(opoutputfields[i], &Erestrict);
      CeedChk(ierr);
      ierr = CeedElemRestrictionGetElementSize(Erestrict, &elemsize);
      CeedChk(ierr);
      ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
      CeedChk(ierr);
      ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &ncomp); CeedChk(ierr);
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        break; // No action
      case CEED_EVAL_INTERP:
        ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
        CeedChk(ierr);
        ierr = CeedVectorSetArray(data->evecsout[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i + numinputfields][e*elemsize*ncomp]);
        CeedChk(ierr);
        ierr = CeedBasisApply(basis, 1, CEED_TRANSPOSE,
                              CEED_EVAL_INTERP, data->qvecsout[i],
                              data->evecsout[i]); CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedOperatorFieldGetBasis(opoutputfields[i], &basis);
        CeedChk(ierr);
        ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
        ncomp /= dim;
        ierr = CeedVectorSetArray(data->evecsout[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &data->Edata[i + numinputfields][e*elemsize*ncomp]);
        CeedChk(ierr);
        ierr = CeedBasisApply(basis, 1, CEED_TRANSPOSE,
                              CEED_EVAL_GRAD, data->qvecsout[i],
                              data->evecsout[i]); CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT: break; // Should not occur
      case CEED_EVAL_DIV: break; // Not implemented
      case CEED_EVAL_CURL: break; // Not implemented
      }
    }
  } // numelements

  // Zero lvecs
  for (CeedInt i=0; i<numoutputfields; i++) {
    ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE)
      vec = outvec;
    ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
  }

  // Output restriction
  for (CeedInt i=0; i<numoutputfields; i++) {
    // Restore evec
    ierr = CeedVectorRestoreArray(data->Evecs[i+data->numein],
                                  &data->Edata[i + numinputfields]);
    CeedChk(ierr);
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
                                    lmode, data->Evecs[i+data->numein], vec,
                                    request); CeedChk(ierr);
    ierr = SyncToHostPointer(vec); CeedChk(ierr);
  }

  // Restore input arrays
  for (CeedInt i=0; i<numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChk(ierr);
    if (emode & CEED_EVAL_WEIGHT) {
    } else {
      // Restriction
      ierr = CeedVectorRestoreArrayRead(data->Evecs[i],
                                        (const CeedScalar **) &data->Edata[i]);
      CeedChk(ierr);
    }
  }
  return 0;
}

// *****************************************************************************
// * Create an operator
// *****************************************************************************
int CeedOperatorCreate_Occa(CeedOperator op) {
  int ierr;
  CeedOperator_Occa *impl;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

  dbg("[CeedOperator][Create]");
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedOperatorSetData(op, (void *)&impl); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Apply",
                                CeedOperatorApply_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Occa); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Create a composite operator
// *****************************************************************************
int CeedCompositeOperatorCreate_Occa(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement composite operators");
}
