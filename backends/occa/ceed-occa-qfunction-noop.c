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
#define CEED_DEBUG_COLOR 177
#include "ceed-occa.h"

// *****************************************************************************
// * Alloc function for no-operator case
// *****************************************************************************
int CeedQFunctionAllocNoOpIn_Occa(CeedQFunction qf, CeedInt Q,
                                  CeedInt *idx_p,
                                  CeedInt *iOf7) {
  int ierr;
  CeedInt idx = 0;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_Occa *data;
  ierr = CeedQFunctionGetData(qf, (void *)&data); CeedChk(ierr);
  Ceed_Occa *ceed_data;
  ierr = CeedGetData(ceed, (void *)&ceed_data); CeedChk(ierr);
  const occaDevice device = ceed_data->device;
  int nIn;
  ierr = CeedQFunctionGetNumArgs(qf, &nIn, NULL); CeedChk(ierr);
  assert(nIn<N_MAX_IDX);
  size_t cbytes;
  ierr = CeedQFunctionGetContextSize(qf, &cbytes); CeedChk(ierr);
  const CeedInt bytes = sizeof(CeedScalar);
  const CeedInt dim = 1; // !?
  // ***************************************************************************
  dbg("[CeedQFunction][AllocNoOpIn] nIn=%d",nIn);
  CeedQFunctionField *inputfields;
  ierr = CeedQFunctionGetFields(qf, &inputfields, NULL); CeedChk(ierr);
  for (CeedInt i=0; i<nIn; i++) {
    char *name;
    ierr = CeedQFunctionFieldGetName(inputfields[i], &name); CeedChk(ierr);
    CeedInt ncomp;
    ierr = CeedQFunctionFieldGetSize(inputfields[i], &ncomp); CeedChk(ierr);
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(inputfields[i], &emode); CeedChk(ierr);
    switch(emode) {
    case CEED_EVAL_INTERP:
      dbg("\t[CeedQFunction][AllocOpIn] \"%s\" > INTERP (%d)", name,Q*ncomp);
      iOf7[idx+1]=iOf7[idx]+Q*ncomp;
      idx+=1;
      break;
    case CEED_EVAL_GRAD:
      ncomp /= dim;
      dbg("\t[CeedQFunction][AllocOpIn] \"%s\" > GRAD (%d)",name,Q*ncomp*dim);
      iOf7[idx+1]=iOf7[idx]+Q*ncomp*dim;;
      idx+=1;
      break;
    case CEED_EVAL_NONE:
      dbg("\t[CeedQFunction][AllocOpIn] \"%s\" > NONE",name);
      iOf7[idx+1]=iOf7[idx]+Q*ncomp;
      idx+=1;
      break;
    case CEED_EVAL_WEIGHT:
      dbg("\t[CeedQFunction][AllocOpIn] \"%s\" > WEIGHT (%d)",name,Q);
      iOf7[idx+1]=iOf7[idx]+Q;
      idx+=1;
      break;
    case CEED_EVAL_DIV:
      break; // Not implimented
    case CEED_EVAL_CURL:
      break; // Not implimented
    }
  }
  for (CeedInt i=0; i<idx+1; i++) {
    dbg("\t[CeedQFunction][AllocNoOpIn] iOf7[%d]=%d", i,iOf7[i]);
  }
  assert(idx==nIn);
  const CeedInt ilen=iOf7[idx];
  *idx_p = idx;

  dbg("[CeedQFunction][AllocNoOpIn] ilen=%d", ilen);
  dbg("[CeedQFunction][AllocNoOpIn] Alloc IN of %d", ilen);
  // INPUT+IDX alloc ***********************************************************
  data->o_indata = occaDeviceMalloc(device, ilen*bytes, NULL, NO_PROPS);
  data->d_idx = occaDeviceMalloc(device, idx*sizeof(int), NULL, NO_PROPS);
  occaCopyPtrToMem(data->d_idx,iOf7,idx*sizeof(int),0,NO_PROPS);
  // CTX alloc *****************************************************************
  data->d_ctx = occaDeviceMalloc(device,cbytes>0?cbytes:32,NULL,NO_PROPS);
  return 0;
}

// *****************************************************************************
// * Alloc function for no-operator case
// *****************************************************************************
int CeedQFunctionAllocNoOpOut_Occa(CeedQFunction qf, CeedInt Q,
                                   CeedInt *odx_p,
                                   CeedInt *oOf7) {
  int ierr;
  CeedInt odx = 0;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_Occa *data;
  ierr = CeedQFunctionGetData(qf, (void *)&data); CeedChk(ierr);
  Ceed_Occa *ceed_data;
  ierr = CeedGetData(ceed, (void *)&ceed_data); CeedChk(ierr);
  const occaDevice device = ceed_data->device;
  const CeedInt bytes = sizeof(CeedScalar);
  const CeedInt dim = 1; // !?
  CeedInt nOut;
  ierr = CeedQFunctionGetNumArgs(qf, NULL, &nOut); CeedChk(ierr);
  assert(nOut<N_MAX_IDX);
  dbg("[CeedQFunction][AllocNoOpOut] nOut=%d",nOut);
  CeedQFunctionField *outputfields;
  ierr = CeedQFunctionGetFields(qf, NULL, &outputfields); CeedChk(ierr);
  for (CeedInt i=0; i<nOut; i++) {
    char *name;
    ierr = CeedQFunctionFieldGetName(outputfields[i], &name); CeedChk(ierr);
    CeedInt ncomp;
    ierr = CeedQFunctionFieldGetSize(outputfields[i], &ncomp); CeedChk(ierr);
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(outputfields[i], &emode); CeedChk(ierr);
    switch(emode) {
    case CEED_EVAL_NONE:
      dbg("[CeedQFunction][AllocOpOut] out \"%s\" NONE (%d)",name,Q*ncomp);
      oOf7[odx+1]=oOf7[odx]+Q*ncomp;
      odx+=1;
      break;
    case CEED_EVAL_INTERP:
      dbg("\t[CeedQFunction][AllocOpOut \"%s\" > INTERP (%d)", name,Q*ncomp);
      oOf7[odx+1]=oOf7[odx]+Q*ncomp;
      odx+=1;
      break;
    case CEED_EVAL_GRAD:
      ncomp /= dim;
      dbg("\t[CeedQFunction][AllocOpOut] \"%s\" > GRAD (%d)",name,Q*ncomp*dim);
      oOf7[odx+1]=oOf7[odx]+Q*ncomp*dim;
      odx+=1;
      break;
    case CEED_EVAL_WEIGHT:
      break; // Should not occur
    case CEED_EVAL_DIV:
      break; // Not implimented
    case CEED_EVAL_CURL:
      break; // Not implimented
    }
  }
  assert(odx==nOut);
  *odx_p = odx;
  const CeedInt olen=oOf7[odx];
  dbg("[CeedQFunction][AllocNoOpOut] olen=%d", olen);
  // OUTPUT alloc **********************************************************
  if (olen>0) {
    data->o_outdata = occaDeviceMalloc(device, olen*bytes, NULL, NO_PROPS);
    data->d_odx = occaDeviceMalloc(device, odx*sizeof(int), NULL, NO_PROPS);
    occaCopyPtrToMem(data->d_odx,oOf7,odx*sizeof(int),0,NO_PROPS);
  }
  return 0;
}

// *****************************************************************************
// * Fill function for no-operator case
// *****************************************************************************
int CeedQFunctionFillNoOp_Occa(CeedQFunction qf, CeedInt Q,
                               occaMemory d_indata,
                               CeedInt *iOf7,
                               CeedInt *oOf7,
                               const CeedScalar *const *in) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedInt nIn;
  ierr = CeedQFunctionGetNumArgs(qf, &nIn, NULL); CeedChk(ierr);
  const CeedInt ilen = iOf7[nIn];
  const CeedInt bytes = sizeof(CeedScalar);
  CeedQFunctionField *inputfields;
  ierr = CeedQFunctionGetFields(qf, &inputfields, NULL); CeedChk(ierr);
  for (CeedInt i=0; i<nIn; i++) {
    CeedInt ncomp;
    ierr = CeedQFunctionFieldGetSize(inputfields[i], &ncomp); CeedChk(ierr);
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(inputfields[i], &emode); CeedChk(ierr);
    const CeedInt length = iOf7[i+1]-iOf7[i];
    switch (emode) {
    case CEED_EVAL_INTERP:
      dbg("[CeedQFunction][FillNoOp] INTERP ilen=%d:%d", ilen, Q*ncomp);
      dbg("[CeedQFunction][FillNoOp] INTERP iOf7[%d]=%d", i,iOf7[i]);
      assert(length==Q*ncomp);
      occaCopyPtrToMem(d_indata,in[i],length*bytes,iOf7[i]*bytes,NO_PROPS);
      break;
    case CEED_EVAL_GRAD:
      dbg("[CeedQFunction][FillNoOp] GRAD ilen=%d:%d", ilen, Q*ncomp);
      dbg("[CeedQFunction][FillNoOp] GRAD iOf7[%d]=%d", i,iOf7[i]);
      assert(length==Q*ncomp);
      occaCopyPtrToMem(d_indata,in[i],length*bytes,iOf7[i]*bytes,NO_PROPS);
      break;
    case CEED_EVAL_WEIGHT:
      dbg("[CeedQFunction][FillNoOp] WEIGHT ilen=%d:%d", ilen, Q);
      dbg("[CeedQFunction][FillNoOp] WEIGHT iOf7[%d]=%d", i,iOf7[i]);
      assert(length==Q);
      occaCopyPtrToMem(d_indata,in[i],length*bytes,iOf7[i]*bytes,NO_PROPS);
      break;
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_CURL:
      break; // Not implimented
    case CEED_EVAL_DIV:
      break; // Not implimented
    }
  }
  return 0;
}
