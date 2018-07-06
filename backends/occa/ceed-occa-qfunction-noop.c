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
  CeedInt idx = 0;
  const Ceed ceed = qf->ceed;
  CeedQFunction_Occa *data = qf->data;
  Ceed_Occa *ceed_data = qf->ceed->data;
  const occaDevice device = ceed_data->device;
  const int nIn = qf->numinputfields; assert(nIn<N_MAX_IDX);
  const CeedInt cbytes = qf->ctxsize;
  const CeedInt bytes = sizeof(CeedScalar);
  const CeedInt dim = 1; // !?
  // ***************************************************************************
  dbg("[CeedQFunction][AllocNoOpIn] nIn=%d",nIn);
  for (CeedInt i=0; i<nIn; i++) {
    const CeedEvalMode emode = qf->inputfields[i].emode;
    const char *name = qf->inputfields[i].fieldname;
    const CeedInt ncomp = qf->inputfields[i].ncomp;
    switch(emode) {
    case CEED_EVAL_INTERP:
      dbg("\t[CeedQFunction][AllocOpIn] \"%s\" > INTERP (%d)", name,Q*ncomp);
      iOf7[idx+1]=iOf7[idx]+Q*ncomp;
      idx+=1;
      break;
    case CEED_EVAL_GRAD:
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
  CeedInt odx = 0;
  const Ceed ceed = qf->ceed;
  CeedQFunction_Occa *data = qf->data;
  Ceed_Occa *ceed_data = qf->ceed->data;
  const occaDevice device = ceed_data->device;
  const CeedInt bytes = sizeof(CeedScalar);
  const CeedInt dim = 1; // !?
  const int nOut = qf->numoutputfields; assert(nOut<N_MAX_IDX);
  dbg("[CeedQFunction][AllocNoOpOut] nOut=%d",nOut);
  for (CeedInt i=0; i<nOut; i++) {
    const char *name = qf->outputfields[i].fieldname;
    const CeedInt ncomp = qf->outputfields[i].ncomp;
    const CeedEvalMode emode = qf->outputfields[i].emode;
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
  const Ceed ceed = qf->ceed;
  const int nIn = qf->numinputfields;
  const CeedInt ilen = iOf7[nIn];
  const CeedInt bytes = sizeof(CeedScalar);
  for (CeedInt i=0; i<nIn; i++) {
    const CeedEvalMode emode = qf->inputfields[i].emode;
    const CeedInt ncomp = qf->inputfields[i].ncomp;
    if (emode & CEED_EVAL_INTERP) {
      dbg("[CeedQFunction][FillNoOp] INTERP ilen=%d:%d", ilen, Q*ncomp);
      dbg("[CeedQFunction][FillNoOp] INTERP iOf7[%d]=%d", i,iOf7[i]);
      const CeedInt length = iOf7[i+1]-iOf7[i];
      assert(length==Q*ncomp);
      occaCopyPtrToMem(d_indata,in[i],length*bytes,iOf7[i]*bytes,NO_PROPS);
    }
    if (emode & CEED_EVAL_GRAD) {
      dbg("[CeedQFunction][FillNoOp] GRAD ilen=%d:%d", ilen, Q*ncomp);
      dbg("[CeedQFunction][FillNoOp] GRAD iOf7[%d]=%d", i,iOf7[i]);
      const CeedInt length = iOf7[i+1]-iOf7[i];
      assert(length==Q*ncomp);
      occaCopyPtrToMem(d_indata,in[i],length*bytes,iOf7[i]*bytes,NO_PROPS);
    }
    if (emode & CEED_EVAL_WEIGHT) {
      dbg("[CeedQFunction][FillNoOp] WEIGHT ilen=%d:%d", ilen, Q);
      dbg("[CeedQFunction][FillNoOp] WEIGHT iOf7[%d]=%d", i,iOf7[i]);
      const CeedInt length = iOf7[i+1]-iOf7[i];
      assert(length==Q);
      occaCopyPtrToMem(d_indata,in[i],length*bytes,iOf7[i]*bytes,NO_PROPS);
    }
  }
  return 0;
}
