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
// * functions for the 'no-operator' case
// *****************************************************************************
int CeedQFunctionAllocNoOpIn_Occa(CeedQFunction, CeedInt, CeedInt *, CeedInt *);
int CeedQFunctionAllocNoOpOut_Occa(CeedQFunction, CeedInt, CeedInt *,
                                   CeedInt *) ;
int CeedQFunctionFillNoOp_Occa(CeedQFunction, CeedInt, occaMemory,
                               CeedInt *, CeedInt *, const CeedScalar *const *);

// *****************************************************************************
// * functions for the 'operator' case
// *****************************************************************************
int CeedQFunctionAllocOpIn_Occa(CeedQFunction, CeedInt, CeedInt *, CeedInt *);
int CeedQFunctionAllocOpOut_Occa(CeedQFunction, CeedInt, CeedInt *, CeedInt *) ;
int CeedQFunctionFillOp_Occa(CeedQFunction, CeedInt, occaMemory,
                             CeedInt *, CeedInt *, const CeedScalar *const *);

// *****************************************************************************
// * buildKernel
// *****************************************************************************
static int CeedQFunctionBuildKernel(CeedQFunction qf, const CeedInt Q) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_Occa *data;
  ierr = CeedQFunctionGetData(qf, (void *)&data); CeedChk(ierr);
  Ceed_Occa *ceed_data;
  ierr = CeedGetData(ceed, (void *)&ceed_data); CeedChk(ierr);
  const bool ocl = ceed_data->ocl;
  assert(ceed_data);
  const occaDevice dev = ceed_data->device;
  dbg("[CeedQFunction][BuildKernel] nc=%d",data->nc);
  dbg("[CeedQFunction][BuildKernel] dim=%d",data->dim);
  dbg("[CeedQFunction][BuildKernel] nelem=%d",data->nelem);
  dbg("[CeedQFunction][BuildKernel] elemsize=%d",data->elemsize);
  occaProperties pKR = occaCreateProperties();
  occaPropertiesSet(pKR, "defines/NC", occaInt(data->nc));
  occaPropertiesSet(pKR, "defines/DIM", occaInt(data->dim));
  occaPropertiesSet(pKR, "defines/epsilon", occaDouble(1.e-14));
  // OpenCL check for this requirement
  const CeedInt q_tile_size = (Q>TILE_SIZE)?TILE_SIZE:Q;
  // OCCA+MacOS implementation need that for now
  const CeedInt tile_size = ocl?1:q_tile_size;
  occaPropertiesSet(pKR, "defines/TILE_SIZE", occaInt(tile_size));
  dbg("[CeedQFunction][BuildKernel] occaDeviceBuildKernel");
  dbg("[CeedQFunction][BuildKernel] oklPath=%s",data->oklPath);
  dbg("[CeedQFunction][BuildKernel] name=%s",data->qFunctionName);
  data->kQFunctionApply =
    occaDeviceBuildKernel(dev, data->oklPath, data->qFunctionName, pKR);
  occaFree(pKR);
  return 0;
}

// *****************************************************************************
// * Q-functions: Apply, Destroy & Create
// *****************************************************************************
// * CEED_EVAL_NONE, no action
// * CEED_EVAL_INTERP: Q*ncomp*nelem
// * CEED_EVAL_GRAD: Q*ncomp*dim*nelem
// * CEED_EVAL_WEIGHT: Q
// *****************************************************************************
static int CeedQFunctionApply_Occa(CeedQFunction qf, CeedInt Q,
                                   CeedVector *In, CeedVector *Out) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  dbg("[CeedQFunction][Apply]");
  CeedQFunction_Occa *data;
  ierr = CeedQFunctionGetData(qf, (void *)&data); CeedChk(ierr);
  const bool from_operator_apply = data->op;
  //Ceed_Occa *ceed_data = qf->ceed->data;
  //const occaDevice device = ceed_data->device;
  const CeedInt bytes = sizeof(CeedScalar);
  const CeedInt ready =  data->ready;
  size_t cbytes;
  CeedInt vlength;
  ierr = CeedQFunctionGetContextSize(qf, &cbytes); CeedChk(ierr);
  ierr = CeedQFunctionGetVectorLength(qf, &vlength); CeedChk(ierr);
  assert((Q%vlength)==0); // Q must be a multiple of vlength
  const CeedInt nelem = 1; // !?
  CeedInt nIn, nOut;
  ierr = CeedQFunctionGetNumArgs(qf, &nIn, &nOut); CeedChk(ierr);
  const CeedScalar *in[16];
  CeedScalar *out[16];
  for (int i = 0; i < nIn; i++) {
    ierr = CeedVectorGetArrayRead(In[i], CEED_MEM_HOST, &in[i]); CeedChk(ierr);
  }
  for (int i = 0; i < nOut; i++) {
    ierr = CeedVectorGetArray(Out[i], CEED_MEM_HOST, &out[i]); CeedChk(ierr);
  }
  // ***************************************************************************
  if (!ready) { // If the kernel has not been built, do it now
    data->ready=true;
    CeedQFunctionBuildKernel(qf,Q);
    if (!from_operator_apply) { // like coming directly from t20-qfunction
      dbg("[CeedQFunction][Apply] NO operator_setup");
      CeedQFunctionAllocNoOpIn_Occa(qf,Q,&data->idx,data->iOf7);
      CeedQFunctionAllocNoOpOut_Occa(qf,Q,&data->odx,data->oOf7);
    } else { // coming from operator_apply
      CeedQFunctionAllocOpIn_Occa(qf,Q,&data->idx,data->iOf7);
      CeedQFunctionAllocOpOut_Occa(qf,Q,&data->odx,data->oOf7);
    }
  }
  const occaMemory d_indata = data->o_indata;
  const occaMemory d_outdata = data->o_outdata;
  const occaMemory d_ctx = data->d_ctx;
  const occaMemory d_idx = data->d_idx;
  const occaMemory d_odx = data->d_odx;
  // ***************************************************************************
  if (!from_operator_apply) {
    CeedQFunctionFillNoOp_Occa(qf,Q,d_indata,data->iOf7,data->oOf7,in);
  } else {
    dbg("[CeedQFunction][Apply] Operator setup, filling");
    CeedQFunctionFillOp_Occa(qf,Q,d_indata,data->iOf7,data->oOf7,in);
  }

  // ***************************************************************************
  void *ctx;
  if (cbytes>0) {
    ierr = CeedQFunctionGetInnerContext(qf, &ctx); CeedChk(ierr);
    occaCopyPtrToMem(d_ctx,ctx,cbytes,0,NO_PROPS);
  }

  // ***************************************************************************
  dbg("[CeedQFunction][Apply] occaKernelRun");
  occaKernelRun(data->kQFunctionApply,
                d_ctx, occaInt(Q),
                d_idx, d_odx,
                d_indata, d_outdata);

  // ***************************************************************************
  if (cbytes>0) {
    occaCopyMemToPtr(ctx,d_ctx,cbytes,0,NO_PROPS);
  }

  // ***************************************************************************
  CeedQFunctionField *outputfields;
  ierr = CeedQFunctionGetFields(qf, NULL, &outputfields); CeedChk(ierr);
  for (CeedInt i=0; i<nOut; i++) {
    char *name;
    ierr = CeedQFunctionFieldGetName(outputfields[i], &name); CeedChk(ierr);
    CeedInt ncomp;
    ierr = CeedQFunctionFieldGetSize(outputfields[i], &ncomp);
    CeedChk(ierr);
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(outputfields[i], &emode); CeedChk(ierr);
    const CeedInt dim = data->dim;
    switch (emode) {
    case CEED_EVAL_NONE:
      dbg("[CeedQFunction][Apply] out \"%s\" NONE",name);
      occaCopyMemToPtr(out[i],d_outdata,Q*ncomp*nelem*bytes,data->oOf7[i]*bytes,
                       NO_PROPS);
      break;
    case CEED_EVAL_INTERP:
      dbg("[CeedQFunction][Apply] out \"%s\" INTERP",name);
      occaCopyMemToPtr(out[i],d_outdata,Q*ncomp*nelem*bytes,data->oOf7[i]*bytes,
                       NO_PROPS);
      break;
    case CEED_EVAL_GRAD:
      dbg("[CeedQFunction][Apply] out \"%s\" GRAD",name);
      ncomp /= dim;
      occaCopyMemToPtr(out[i],d_outdata,Q*ncomp*dim*nelem*bytes,data->oOf7[i]*bytes,
                       NO_PROPS);
      break;
    case CEED_EVAL_WEIGHT:
      break; // no action
    case CEED_EVAL_CURL:
      break; // Not implimented
    case CEED_EVAL_DIV:
      break; // Not implimented
    }
  }
  for (int i = 0; i < nIn; i++) {
    ierr = CeedVectorRestoreArrayRead(In[i], &in[i]); CeedChk(ierr);
  }
  for (int i = 0; i < nOut; i++) {
    ierr = CeedVectorRestoreArray(Out[i], &out[i]); CeedChk(ierr);
  }
  return 0;
}

// *****************************************************************************
// * CeedQFunctionDestroy_Occa
// *****************************************************************************
static int CeedQFunctionDestroy_Occa(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_Occa *data;
  ierr = CeedQFunctionGetData(qf, (void *)&data); CeedChk(ierr);
  const bool operator_setup = data->op;
  free(data->oklPath);
  dbg("[CeedQFunction][Destroy]");
  occaFree(data->kQFunctionApply);
  if (data->ready) {
    if (!operator_setup) {
      occaFree(data->d_ctx);
      occaFree(data->o_indata);
      occaFree(data->o_outdata);
    }
    //occaFree(data->d_u);
    //occaFree(data->d_v);
  }
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedQFunctionCreate_Occa
// *****************************************************************************
int CeedQFunctionCreate_Occa(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_Occa *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  // Populate the CeedQFunction structure **************************************
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Occa); CeedChk(ierr);
  // Fill CeedQFunction_Occa struct ********************************************
  data->op = false;
  data->ready = false;
  data->nc = data->dim = 1;
  data->nelem = data->elemsize = 1;
  data->e = 0;
  ierr = CeedQFunctionSetData(qf, (void *)&data); CeedChk(ierr);
  // Locate last ':' character in qf->source ************************************
  char *source;
  ierr = CeedQFunctionGetSourcePath(qf, &source); CeedChk(ierr);
  dbg("[CeedQFunction][Create] source path: %s",source);
  const char *last_colon = strrchr(source,':');
  const char *last_dot = strrchr(source,'.');
  if (!last_colon)
    return CeedError(ceed, 1, "Can not find ':' in source path field!");
  if (!last_dot)
    return CeedError(ceed, 1, "Can not find '.' in source path field!");
  // get the function name
  data->qFunctionName = last_colon+1;
  dbg("[CeedQFunction][Create] qFunctionName: %s",data->qFunctionName);
  // extract file base name
  const char *last_slash_pos = strrchr(source,'/');
  // if no slash has been found, revert to source field
  const char *last_slash = last_slash_pos?last_slash_pos+1:source;
  dbg("[CeedQFunction][Create] last_slash: %s",last_slash);
  // extract c_src_file & okl_base_name
  char *c_src_file, *okl_base_name;
  ierr = CeedCalloc(OCCA_PATH_MAX,&okl_base_name); CeedChk(ierr);
  ierr = CeedCalloc(OCCA_PATH_MAX,&c_src_file); CeedChk(ierr);
  memcpy(okl_base_name,last_slash,last_dot-last_slash);
  memcpy(c_src_file,source,last_colon-source);
  dbg("[CeedQFunction][Create] c_src_file: %s",c_src_file);
  dbg("[CeedQFunction][Create] okl_base_name: %s",okl_base_name);
  // Now fetch OKL filename ****************************************************
  ierr = CeedOklPath_Occa(ceed,c_src_file, okl_base_name, &data->oklPath);
  CeedChk(ierr);
  // free **********************************************************************
  ierr = CeedFree(&okl_base_name); CeedChk(ierr);
  ierr = CeedFree(&c_src_file); CeedChk(ierr);
  return 0;
}
