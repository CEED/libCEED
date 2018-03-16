// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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
#include "ceed-occa.h"
#include <sys/stat.h>

// *****************************************************************************
static int buildKernelForThisQfunction(CeedQFunction qf){
  CeedQFunction_Occa *occa=qf->data;
  const Ceed_Occa *ceed_data=qf->ceed->data;
  assert(ceed_data);
  const occaDevice dev = *ceed_data->device;
  CeedDebug("\033[33m[CeedQFunction][buildKernelForThisQfunction] nc=%d",occa->nc);
  CeedDebug("\033[33m[CeedQFunction][buildKernelForThisQfunction] dim=%d",occa->dim);
  occaProperties pKR = occaCreateProperties();
  occaPropertiesSet(pKR, "defines/NC", occaInt(occa->nc));
  occaPropertiesSet(pKR, "defines/DIM", occaInt(occa->dim));
  occaPropertiesSet(pKR, "defines/TILE_SIZE", occaInt(TILE_SIZE));
  CeedDebug("\033[33m[CeedQFunction][buildKernelForThisQfunction] occaDeviceBuildKernel");
  CeedDebug("\033[33m[CeedQFunction][buildKernelForThisQfunction] oklPath=%s",occa->oklPath);
  CeedDebug("\033[33m[CeedQFunction][buildKernelForThisQfunction] qFunctionName=%s",occa->qFunctionName);
  occa->kQFunctionApply = occaDeviceBuildKernel(dev, occa->oklPath, occa->qFunctionName, pKR);  
  occaPropertiesFree(pKR);
  return 0;
}

// *****************************************************************************
// * localCeedQFunctionApply_Occa
// *****************************************************************************
__attribute__((unused))
static int localCeedQFunctionApply_Occa(CeedQFunction qf,
                                        void *qdata, CeedInt Q,
                                        const CeedScalar *const *u,
                                        CeedScalar *const *v) {
  int ierr;
  CeedDebug("\033[36m[CeedQFunction][Apply] qf->function");
  ierr = qf->function(qf->ctx, qdata, Q, u, v); CeedChk(ierr);
  return 0;
}


// *****************************************************************************
// * Q-functions: Apply, Destroy & Create
// *****************************************************************************
static int CeedQFunctionApply_Occa(CeedQFunction qf, void *qdata, CeedInt Q,
                                   const CeedScalar *const *u,
                                   CeedScalar *const *v) {
  // Number of quadrature points Q must be a multiple of vlength
  assert((Q%qf->vlength)==0);
  CeedDebug("\033[36m[CeedQFunction][Apply]");
  CeedQFunction_Occa *data=qf->data;
  const Ceed_Occa *ceed=qf->ceed->data;
  const CeedInt nc = data->nc, dim = data->dim;
  occaMemory *d_qdata = data->d_qdata;
  const void *qd_base = data->qd;
  const size_t bytes = qf->qdatasize;
  const CeedEvalMode inmode = qf->inmode;
  const CeedEvalMode outmode = qf->outmode;

  // If the kernel has not been built, do it now
  // We were waiting to get the (nc,dim) pushed to the structure
  // to pass them to the kernels as OCCA properties
  if (!data->ready){
    data->ready=true;
    buildKernelForThisQfunction(qf);
    data->d_u = occaDeviceMalloc(*ceed->device,(Q+Q*nc*(dim+1))*bytes,NULL,NO_PROPS);
    data->d_v = occaDeviceMalloc(*ceed->device,Q*nc*dim*bytes,NULL,NO_PROPS);
  }

  // Context
  //CeedDebug("\033[36;1m[CeedQFunction][Apply] Context ssize=%d,%d",qf->ctxsize, sizeof(CeedInt));
  // We don't push the context to device
  // Avoid OCCA's "Trying to allocate zero bytes"
  //occaMemory o_ctx = occaDeviceMalloc(ceed->device,qf->ctxsize>0?qf->ctxsize:32,NULL,NO_PROPS);
  //if (qf->ctxsize>0) occaCopyPtrToMem(o_ctx,qf->ctx,qf->ctxsize,0,NO_PROPS);

  // On devrait pointer directement en d_memory
  CeedInt qoffset = (qdata-qd_base)/bytes;

  CeedDebug("\033[31;1m[CeedQFunction][Apply] o_qdata");
  occaMemory o_qdata;
  // assert(d_qdata); //d_qdata can be NULL, when not coming from operator, like t20
  if (!d_qdata){
    o_qdata = occaDeviceMalloc(*ceed->device,Q*bytes,qdata,NO_PROPS);
    qoffset = 0;
  }else{
    o_qdata = *d_qdata;
  }
  CeedDebug("\033[31;1m[CeedQFunction][Apply] o_u");
  occaMemory o_u = data->d_u;//occaDeviceMalloc(*ceed->device,(Q+Q*nc*(dim+1))*bytes,NULL,NO_PROPS);

  // on remplit
  if (!data->op){ // t20-qfunction to look at WEIGHT or not
    assert(u[0]);
    occaCopyPtrToMem(o_u,u[0],Q*bytes,0,NO_PROPS);
  }else{ // CeedQFunctionApply via CeedOperatorApply
    if (inmode & CEED_EVAL_INTERP)
      occaCopyPtrToMem(o_u, u[0],Q*nc*bytes,0,NO_PROPS);
    if (inmode & CEED_EVAL_GRAD)
      occaCopyPtrToMem(o_u, u[1],Q*nc*dim*bytes,Q*nc*bytes,NO_PROPS);
    if (inmode & CEED_EVAL_WEIGHT)
      occaCopyPtrToMem(o_u, u[4],Q*bytes,Q*nc*(dim+1)*bytes,NO_PROPS);
  }
  
  //CeedDebug("\033[36m[CeedQFunction][Apply] o_v");
  occaMemory o_v = data->d_v;//occaDeviceMalloc(*ceed->device,Q*nc*dim*bytes,NULL,NO_PROPS);
  int rtn=~0;

  CeedDebug("\033[31;1m[CeedQFunction][Apply] occaKernelRun: %s", data->qFunctionName);
  occaKernelRun(data->kQFunctionApply,
                qf->ctx?occaPtr(qf->ctx):occaInt(0),//o_ctx,
                o_qdata,occaInt(qoffset),
                occaInt(Q), o_u, o_v, occaPtr(&rtn));    
  CeedDebug("\033[31;1m[CeedQFunction][Apply] kernel ran");

  if (rtn!=0){
    CeedDebug("\033[31;1m[CeedQFunction][Apply] return code !=0");
    return CeedError(NULL, 1, "Return code !=0");
  }
  
  if (outmode==CEED_EVAL_NONE){
    //localCeedQFunctionApply_Occa(qf,qdata,Q,u,v);
//#warning localCeedQFunctionApply
    if (!data->op) // t20-qfunction needs this
      occaCopyMemToPtr(qdata,o_qdata,Q*bytes,NO_OFFSET,NO_PROPS);
    //for(int i=0;i<Q;i+=1) printf("\n[CeedQFunctionApply_Occa]  %g",((double*)qdata)[i]);
  }

  if (outmode==CEED_EVAL_INTERP){
    //localCeedQFunctionApply_Occa(qf,qdata,Q,u,v);
//#warning localCeedQFunctionApply
    occaCopyMemToPtr(v[0],o_v,Q*nc*dim*bytes,NO_OFFSET,NO_PROPS);
    //for(int i=0;i<Q*nc*dim;i+=1) printf("\n[CeedQFunctionApply_Occa]  %g",((double*)qdata)[i]);
  }
  
  assert(outmode==CEED_EVAL_NONE || outmode==CEED_EVAL_INTERP);

  //CeedDebug("\033[36;1m[CeedQFunction][Apply] done");
  //occaMemoryFree(o_qdata);
  //occaMemoryFree(o_ctx);
  //occaMemoryFree(o_u);
  //occaMemoryFree(o_v);
  return 0;
}

// *****************************************************************************
static int CeedQFunctionDestroy_Occa(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Occa *occa=qf->data;
  assert(occa);
  free(occa->oklPath);
  CeedDebug("\033[36m[CeedQFunction][Destroy]");
  occaMemoryFree(occa->d_u);
  occaMemoryFree(occa->d_v);
  occaKernelFree(occa->kQFunctionApply);
  ierr = CeedFree(&occa); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
//struct CeedQFunction_private {
//  Ceed ceed;
//  int (*Apply)(CeedQFunction,void*,CeedInt,const CeedScalar*const*,CeedScalar *const*);
//  int (*Destroy)(CeedQFunction);
//  CeedInt vlength; // Number of quadrature points must be padded to a multiple of vlength
//  CeedInt nfields;
//  size_t qdatasize; // Number of bytes of qdata per quadrature point
//  CeedEvalMode inmode, outmode;
//  int (*function)(void *, void *, CeedInt, const CeedScalar *const *,CeedScalar *const *);
//  const char *focca;
//  void *ctx;      // user context for function 
//  size_t ctxsize; // size of user context; may be used to copy to a device
//  void *data;     // backend data
//};
// *****************************************************************************
int CeedQFunctionCreate_Occa(CeedQFunction qf) {  
  CeedQFunction_Occa *data;
  int ierr = CeedCalloc(1,&data); CeedChk(ierr);
  // Populate the CeedQFunction structure
  qf->Apply = CeedQFunctionApply_Occa;
  qf->Destroy = CeedQFunctionDestroy_Occa;
  qf->data = data;
  // Fill CeedQFunction_Occa struct
  data->op = false;
  data->ready = false;
  data->nc = 1;
  data->dim = 1;
  data->qd = NULL;
  data->d_qdata = NULL;
  // Locate last ':' character in qf->focca
  //CeedDebug("\033[36;1m[CeedQFunction][Create] qf->focca=%s",qf->focca);
  char *last_colon = strrchr(qf->focca,':');
  if (!last_colon) return EXIT_FAILURE;
  assert(last_colon);
  //CeedDebug("\033[36;1m[CeedQFunction][Create] last_colon=%s",last_colon);
  // Focus on the function name
  data->qFunctionName = last_colon+1;
  assert(data->qFunctionName);
  //CeedDebug("\033[36;1m[CeedQFunction][Create] qFunctionName=%s",occa->qFunctionName);
  // Now extract filename
  data->oklPath=calloc(4096,sizeof(char));
  const size_t oklPathLen = last_colon - qf->focca;
  memcpy(data->oklPath,qf->focca,oklPathLen);
  data->oklPath[oklPathLen]='\0';
  strcpy(&data->oklPath[oklPathLen - 2],".okl");
  CeedDebug("\033[36;1m[CeedQFunction][Create] qFunctionName=%s",data->qFunctionName);
  CeedDebug("\033[36;1m[CeedQFunction][Create] filename=%s",data->oklPath);
  // Test if we can get file's status
  struct stat buf;          
  if (stat(data->oklPath, &buf)!=0) return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
