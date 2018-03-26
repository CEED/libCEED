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
#define CEED_DEBUG_COLOR 14
#include "ceed-occa.h"
#include <sys/stat.h>

// *****************************************************************************
// * buildKernel
// *****************************************************************************
static int CeedQFunctionBuildKernel(CeedQFunction qf) {
  CeedQFunction_Occa *data=qf->data;
  const Ceed_Occa *ceed_data=qf->ceed->data;
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
  const CeedInt tile_size = (data->nelem>TILE_SIZE)?TILE_SIZE:data->nelem;
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
// * Fill function for t20 case, should be removed
// *****************************************************************************
__attribute__((unused))
static int CeedQFunctionFill20_Occa(occaMemory d_u,
                                    occaMemory b_u,
                                    const CeedScalar *const *u,
                                    CeedInt size) {
  occaCopyPtrToMem(d_u,u[0],size,0,NO_PROPS);
  occaCopyPtrToMem(b_u,u[0],size,0,NO_PROPS);
  return 0;
}

// *****************************************************************************
// *  Fill function for the operator case
// *****************************************************************************
__attribute__((unused))
static int CeedQFunctionFillOp_Occa(occaMemory d_u,
                                    const CeedScalar *const *u,
                                    const CeedEvalMode inmode,
                                    const CeedInt Q, const CeedInt nc,
                                    const CeedInt dim, const size_t bytes) {
  if (inmode & CEED_EVAL_INTERP)
    occaCopyPtrToMem(d_u, u[0],Q*nc*bytes,0,NO_PROPS);
  if (inmode & CEED_EVAL_GRAD)
    occaCopyPtrToMem(d_u, u[1],Q*nc*dim*bytes,Q*nc*bytes,NO_PROPS);
  if (inmode & CEED_EVAL_WEIGHT)
    occaCopyPtrToMem(d_u, u[4],Q*bytes,Q*nc*(dim+1)*bytes,NO_PROPS);
  return 0;
}

// *****************************************************************************
// * Q-functions: Apply, Destroy & Create
// *****************************************************************************
static int CeedQFunctionApply_Occa(CeedQFunction qf, void *qdata, CeedInt Q,
                                   const CeedScalar *const *u,
                                   CeedScalar *const *v) {
  //dbg("[CeedQFunction][Apply]");
  CeedQFunction_Occa *data = qf->data;
  const Ceed_Occa *ceed = qf->ceed->data;
  const CeedInt nc = data->nc, dim = data->dim;
  const CeedEvalMode inmode = qf->inmode;
  const CeedEvalMode outmode = qf->outmode;
  const CeedInt bytes = qf->qdatasize;
  const CeedInt qbytes = Q*bytes;
  const CeedInt ubytes = (Q*nc*(dim+2))*bytes;
  const CeedInt vbytes = Q*nc*dim*bytes;
  const CeedInt e = data->e;
  const CeedInt ready =  data->ready;
  assert((Q%qf->vlength)==0); // Q must be a multiple of vlength
  // ***************************************************************************
  if (!ready) { // If the kernel has not been built, do it now
    data->ready=true;
    CeedQFunctionBuildKernel(qf);
    if (!data->op) { // like from t20
      const CeedInt bbytes = Q*nc*(dim+2)*bytes;
      data->d_q = occaDeviceMalloc(ceed->device,qbytes, qdata, NO_PROPS);
      data->b_u = occaDeviceMalloc(ceed->device,bbytes, NULL, NO_PROPS);
      data->b_v = occaDeviceMalloc(ceed->device,bbytes, NULL, NO_PROPS);
    } else {
      /* b_u, b_v come form cee-occa-operator BEu, BEv */
    }
    data->d_u = occaDeviceMalloc(ceed->device,ubytes, NULL, NO_PROPS);
    data->d_v = occaDeviceMalloc(ceed->device,ubytes, NULL, NO_PROPS);
  }
  const occaMemory d_q = data->d_q;
  const occaMemory d_u = data->d_u;
  const occaMemory d_v = data->d_v;
  const occaMemory b_u = data->b_u;
  const occaMemory b_v = data->b_v;
  // ***************************************************************************
  if (!data->op)
    CeedQFunctionFill20_Occa(d_u,b_u,u,qbytes);
  else
    CeedQFunctionFillOp_Occa(d_u,u,inmode,Q,nc,dim,bytes);
  // ***************************************************************************
  occaKernelRun(data->kQFunctionApply,
                qf->ctx?occaPtr(qf->ctx):occaInt(0),
                d_q,occaInt(e),occaInt(Q),
                d_u, b_u,d_v, b_v);
  // ***************************************************************************
  if (outmode==CEED_EVAL_NONE && !data->op)
    occaCopyMemToPtr(qdata,d_q,qbytes,NO_OFFSET,NO_PROPS);
  if (outmode==CEED_EVAL_INTERP)
    occaCopyMemToPtr(*v,d_v,vbytes,NO_OFFSET,NO_PROPS);
  assert(outmode==CEED_EVAL_NONE || outmode==CEED_EVAL_INTERP);
  return 0;
}

// *****************************************************************************
// * CeedQFunctionDestroy_Occa
// *****************************************************************************
static int CeedQFunctionDestroy_Occa(CeedQFunction qf) {
  CeedQFunction_Occa *data=qf->data;
  free(data->oklPath);
  dbg("[CeedQFunction][Destroy]");
  if (data->ready) {
    if (!data->op) occaFree(data->d_q);
    occaFree(data->d_u);
    occaFree(data->d_v);
  }
  int ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedQFunctionCreate_Occa
// *****************************************************************************
int CeedQFunctionCreate_Occa(CeedQFunction qf) {
  CeedQFunction_Occa *data;
  int ierr = CeedCalloc(1,&data); CeedChk(ierr);
  // Populate the CeedQFunction structure **************************************
  qf->Apply = CeedQFunctionApply_Occa;
  qf->Destroy = CeedQFunctionDestroy_Occa;
  qf->data = data;
  // Fill CeedQFunction_Occa struct ********************************************
  data->op = false;
  data->ready = false;
  data->nc = data->dim = 1;
  data->nelem = data->elemsize = 1;
  data->e = 0;
  // Locate last ':' character in qf->focca ************************************
  dbg("[CeedQFunction][Create] focca=%s",qf->focca);
  const char *last_colon = strrchr(qf->focca,':');
  char *last_dot = strrchr(qf->focca,'.');
  if (!last_colon)
    return CeedError(qf->ceed, 1, "Can not find ':' in focca field!");
  if (!last_dot)
    return CeedError(qf->ceed, 1, "Can not find '.' in focca field!");
  // Focus on the function name
  data->qFunctionName = last_colon+1;
  dbg("[CeedQFunction][Create] qFunctionName=%s",
            data->qFunctionName);
  // Now extract filename
  data->oklPath=calloc(4096,sizeof(char));
  const size_t oklPathLen = last_dot - qf->focca;
  memcpy(data->oklPath,qf->focca,oklPathLen);
  data->oklPath[oklPathLen]='\0';
  strcpy(&data->oklPath[oklPathLen],".okl");
  dbg("[CeedQFunction][Create] filename=%s",data->oklPath);
  // Test if we can get file's status ******************************************
  struct stat buf;
  if (stat(data->oklPath, &buf)!=0)
    return CeedError(qf->ceed, 1, "Can not find OKL file!");
  return 0;
}
