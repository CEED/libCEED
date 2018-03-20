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
static int buildKernelForThisQfunction(CeedQFunction qf) {
  CeedQFunction_Occa *occa=qf->data;
  const Ceed_Occa *ceed_data=qf->ceed->data;
  assert(ceed_data);
  const occaDevice dev = *ceed_data->device;
  CeedDebug("\033[33m[CeedQFunction][buildKernelForThisQfunction] nc=%d",
            occa->nc);
  CeedDebug("\033[33m[CeedQFunction][buildKernelForThisQfunction] dim=%d",
            occa->dim);
  occaProperties pKR = occaCreateProperties();
  occaPropertiesSet(pKR, "defines/NC", occaInt(occa->nc));
  occaPropertiesSet(pKR, "defines/DIM", occaInt(occa->dim));
  occaPropertiesSet(pKR, "defines/TILE_SIZE", occaInt(TILE_SIZE));
  CeedDebug("\033[33m[CeedQFunction][buildKernelForThisQfunction] occaDeviceBuildKernel");
  CeedDebug("\033[33m[CeedQFunction][buildKernelForThisQfunction] oklPath=%s",
            occa->oklPath);
  CeedDebug("\033[33m[CeedQFunction][buildKernelForThisQfunction] qFunctionName=%s",
            occa->qFunctionName);
  occa->kQFunctionApply = occaDeviceBuildKernel(dev, occa->oklPath,
                          occa->qFunctionName, pKR);
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
// * Fill function for t20 & operator cases
// *****************************************************************************
static int CeedQFunctionFill20_Occa(occaMemory o_u,
                                    const CeedScalar *const *u, CeedInt size) {
  occaCopyPtrToMem(o_u,u[0],size,0,NO_PROPS);
  return 0;
}
// *****************************************************************************
static int CeedQFunctionFillOp_Occa(occaMemory o_u,
                                    const CeedScalar *const *u,
                                    const CeedEvalMode inmode,
                                    const CeedInt Q, const CeedInt nc,
                                    const CeedInt dim, const size_t bytes) {
  if (inmode & CEED_EVAL_INTERP)
    occaCopyPtrToMem(o_u, u[0],Q*nc*bytes,0,NO_PROPS);
  if (inmode & CEED_EVAL_GRAD)
    occaCopyPtrToMem(o_u, u[1],Q*nc*dim*bytes,Q*nc*bytes,NO_PROPS);
  if (inmode & CEED_EVAL_WEIGHT)
    occaCopyPtrToMem(o_u, u[4],Q*bytes,Q*nc*(dim+1)*bytes,NO_PROPS);
  return 0;
}

// *****************************************************************************
// * Q-functions: Apply, Destroy & Create
// *****************************************************************************
static int CeedQFunctionApply_Occa(CeedQFunction qf, void *qdata, CeedInt Q,
                                   const CeedScalar *const *u,
                                   CeedScalar *const *v) {
  //CeedDebug("\033[36m[CeedQFunction][Apply]");
  int ierr;
  CeedQFunction_Occa *data = qf->data;
  const Ceed_Occa *ceed = qf->ceed->data;
  const CeedInt nc = data->nc, dim = data->dim;
  const occaMemory *d_qdata = data->d_qdata;
  const void *qd_base = data->qdata;
  const CeedEvalMode inmode = qf->inmode;
  const CeedEvalMode outmode = qf->outmode;
  const CeedInt bytes = qf->qdatasize;
  const CeedInt qubytes = Q*bytes;
  const CeedInt vbytes = Q*nc*dim*bytes;
  const CeedInt qoffset = d_qdata?(qdata-qd_base)/bytes:0;
  const CeedInt ready =  data->ready;
  assert((Q%qf->vlength)==0); // Q must be a multiple of vlength

  if (!ready) { // If the kernel has not been built, do it now
    data->ready=true;
    buildKernelForThisQfunction(qf);
    data->d_u = occaDeviceMalloc(*ceed->device,(Q+Q*nc*(dim+1))*bytes,NULL,
                                 NO_PROPS);
    data->d_v = occaDeviceMalloc(*ceed->device,vbytes,NULL,NO_PROPS);
  }
  const occaMemory d_u = data->d_u;
  const occaMemory d_v = data->d_v;
  const occaMemory d_q = d_qdata?*d_qdata: // d_qdata can be NULL, like test t20
                         occaDeviceMalloc(*ceed->device,Q*bytes,qdata,NO_PROPS);
  if (!data->op) {
    //eedDebug("\033[31;1m[CeedQFunction][Apply] CeedQFunctionFill20_Occa");
    CeedQFunctionFill20_Occa(d_u,u,qubytes);
  }
  else {
    //CeedDebug("\033[31;1m[CeedQFunction][Apply] CeedQFunctionFillOp_Occa");
    CeedQFunctionFillOp_Occa(d_u,u,inmode,Q,nc,dim,bytes);
  }
  //CeedDebug("\033[31;1m[CeedQFunction][Apply] run: %s", data->qFunctionName);
  occaKernelRun(data->kQFunctionApply,
                qf->ctx?occaPtr(qf->ctx):occaInt(0),
                d_q,occaInt(qoffset),
                occaInt(Q), d_u, d_v, occaPtr(&ierr));
  CeedChk(ierr);
  if (outmode==CEED_EVAL_NONE && !data->op)
    occaCopyMemToPtr(qdata,d_q,qubytes,NO_OFFSET,NO_PROPS);
  if (outmode==CEED_EVAL_INTERP)
    occaCopyMemToPtr(*v,d_v,vbytes,NO_OFFSET,NO_PROPS);
  assert(outmode==CEED_EVAL_NONE || outmode==CEED_EVAL_INTERP);
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
// * CeedQFunctionCreate_Occa
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
  data->nc = data->dim = 1;
  data->qdata = NULL;
  data->d_qdata = NULL;
  // Locate last ':' character in qf->focca
  char *last_colon = strrchr(qf->focca,':');
  if (!last_colon) return EXIT_FAILURE;
  assert(last_colon);
  // Focus on the function name
  data->qFunctionName = last_colon+1;
  assert(data->qFunctionName);
  // Now extract filename
  data->oklPath=calloc(4096,sizeof(char));
  const size_t oklPathLen = last_colon - qf->focca;
  memcpy(data->oklPath,qf->focca,oklPathLen);
  data->oklPath[oklPathLen]='\0';
  strcpy(&data->oklPath[oklPathLen - 2],".okl");
  CeedDebug("\033[36;1m[CeedQFunction][Create] qFunctionName=%s",
            data->qFunctionName);
  CeedDebug("\033[36;1m[CeedQFunction][Create] filename=%s",data->oklPath);
  // Test if we can get file's status
  struct stat buf;
  if (stat(data->oklPath, &buf)!=0) return EXIT_FAILURE;
  return 0;
}
