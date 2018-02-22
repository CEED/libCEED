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
#undef NDEBUG

// *****************************************************************************
// * Q-Functions
// *****************************************************************************
typedef struct {
  occaKernel kQFunctionApply;
} CeedQFunction_Occa;


// *****************************************************************************
__attribute__((unused))
static int localCeedQFunctionApply_Occa(CeedQFunction qf, void *qdata, CeedInt Q,
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
  CeedDebug("\033[36m[CeedQFunction][Apply]");
  CeedQFunction_Occa *occa=qf->data;
  const Ceed_Occa *ceed=qf->ceed->data;
  const size_t qb = Q*sizeof(CeedScalar);
  //CeedDebug("\033[36m[CeedQFunction][Apply] bytes=%d",qb);

  const CeedEvalMode inmode = qf->inmode;
  const CeedEvalMode outmode = qf->outmode;

  // Context
  occaMemory o_ctx = occaDeviceMalloc(ceed->device,sizeof(void*),NULL,NO_PROPS);
   
  // qdata
  //CeedDebug("\033[36m[CeedQFunction][Apply] o_qdata");
  //for(int i=0;i<Q;i+=1) printf("\tQ[%d]=%f\n",i,((double*)qdata)[i]);
  occaMemory o_qdata = occaDeviceMalloc(ceed->device,qb,qdata,NO_PROPS);
  
  //CeedDebug("\033[36m[CeedQFunction][Apply] o_u");
  //for(int i=0;i<Q;i+=1) printf("\tu[0][%d]=%f\n",i,u[0][i]);
  occaMemory o_u = occaDeviceMalloc(ceed->device,5*qb,NULL,NO_PROPS);
  occaCopyPtrToMem(o_u,u[0],qb,0*qb,NO_PROPS);
  if (inmode & CEED_EVAL_GRAD) occaCopyPtrToMem(o_u,u[1],qb,1*qb,NO_PROPS);
  //if (inmode & CEED_EVAL_WEIGHT) occaCopyPtrToMem(o_u,u[4],qb,4*qb,NO_PROPS);
  
  //CeedDebug("\033[36m[CeedQFunction][Apply] o_v");
  occaMemory o_v = occaDeviceMalloc(ceed->device,qb,NULL,NO_PROPS);
  
  int rtn=~0;

  //CeedDebug("\033[36m[CeedQFunction][Apply] occaKernelRun");
  occaKernelRun(occa->kQFunctionApply,o_ctx,o_qdata,occaInt(Q),o_u,o_v,occaPtr(&rtn));

  if (rtn!=0){
    CeedDebug("\033[31;1m[CeedQFunction][Apply] return code !=0");
    return EXIT_FAILURE;
  }
  
  if (outmode==CEED_EVAL_NONE){
    //localCeedQFunctionApply_Occa(qf,qdata,Q,u,v);
    occaCopyMemToPtr(qdata,o_qdata,qb,NO_OFFSET,NO_PROPS);
    //for(int i=0;i<Q;i+=1) printf("\t\t=>%f\n",((double*)qdata)[i]);
  }
  
  if (outmode==CEED_EVAL_INTERP){
    //localCeedQFunctionApply_Occa(qf,qdata,Q,u,v);
    occaCopyMemToPtr(v[0],o_v,qb,NO_OFFSET,NO_PROPS);
  }
  
  assert(outmode==CEED_EVAL_NONE || outmode==CEED_EVAL_INTERP);

  //CeedDebug("\033[36m[CeedQFunction][Apply] done");
  occaMemoryFree(o_u);
  occaMemoryFree(o_v);
  return 0;
}

// *****************************************************************************
static int CeedQFunctionDestroy_Occa(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Occa *occa=qf->data;
  assert(occa);
  CeedDebug("\033[36m[CeedQFunction][Destroy]");
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
  const Ceed_Occa *ceed_data=qf->ceed->data;
  const occaDevice dev = ceed_data->device;
  
  CeedQFunction_Occa *occa;
  int ierr = CeedCalloc(1,&occa); CeedChk(ierr);
  qf->data = occa;

  // Locate last ':' character in qf->focca
  char *last_colon = strrchr(qf->focca,':');
  if (!last_colon) return EXIT_FAILURE;
  assert(last_colon);
  // Focus on the function name
  const char *qFunctionName = last_colon+1;
  assert(qFunctionName);
  // Now extract filename
  char oklPath[4096];
  const size_t oklPathLen = last_colon-qf->focca;
  memcpy(oklPath,qf->focca,oklPathLen);
  oklPath[oklPathLen]='\0';
  strcpy(&oklPath[oklPathLen - 2],".okl");
  CeedDebug("\033[36;1m[CeedQFunction][Create] qFunctionName=%s",qFunctionName);
  CeedDebug("\033[36;1m[CeedQFunction][Create] filename=%s",oklPath);

  // Test if we can get file's status
  struct stat buf;          
  if (stat(oklPath, &buf)!=0) return EXIT_FAILURE;
  
  // Build the kernel for this Q function
  occaProperties pKR = occaCreateProperties();
  occaPropertiesSet(pKR, "defines/TILE_SIZE", occaInt(TILE_SIZE));
  occa->kQFunctionApply = occaDeviceBuildKernel(dev, oklPath, qFunctionName, pKR);
  occaPropertiesFree(pKR);
  // Populate the qf structure
  qf->Apply = CeedQFunctionApply_Occa;
  qf->Destroy = CeedQFunctionDestroy_Occa;
  
  return EXIT_SUCCESS;
}
