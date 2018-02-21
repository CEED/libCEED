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
// * Q-Functions
// *****************************************************************************
typedef struct {
  occaKernel kQFunctionApply;
} CeedQFunction_Occa;


// *****************************************************************************
// * Q-functions: Apply, Destroy & Create
// *****************************************************************************
static int CeedQFunctionApply_Occa(CeedQFunction qf, void *qdata, CeedInt Q,
                                   const CeedScalar *const *u,
                                   CeedScalar *const *v) {
  int ierr;
  //CeedDebug("\033[36m[CeedQFunction][Apply]");
  ierr = qf->function(qf->ctx, qdata, Q, u, v); CeedChk(ierr);
  
  //CeedQFunction_Occa *data=qf->data;
  //CeedScalar
  //const occaMemory ud = *u_impl->array_device;
  //occaMemory vd = *v_impl->array_device;
  
  //occaKernelRun(data->kQFunctionApply, occaInt(Q), u/*d*/, v/*d*/);
  return 0;
}

// *****************************************************************************
static int CeedQFunctionDestroy_Occa(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Occa *occa=qf->data;
  assert(occa);
  ierr = CeedFree(&occa); CeedChk(ierr);
  CeedDebug("\033[36m[CeedQFunction][Destroy]");
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
  
  CeedQFunction_Occa *data;
  int ierr = CeedCalloc(1,&data); CeedChk(ierr);
  qf->data = data;

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
  data->kQFunctionApply = occaDeviceBuildKernel(dev, oklPath, qFunctionName, pKR);
  occaPropertiesFree(pKR);
  // Populate the qf structure
  qf->Apply = CeedQFunctionApply_Occa;
  qf->Destroy = CeedQFunctionDestroy_Occa;
  
  return EXIT_SUCCESS;
}
