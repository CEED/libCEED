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

#include <ceed-impl.h>
#include <string.h>
#include <stdio.h>
#include "ceed-cuda.h"

int CeedQFunctionApplyElems_Cuda(CeedQFunction qf, const CeedInt Q,
    const CeedVector *const u, const CeedVector* v) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  Ceed_Cuda* ceed_Cuda;
  ierr = CeedGetData(ceed, (void*)&ceed_Cuda); CeedChk(ierr);
  CeedQFunction_Cuda *data;
  ierr = CeedQFunctionGetData(qf, (void*)&data); CeedChk(ierr);
  const int blocksize = ceed_Cuda->optblocksize;

  if (qf->ctxsize > 0) {
    ierr = cudaMemcpy(data->d_c, qf->ctx, qf->ctxsize, cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
  }

  const CeedScalar *h_u[qf->numinputfields];
  for (CeedInt i = 0; i < qf->numinputfields; i++) {
    CeedVectorGetArrayRead(u[i], CEED_MEM_DEVICE, h_u + i);
  }
  ierr = cudaMemcpy((void**)data->d_u, h_u, qf->numinputfields * sizeof(CeedScalar*), cudaMemcpyHostToDevice);
  CeedChk_Cu(ceed, ierr);

  CeedScalar *h_v[qf->numoutputfields];
  for (CeedInt i = 0; i < qf->numoutputfields; i++) {
    CeedVectorGetArray(v[i], CEED_MEM_DEVICE, h_v + i);
  }
  ierr = cudaMemcpy((void*)data->d_v, h_v, qf->numoutputfields * sizeof(CeedScalar*), cudaMemcpyDeviceToHost);
  CeedChk_Cu(ceed, ierr);

  void *args[] = {&data->d_c, (void*)&Q, &data->d_u, &data->d_v};
  ierr = run_kernel(qf->ceed, data->callback, CeedDivUpInt(Q, blocksize), blocksize, args);
  CeedChk(ierr);


  if (qf->ctxsize > 0) {
    ierr = cudaMemcpy(qf->ctx, data->d_c, qf->ctxsize, cudaMemcpyDeviceToHost); CeedChk_Cu(ceed, ierr);
  }

  return 0;
}

static int CeedQFunctionApply_Cuda(CeedQFunction qf, CeedInt Q,
                                   CeedVector *U, CeedVector *V) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_Cuda *data;
  ierr = CeedQFunctionGetData(qf, (void*)&data); CeedChk(ierr);
  Ceed_Cuda *ceed_Cuda;
  ierr = CeedGetData(ceed, (void*)&ceed_Cuda);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  const int blocksize = ceed_Cuda->optblocksize;

  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedVectorGetArrayRead(U[i], CEED_MEM_DEVICE, data->d_u + i);
    CeedChk(ierr);
  }

  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedVectorGetArray(V[i], CEED_MEM_DEVICE, data->d_v + i);
    CeedChk(ierr);
  }

  // void *args[] = {&data->d_c, (void*)&Q, &data->d_u, &data->d_v};
  void* ctx;
  ierr = CeedQFunctionGetContext(qf, &ctx); CeedChk(ierr);
  void *args[] = {&ctx, (void*)&Q, &data->d_u, &data->d_v};
  ierr = run_kernel(ceed, data->callback, CeedDivUpInt(Q, blocksize), blocksize, args);
  CeedChk(ierr);

  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedVectorRestoreArrayRead(U[i], data->d_u + i);
    CeedChk(ierr);
  }

  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedVectorRestoreArray(V[i], data->d_v + i);
    CeedChk(ierr);
  }

  return 0;
}

static int CeedQFunctionDestroy_Cuda(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Cuda *data;
  ierr = CeedQFunctionGetData(qf, (void*)&data); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);

  CeedChk_Cu(ceed, cuModuleUnload(data->module)); 
  ierr = cudaFree((void*)data->d_u); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree((void*)data->d_v); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(data->d_c); CeedChk_Cu(ceed, ierr);

  ierr = CeedFree(&data); CeedChk(ierr);

  return 0;
}

int CeedQFunctionCreate_Cuda(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed); 
  CeedQFunction_Cuda *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  ierr = cudaMalloc((void**)&data->d_u, numinputfields * sizeof(CeedScalar*)); CeedChk_Cu(ceed, ierr);
  ierr = cudaMalloc((void**)&data->d_v, numoutputfields * sizeof(CeedScalar*)); CeedChk_Cu(ceed, ierr);
  size_t ctxsize;
  ierr = CeedQFunctionGetContextSize(qf, &ctxsize); CeedChk(ierr);
  ierr = cudaMalloc(&data->d_c, ctxsize); CeedChk_Cu(ceed, ierr);
  
  const char *funname = strrchr(qf->fcuda, ':') + 1;
  // Including final NUL char
  const int filenamelen = funname - qf->fcuda;
  char filename[filenamelen];
  memcpy(filename, qf->fcuda, filenamelen - 1);
  filename[filenamelen - 1] = '\0';
  FILE *file = fopen(filename, "r");
  if (!file) {
    return CeedError(qf->ceed, 1, "The file %s cannot be read", filename);
  }

  fseek(file, 0, SEEK_END);
  const int contentslen = ftell(file);
  fseek (file, 0, SEEK_SET);
  char *contents;
  ierr = CeedCalloc(contentslen + 1, &contents); CeedChk(ierr);
  fread(contents, 1, contentslen, file);

  ierr = compile(ceed, contents, &data->module, 0); CeedChk(ierr);
  ierr = get_kernel(ceed, data->module, funname, &data->callback); CeedChk(ierr);
  ierr = CeedFree(&contents); CeedChk(ierr);

  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  ierr = CeedQFunctionSetData(qf, (void*)&data); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Cuda); CeedChk(ierr);
  return 0;
}
