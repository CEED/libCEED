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
  const Ceed_Cuda* ceed = (Ceed_Cuda*)qf->ceed->data;
  CeedQFunction_Cuda *data = (CeedQFunction_Cuda*) qf->data;
  const int blocksize = ceed->optblocksize;

  if (qf->ctxsize > 0) {
    ierr = cudaMemcpy(data->d_c, qf->ctx, qf->ctxsize, cudaMemcpyHostToDevice); CeedChk(ierr);
  }

  const CeedScalar *h_u[qf->numinputfields];
  for (CeedInt i = 0; i < qf->numinputfields; i++) {
    CeedVectorGetArrayRead(u[i], CEED_MEM_DEVICE, h_u + i);
  }
  ierr = cudaMemcpy((void**)data->d_u, h_u, qf->numinputfields * sizeof(CeedScalar*), cudaMemcpyHostToDevice);

  CeedScalar *h_v[qf->numoutputfields];
  for (CeedInt i = 0; i < qf->numoutputfields; i++) {
    CeedVectorGetArray(v[i], CEED_MEM_DEVICE, h_v + i);
  }
  ierr = cudaMemcpy((void*)data->d_v, h_v, qf->numoutputfields * sizeof(CeedScalar*), cudaMemcpyDeviceToHost);

  void *args[] = {&data->d_c, (void*)&Q, &data->d_u, &data->d_v};
  ierr = run_kernel(qf->ceed, data->callback, CeedDivUpInt(Q, blocksize), blocksize, args);


  if (qf->ctxsize > 0) {
    ierr = cudaMemcpy(qf->ctx, data->d_c, qf->ctxsize, cudaMemcpyDeviceToHost); CeedChk(ierr);
  }

  return 0;
}

static int CeedQFunctionApply_Cuda(CeedQFunction qf, CeedInt Q,
    const CeedScalar *const *u, CeedScalar *const *v) {
  // TODO Not enough info to implement
  return 1;
}

static int CeedQFunctionDestroy_Cuda(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Cuda *data = (CeedQFunction_Cuda *) qf->data;

  CeedChk_Cu(qf->ceed, cuModuleUnload(data->module)); 
  ierr = cudaFree((void*)data->d_u); CeedChk(ierr);
  ierr = cudaFree((void*)data->d_v); CeedChk(ierr);
  ierr = cudaFree(data->d_c); CeedChk(ierr);

  ierr = CeedFree(&data); CeedChk(ierr);

  return 0;
}

int CeedQFunctionCreate_Cuda(CeedQFunction qf) {
  int ierr;
  
  CeedQFunction_Cuda *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  ierr = cudaMalloc((void**)&data->d_u, qf->numinputfields * sizeof(CeedScalar*)); CeedChk(ierr);
  ierr = cudaMalloc((void**)&data->d_v, qf->numoutputfields * sizeof(CeedScalar*)); CeedChk(ierr);
  ierr = cudaMalloc(&data->d_c, qf->ctxsize); CeedChk(ierr);
  
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

  ierr = compile(qf->ceed, contents, &data->module, 0); CeedChk(ierr);
  ierr = get_kernel(qf->ceed, data->module, funname, &data->callback); CeedChk(ierr);
  ierr = CeedFree(&contents); CeedChk(ierr);

  qf->data = data;
  qf->Apply = CeedQFunctionApply_Cuda;
  qf->Destroy = CeedQFunctionDestroy_Cuda;
  return 0;
}
