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
#include "ceed-cuda.cuh"

/*#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
     if (code != cudaSuccess) 
          {
                  fprintf(stderr,"GPUassert %d: %s %s %d\n", code, cudaGetErrorString(code), file, line);
                        if (abort) exit(code);
                           } else {printf("!\n");}
}*/

static int CeedQFunctionApply_Cuda(CeedQFunction qf, void *qdata, CeedInt Q,
                                  const CeedScalar *const *u,
                                  CeedScalar *const *v) {
  int ierr;
  CeedQFunction_Cuda *data = (CeedQFunction_Cuda*) qf->data;

  const CeedInt nc = data->nc, dim = data->dim;
  const CeedEvalMode outmode = qf->outmode;
  const CeedInt qbytes = Q*qf->qdatasize;
  const CeedInt ubytes = 5;
  const CeedInt ready =  data->ready;
  const CeedInt cbytes = qf->ctxsize;

  if (!ready) {
    data->ready = true;
    ierr = cudaMalloc(&data->d_q, qbytes); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_u, ubytes); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_v, ubytes); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_c, ubytes); CeedChk(ierr);
  }

  ierr = cudaMemcpy(data->d_u, u, ubytes, cudaMemcpyHostToDevice); CeedChk(ierr);
  if (cbytes > 0) {
    ierr = cudaMemcpy(data->d_c, qf->ctx, cbytes, cudaMemcpyHostToDevice); CeedChk(ierr);
  }
  ierr = cudaMemcpy(data->d_q, qdata, qbytes, cudaMemcpyHostToDevice); CeedChk(ierr);

  (qf->cudafunction)<<<1,1>>>(data->d_c, data->d_q, Q, data->d_u, data->d_v, nc, dim);
  CeedChk(cudaGetLastError());

  if (cbytes > 0) {
    ierr = cudaMemcpy(qf->ctx, data->d_c, cbytes, cudaMemcpyDeviceToHost); CeedChk(ierr);
  }

  if (outmode == CEED_EVAL_NONE) {
    ierr = cudaMemcpy(qdata, data->d_q, qbytes, cudaMemcpyDeviceToHost); CeedChk(ierr);
  }

  return 0;
}

static int CeedQFunctionDestroy_Cuda(CeedQFunction qf) {
  return 0;
}

int CeedQFunctionCreate_Cuda(CeedQFunction qf) {
  CeedQFunction_Cuda *data;
  int ierr = CeedCalloc(1,&data); CeedChk(ierr);
  qf->data = data;
  data->ready = false;
  data->nc = data->dim = 1;
  data->nelem = data->elemsize = 1;

  qf->Apply = CeedQFunctionApply_Cuda;
  qf->Destroy = CeedQFunctionDestroy_Cuda;
  return 0;
}
