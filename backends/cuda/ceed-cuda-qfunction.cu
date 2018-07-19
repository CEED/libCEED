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

int CeedQFunctionApplyElems_Cuda(CeedQFunction qf, CeedVector qdata, const CeedInt Q, const CeedInt nelem,
    const CeedVector u, CeedVector v) {
  int ierr;
  CeedQFunction_Cuda *data = (CeedQFunction_Cuda*) qf->data;

  const CeedInt cbytes = qf->ctxsize;

  if (!data->ready) {
    data->ready = true;
    ierr = cudaMalloc(&data->d_c, cbytes); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_ierr, sizeof(int)); CeedChk(ierr);
    ierr = cudaMemset(data->d_ierr, 0, sizeof(int)); CeedChk(ierr);
  }

  if (cbytes > 0) {
    ierr = cudaMemcpy(data->d_c, qf->ctx, cbytes, cudaMemcpyHostToDevice); CeedChk(ierr);
  }

  const CeedScalar *d_u;
  CeedScalar *d_v;
  char *d_q;
  CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u);
  CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v);
  CeedVectorGetArray(qdata, CEED_MEM_DEVICE, (CeedScalar**)&d_q);

  ierr = run_cuda(qf->fcuda, 0, 0, nelem,
      Q, data->nc, data->dim, qf->qdatasize, qf->inmode, qf->outmode,
      d_u, d_v, data->d_c, d_q, data->d_ierr); CeedChk(ierr);
  cudaMemcpy(&ierr, data->d_ierr, sizeof(int), cudaMemcpyDeviceToHost); CeedChk(ierr);

  if (cbytes > 0) {
    ierr = cudaMemcpy(qf->ctx, data->d_c, cbytes, cudaMemcpyDeviceToHost); CeedChk(ierr);
  }

  return 0;
}

static int CeedQFunctionApply_Cuda(CeedQFunction qf, void *qdata, CeedInt Q,
    const CeedScalar *const *u, CeedScalar *const *v) {

  return 0;
}

static int CeedQFunctionDestroy_Cuda(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Cuda *data = (CeedQFunction_Cuda *) qf->data;

  if (data->ready) {
    ierr = cudaFree(data->d_c); CeedChk(ierr);
    ierr = cudaFree(data->d_ierr); CeedChk(ierr);
  }

  ierr = CeedFree(&data); CeedChk(ierr);

  return 0;
}

int CeedQFunctionCreate_Cuda(CeedQFunction qf) {
  CeedQFunction_Cuda *data;
  int ierr = CeedCalloc(1,&data); CeedChk(ierr);
  qf->data = data;
  data->ready = false;
  data->nc = 1;
  data->dim = 1;

  qf->Apply = CeedQFunctionApply_Cuda;
  qf->Destroy = CeedQFunctionDestroy_Cuda;
  return 0;
}
