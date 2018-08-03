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

int CeedQFunctionApplyElems_Cuda(CeedQFunction qf, const CeedInt nelem, const CeedInt Q,
    const CeedVector *const u, const CeedVector* v) {
  int ierr;
  CeedQFunction_Cuda *data = (CeedQFunction_Cuda*) qf->data;

  const CeedInt cbytes = qf->ctxsize;

  if (!data->ready) {
    data->ready = true;
    ierr = cudaMalloc(&data->d_u, qf->numinputfields * sizeof(CeedScalar*)); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_v, qf->numoutputfields * sizeof(CeedScalar*)); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_uoffsets, qf->numinputfields * sizeof(CeedInt)); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_voffsets, qf->numoutputfields * sizeof(CeedInt)); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_c, cbytes); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_ierr, sizeof(int)); CeedChk(ierr);
    ierr = cudaMemset(data->d_ierr, 0, sizeof(int)); CeedChk(ierr);
  }

  if (cbytes > 0) {
    ierr = cudaMemcpy(data->d_c, qf->ctx, cbytes, cudaMemcpyHostToDevice); CeedChk(ierr);
  }

  const CeedScalar *h_u[qf->numinputfields];
  CeedInt h_uoffsets[qf->numinputfields];
  for (CeedInt i = 0; i < qf->numinputfields; i++) {
    CeedVectorGetArrayRead(u[i], CEED_MEM_DEVICE, h_u + i);
    h_uoffsets[i] = u[i]->length / nelem;
  }
  ierr = cudaMemcpy((void*)data->d_u, h_u, qf->numinputfields * sizeof(CeedScalar*), cudaMemcpyHostToDevice);
  ierr = cudaMemcpy((void*)data->d_uoffsets, h_uoffsets, qf->numinputfields * sizeof(CeedInt), cudaMemcpyDeviceToHost);

  CeedScalar *h_v[qf->numoutputfields];
  CeedInt h_voffsets[qf->numoutputfields];
  for (CeedInt i = 0; i < qf->numoutputfields; i++) {
    CeedVectorGetArray(v[i], CEED_MEM_DEVICE, h_v + i);
    h_voffsets[i] = v[i]->length / nelem;
  }
  ierr = cudaMemcpy((void*)data->d_v, h_v, qf->numoutputfields * sizeof(CeedScalar*), cudaMemcpyDeviceToHost);
  ierr = cudaMemcpy((void*)data->d_voffsets, h_voffsets, qf->numoutputfields * sizeof(CeedInt), cudaMemcpyDeviceToHost);

  ierr = run_cuda(qf->fcuda, 0, 0, nelem, Q, qf->numinputfields, qf->numoutputfields, data->d_c,
      data->d_u, data->d_v, data->d_uoffsets, data->d_voffsets, data->d_ierr); CeedChk(ierr);
  cudaMemcpy(&ierr, data->d_ierr, sizeof(int), cudaMemcpyDeviceToHost); CeedChk(ierr);

  if (cbytes > 0) {
    ierr = cudaMemcpy(qf->ctx, data->d_c, cbytes, cudaMemcpyDeviceToHost); CeedChk(ierr);
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

  if (data->ready) {
    ierr = cudaFree((void*)data->d_u); CeedChk(ierr);
    ierr = cudaFree((void*)data->d_v); CeedChk(ierr);
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

  qf->Apply = CeedQFunctionApply_Cuda;
  qf->Destroy = CeedQFunctionDestroy_Cuda;
  return 0;
}
