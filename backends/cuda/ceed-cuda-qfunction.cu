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

__global__ void apply(const CeedInt nelem, CeedQFunctionCallback qf, const CeedInt Q, const CeedInt nc, const CeedInt dim, const CeedInt qdatasize,
    const CeedEvalMode inmode, const CeedEvalMode outmode, const CeedScalar *u, CeedScalar *v, void *ctx, char *qdata, int *ierr) {
  const int elem = blockIdx.x*blockDim.x + threadIdx.x;

  if (elem >= nelem) return;

  CeedScalar *out[5] = {0, 0, 0, 0, 0};
  const CeedScalar *in[5] = {0, 0, 0, 0, 0};

  if (inmode & CEED_EVAL_WEIGHT) {
    in[4] = u + Q * nelem * nc * (dim + 1);
  }

  if (inmode & CEED_EVAL_INTERP) { in[0] = u + Q * nc * elem; }
  if (inmode & CEED_EVAL_GRAD) { in[1] = u + Q * nc * (nelem + dim * elem); }
  if (outmode & CEED_EVAL_INTERP) { out[0] = v + Q * nc * elem; }
  if (outmode & CEED_EVAL_GRAD) { out[1] = v + Q * nc * (nelem + dim * elem); }
  *ierr = qf(ctx, qdata + elem * Q * qdatasize, Q, in, out);
  printf("%d\n", *ierr);
}

int CeedQFunctionApplyElems_Cuda(CeedQFunction qf, CeedVector qdata, const CeedInt Q, const CeedInt nelem,
    const CeedVector u, CeedVector v) {
  int ierr;
  CeedQFunction_Cuda *data = (CeedQFunction_Cuda*) qf->data;
  const Ceed_Cuda *ceed = (Ceed_Cuda*)qf->ceed->data;

  const CeedInt cbytes = qf->ctxsize;

  if (!data->ready) {
    data->ready = true;
    ierr = cudaMalloc(&data->d_c, cbytes); CeedChk(ierr);
    ierr = cudaMallocManaged(&data->m_ierr, sizeof(int)); CeedChk(ierr);
  }

  const CeedScalar *d_u = ((CeedVector_Cuda *)u->data)->d_array;
  CeedScalar *d_v = ((CeedVector_Cuda *)u->data)->d_array;
  char *d_q = (char *)((CeedVector_Cuda *)qdata->data)->d_array;

  ierr = run1d(ceed, apply, 0, nelem, qf->cudafunction,
      Q, data->nc, data->dim, qf->qdatasize, qf->inmode, qf->outmode,
      d_u, d_v, data->d_c, d_q, data->m_ierr); CeedChk(ierr);
  CeedChk(*data->m_ierr);

  if (cbytes > 0) {
    ierr = cudaMemcpy(qf->ctx, data->d_c, cbytes, cudaMemcpyDeviceToHost); gpuErrchk((cudaError_t)ierr); //CeedChk(ierr);
  }

  return 0;
}

static int CeedQFunctionApply_Cuda(CeedQFunction qf, void *qdata, CeedInt Q,
    const CeedScalar *const *u, CeedScalar *const *v) {

  return 0;
}

/*
static int CeedQFunctionApply_Cuda(CeedQFunction qf, void *qdata, CeedInt Q,
                                  const CeedScalar *const *u,
                                  CeedScalar *const *v) {
  int ierr;
  CeedQFunction_Cuda *data = (CeedQFunction_Cuda*) qf->data;

  const CeedInt nc = data->nc, dim = data->dim;
  const CeedInt ubytes = 5 * sizeof(CeedScalar*);
  const CeedInt ready =  data->ready;
  const CeedInt cbytes = qf->ctxsize;

  if (!ready) {
    data->ready = true;
    ierr = cudaMalloc(&data->d_u, ubytes); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_v, ubytes); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_c, cbytes); CeedChk(ierr);
  }

  ierr = cudaMemcpy(data->d_u, u, ubytes, cudaMemcpyHostToDevice); CeedChk(ierr);
  ierr = cudaMemcpy(data->d_v, v, ubytes, cudaMemcpyHostToDevice); CeedChk(ierr);
  if (cbytes > 0) {
    ierr = cudaMemcpy(data->d_c, qf->ctx, cbytes, cudaMemcpyHostToDevice); CeedChk(ierr);
  }

  (qf->cudafunction)<<<1,1>>>(data->d_c, qdata, Q, data->d_u, data->d_v, nc, dim);

  ierr = cudaGetLastError(); CeedChk(ierr);

  if (cbytes > 0) {
    ierr = cudaMemcpy(qf->ctx, data->d_c, cbytes, cudaMemcpyDeviceToHost); CeedChk(ierr);
  }

  return 0;
}
*/

static int CeedQFunctionDestroy_Cuda(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Cuda *data = (CeedQFunction_Cuda *) qf->data;

  if (data->ready) {
    ierr = cudaFree(data->d_c); CeedChk(ierr);
    ierr = cudaFree(data->m_ierr); CeedChk(ierr);
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
  data->nelem = 1;

  qf->Apply = CeedQFunctionApply_Cuda;
  qf->Destroy = CeedQFunctionDestroy_Cuda;
  return 0;
}
