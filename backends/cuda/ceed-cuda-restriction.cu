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
#include "ceed-cuda.cuh"

static __global__ void noTrScalar(const CeedInt esize, const CeedInt * __restrict__ indices, const CeedScalar * __restrict__ u, CeedScalar * __restrict__ v) {
  for (CeedInt i = blockIdx.x*blockDim.x + threadIdx.x; i < esize; i += blockDim.x * gridDim.x) {
    v[i] = u[indices[i]];
  }
}

static __global__ void noTrNoTr(const CeedInt esize, const CeedInt ncomp, const CeedInt elemsize, const CeedInt nelem, const CeedInt ndof, const CeedInt * __restrict__ indices, const CeedScalar * __restrict__ u, CeedScalar * __restrict__ v) {
  for (CeedInt i = blockIdx.x*blockDim.x + threadIdx.x; i < esize; i += blockDim.x * gridDim.x) {
    const CeedInt e = i / (ncomp * elemsize);
    const CeedInt d = (i / elemsize) % ncomp;
    const CeedInt s = i % elemsize;

    v[i] = u[indices[s + elemsize * e] + ndof * d];
  }
}

static __global__ void noTrTr(const CeedInt esize, const CeedInt ncomp, const CeedInt elemsize, const CeedInt nelem, const CeedInt * __restrict__ indices, const CeedScalar * __restrict__ u, CeedScalar * __restrict__ v) {
  for (CeedInt i = blockIdx.x*blockDim.x + threadIdx.x; i < esize; i += blockDim.x * gridDim.x) {
    const CeedInt e = i / (ncomp * elemsize);
    const CeedInt d = (i / elemsize) % ncomp;
    const CeedInt s = i % elemsize;

    v[i] = u[ncomp * indices[s + elemsize * e] + d];
  }
}

static __global__ void trScalar(const CeedInt esize, const CeedInt * __restrict__ indices, const CeedScalar * __restrict__ u, CeedScalar * __restrict__ v) {
  for (CeedInt i = blockIdx.x*blockDim.x + threadIdx.x; i < esize; i += blockDim.x * gridDim.x) {
    atomicAdd(v + indices[i], u[i]);
  }
}

static __global__ void trNoTr(const CeedInt esize, const CeedInt ncomp, const CeedInt elemsize, const CeedInt nelem, const CeedInt ndof, const CeedInt * __restrict__ indices, const CeedScalar * __restrict__ u, CeedScalar * __restrict__ v) {
  for (CeedInt i = blockIdx.x*blockDim.x + threadIdx.x; i < esize; i += blockDim.x * gridDim.x) {
    const CeedInt e = i / (ncomp * elemsize);
    const CeedInt d = (i / elemsize) % ncomp;
    const CeedInt s = i % elemsize;

    atomicAdd(v + (indices[s + elemsize * e] + ndof * d), u[i]);
  }
}

static __global__ void trTr(const CeedInt ncomp, const CeedInt esize, const CeedInt elemsize, const CeedInt nelem, const CeedInt * __restrict__ indices, const CeedScalar * __restrict__ u, CeedScalar * __restrict__ v) {
  for (CeedInt i = blockIdx.x*blockDim.x + threadIdx.x; i < esize; i += blockDim.x * gridDim.x) {
    const CeedInt e = i / (ncomp * elemsize);
    const CeedInt d = (i / elemsize) % ncomp;
    const CeedInt s = i % elemsize;

    atomicAdd(v + (ncomp * indices[s + elemsize * e] + d), u[i]);
  }
}

static int CeedElemRestrictionApply_Cuda(CeedElemRestriction r,
    CeedTransposeMode tmode, CeedTransposeMode lmode,
    CeedVector u, CeedVector v, CeedRequest *request) {
  CeedElemRestriction_Cuda *impl = (CeedElemRestriction_Cuda*)r->data;
  Ceed_Cuda *data = (Ceed_Cuda*)r->ceed->data;
  int ierr;
  const CeedInt nelem = r->nelem;
  const CeedInt ncomp = r->ncomp;
  const CeedInt elemsize = r->elemsize;
  const CeedInt ndof = r->ndof;
  const CeedInt esize = nelem*elemsize*ncomp;
  const CeedScalar *d_u;
  CeedScalar *d_v;
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChk(ierr);

  if (!impl->indices) {
    cudaMemcpy(d_v, d_u, esize * sizeof(CeedScalar), cudaMemcpyDeviceToDevice);
  } else {
    const CeedInt* d_indices;
    ierr = CeedVectorGetArrayRead(impl->indices, CEED_MEM_DEVICE, (const CeedScalar**)&d_indices); CeedChk(ierr);
    if (tmode == CEED_NOTRANSPOSE) {
      // Perform: v = r * u
      if (ncomp == 1) {
        ierr = run_cuda(noTrScalar, data->optBlockSize, 0, esize, d_indices, d_u, d_v); CeedChk(ierr);
        // vv is (elemsize x ncomp x nelem), column-major
      } else if (lmode == CEED_NOTRANSPOSE) {
        // u is (ndof x ncomp), column-major
        ierr = run_cuda(noTrNoTr, data->optBlockSize, 0, esize, ncomp, elemsize, nelem, ndof, d_indices, d_u, d_v); CeedChk(ierr);
      } else { // u is (ncomp x ndof), column-major
        ierr = run_cuda(noTrTr, data->optBlockSize, 0, esize, ncomp, elemsize, nelem, d_indices, d_u, d_v); CeedChk(ierr);
      }
    } else {
      // Note: in transpose mode, we perform: v += r^t * u
      if (ncomp == 1) {
        ierr = run_cuda(trScalar, data->optBlockSize, 0, esize, d_indices, d_u, d_v); CeedChk(ierr);
        // u is (elemsize x ncomp x nelem)
      } else if (lmode == CEED_NOTRANSPOSE) {
        // vv is (ndof x ncomp), column-major
        ierr = run_cuda(trNoTr, data->optBlockSize, 0, esize, ncomp, elemsize, nelem, ndof, d_indices, d_u, d_v); CeedChk(ierr);
      } else { // vv is (ncomp x ndof), column-major
        ierr = run_cuda(trTr, data->optBlockSize, 0, esize, ncomp, elemsize, nelem, d_indices, d_u, d_v); CeedChk(ierr);
      }
    }
  } 
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;

  return 0;
}

static int CeedElemRestrictionDestroy_Cuda(CeedElemRestriction r) {
  CeedElemRestriction_Cuda *impl = (CeedElemRestriction_Cuda*)r->data;
  int ierr;

  ierr = CeedVectorDestroy(&impl->indices); CeedChk(ierr);
  ierr = CeedFree(&r->data); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreate_Cuda(CeedElemRestriction r,
    CeedMemType mtype,
    CeedCopyMode cmode, const CeedInt *indices) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  if (indices) {
    ierr = CeedVectorCreate(r->ceed, r->nelem*r->elemsize*sizeof(CeedInt)/sizeof(CeedScalar) + 1, &impl->indices); CeedChk(ierr);
    ierr = CeedVectorSetArray(impl->indices, mtype, cmode, (CeedScalar*)indices); CeedChk(ierr);
  } else {
    impl->indices = NULL;
  }

  r->data = impl;
  r->Apply = CeedElemRestrictionApply_Cuda;
  r->Destroy = CeedElemRestrictionDestroy_Cuda;
  return 0;
}
