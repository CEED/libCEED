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
#include "ceed-cuda.h"

static const char *restrictionkernels = QUOTE(
    extern "C" __global__ void noTrNoTr(const CeedInt nelem,
                                        const CeedInt * __restrict__ indices, const CeedScalar * __restrict__ u,
CeedScalar * __restrict__ v) {
  const CeedInt esize = RESTRICTION_ELEMSIZE * RESTRICTION_NCOMP * nelem;
  if (indices) {
    for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
         i += blockDim.x * gridDim.x) {
      const CeedInt e = i / (RESTRICTION_NCOMP * RESTRICTION_ELEMSIZE);
      const CeedInt d = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NCOMP;
      const CeedInt s = i % RESTRICTION_ELEMSIZE;

      v[i] = u[indices[s + RESTRICTION_ELEMSIZE * e] + RESTRICTION_NDOF * d];
    }
  } else {
    for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
         i += blockDim.x * gridDim.x) {
      const CeedInt e = i / (RESTRICTION_NCOMP * RESTRICTION_ELEMSIZE);
      const CeedInt d = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NCOMP;
      const CeedInt s = i % RESTRICTION_ELEMSIZE;

      v[i] = u[s + RESTRICTION_ELEMSIZE * e + RESTRICTION_NDOF * d];
    }
  }

}

extern "C" __global__ void noTrTr(const CeedInt nelem,
                                  const CeedInt * __restrict__ indices, const CeedScalar * __restrict__ u,
                                  CeedScalar * __restrict__ v) {
  const CeedInt esize = RESTRICTION_ELEMSIZE * RESTRICTION_NCOMP * nelem;
  if (indices) {
    for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
         i += blockDim.x * gridDim.x) {
      const CeedInt e = i / (RESTRICTION_NCOMP * RESTRICTION_ELEMSIZE);
      const CeedInt d = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NCOMP;
      const CeedInt s = i % RESTRICTION_ELEMSIZE;

      v[i] = u[RESTRICTION_NCOMP * indices[s + RESTRICTION_ELEMSIZE * e] + d];
    }
  } else {
    for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
         i += blockDim.x * gridDim.x) {
      const CeedInt e = i / (RESTRICTION_NCOMP * RESTRICTION_ELEMSIZE);
      const CeedInt d = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NCOMP;
      const CeedInt s = i % RESTRICTION_ELEMSIZE;

      v[i] = u[RESTRICTION_NCOMP * s + RESTRICTION_ELEMSIZE * e + d];
    }
  }
}

extern "C" __global__ void trNoTr(const CeedInt nelem,
                                  const CeedInt * __restrict__ indices, const CeedScalar * __restrict__ u,
                                  CeedScalar * __restrict__ v) {
  const CeedInt esize = RESTRICTION_ELEMSIZE * RESTRICTION_NCOMP * nelem;
  if (indices) {
    for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
         i += blockDim.x * gridDim.x) {
      const CeedInt e = i / (RESTRICTION_NCOMP * RESTRICTION_ELEMSIZE);
      const CeedInt d = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NCOMP;
      const CeedInt s = i % RESTRICTION_ELEMSIZE;

      atomicAdd(v + (indices[s + RESTRICTION_ELEMSIZE * e] + RESTRICTION_NDOF * d),
                u[i]);
    }
  } else {
    for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
         i += blockDim.x * gridDim.x) {
      const CeedInt e = i / (RESTRICTION_NCOMP * RESTRICTION_ELEMSIZE);
      const CeedInt d = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NCOMP;
      const CeedInt s = i % RESTRICTION_ELEMSIZE;

      atomicAdd(v + (s + RESTRICTION_ELEMSIZE * e + RESTRICTION_NDOF * d), u[i]);
    }
  }
}

extern "C" __global__ void trTr(const CeedInt nelem,
                                const CeedInt * __restrict__ indices, const CeedScalar * __restrict__ u,
                                CeedScalar * __restrict__ v) {
  const CeedInt esize = RESTRICTION_ELEMSIZE * RESTRICTION_NCOMP * nelem;
  if (indices) {
    for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
         i += blockDim.x * gridDim.x) {
      const CeedInt e = i / (RESTRICTION_NCOMP * RESTRICTION_ELEMSIZE);
      const CeedInt d = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NCOMP;
      const CeedInt s = i % RESTRICTION_ELEMSIZE;

      atomicAdd(v + (RESTRICTION_NCOMP * indices[s + RESTRICTION_ELEMSIZE * e] + d),
                u[i]);
    }
  } else {
    for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
         i += blockDim.x * gridDim.x) {
      const CeedInt e = i / (RESTRICTION_NCOMP * RESTRICTION_ELEMSIZE);
      const CeedInt d = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NCOMP;
      const CeedInt s = i % RESTRICTION_ELEMSIZE;

      atomicAdd(v + (RESTRICTION_NCOMP * s + RESTRICTION_ELEMSIZE * e + d), u[i]);
    }
  }
}
                                        );

static int CeedElemRestrictionApply_Cuda(CeedElemRestriction r,
    CeedTransposeMode tmode, CeedTransposeMode lmode,
    CeedVector u, CeedVector v, CeedRequest *request) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(r, (void*)&impl); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  Ceed_Cuda *data;
  ierr = CeedGetData(ceed, (void*)&data); CeedChk(ierr);
  const CeedScalar *d_u;
  CeedScalar *d_v;
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChk(ierr);
  CUfunction kernel;
  if (tmode == CEED_NOTRANSPOSE) {
    if (lmode == CEED_NOTRANSPOSE) {
      kernel = impl->noTrNoTr;
    } else {
      kernel = impl->noTrTr;
    }
  } else {
    if (lmode == CEED_NOTRANSPOSE) {
      kernel = impl->trNoTr;
    } else {
      kernel = impl->trTr;
    }
  }
  const CeedInt blocksize = data->optblocksize;
  CeedInt nelem;
  CeedElemRestrictionGetNumElements(r, &nelem);
  void *args[] = {&nelem, &impl->d_ind, &d_u, &d_v};
  ierr = run_kernel(ceed, kernel, CeedDivUpInt(nelem, blocksize), blocksize,
                    args); CeedChk(ierr);
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;

  ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChk(ierr);
  return 0;
}

static int CeedElemRestrictionDestroy_Cuda(CeedElemRestriction r) {
  CeedElemRestriction_Cuda *impl = (CeedElemRestriction_Cuda*)r->data;
  int ierr;

  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  ierr = cuModuleUnload(impl->module);
  CeedChk_Cu(ceed, ierr);
  if (impl->h_ind_allocated) {
    ierr = CeedFree(&impl->h_ind_allocated); CeedChk(ierr);
  }
  if (impl->d_ind_allocated) {
    ierr = cudaFree(impl->d_ind_allocated); CeedChk_Cu(ceed, ierr);
  }
  ierr = CeedFree(&r->data); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreate_Cuda(CeedMemType mtype,
                                   CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedElemRestriction_Cuda *impl;
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  CeedInt nelem, elemsize;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  CeedInt size = nelem * elemsize;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  impl->h_ind           = NULL;
  impl->h_ind_allocated = NULL;
  impl->d_ind           = NULL;
  impl->d_ind_allocated = NULL;

  if (mtype == CEED_MEM_HOST) {
    switch (cmode) {
    case CEED_OWN_POINTER:
      impl->h_ind_allocated = (CeedInt *)indices;
    case CEED_USE_POINTER:
      impl->h_ind = (CeedInt *)indices;
    case CEED_COPY_VALUES:
      if (indices != NULL) {
        ierr = cudaMalloc( (void**)&impl->d_ind, size * sizeof(CeedInt));
        CeedChk_Cu(ceed, ierr);
        impl->d_ind_allocated = impl->d_ind;//We own the device memory
        ierr = cudaMemcpy(impl->d_ind, indices, size * sizeof(CeedInt),
                          cudaMemcpyHostToDevice);
        CeedChk_Cu(ceed, ierr);
      }
    }
  } else if (mtype == CEED_MEM_DEVICE) {
    switch (cmode) {
    case CEED_COPY_VALUES:
      if (indices != NULL) {
        ierr = cudaMalloc( (void**)&impl->d_ind, size * sizeof(CeedInt));
        CeedChk_Cu(ceed, ierr);
        impl->d_ind_allocated = impl->d_ind;//We own the device memory
        ierr = cudaMemcpy(impl->d_ind, indices, size * sizeof(CeedInt),
                          cudaMemcpyDeviceToDevice);
        CeedChk_Cu(ceed, ierr);
      }
      break;
    case CEED_OWN_POINTER:
      impl->d_ind = (CeedInt *)indices;
      impl->d_ind_allocated = impl->d_ind;
      break;
    case CEED_USE_POINTER:
      impl->d_ind = (CeedInt *)indices;
    }
  } else
    return CeedError(ceed, 1, "Only MemType = HOST or DEVICE supported");

  CeedInt ncomp, ndof;
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumDoF(r, &ndof); CeedChk(ierr);
  ierr = compile(ceed, restrictionkernels, &impl->module, 3,
                 "RESTRICTION_ELEMSIZE", elemsize,
                 "RESTRICTION_NCOMP", ncomp,
                 "RESTRICTION_NDOF", ndof); CeedChk(ierr);
  ierr = get_kernel(ceed, impl->module, "noTrNoTr", &impl->noTrNoTr);
  CeedChk(ierr);
  ierr = get_kernel(ceed, impl->module, "noTrTr", &impl->noTrTr); CeedChk(ierr);
  ierr = get_kernel(ceed, impl->module, "trNoTr", &impl->trNoTr); CeedChk(ierr);
  ierr = get_kernel(ceed, impl->module, "trTr", &impl->trTr); CeedChk(ierr);

  ierr = CeedElemRestrictionSetData(r, (void*)&impl); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Cuda); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreateBlocked_Cuda(const CeedMemType mtype,
    const CeedCopyMode cmode,
    const CeedInt *indices,
    const CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
}
