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

#include <ceed-backend.h>
#include "ceed-cuda-reg.h"
#include "../cuda/ceed-cuda.h"

static const char *restrictionkernels = QUOTE(

extern "C" __global__ void noTrNoTr(const CeedInt nelem,
                                    const CeedInt *__restrict__ indices,
                                    const CeedScalar *__restrict__ u,
                                    CeedScalar *__restrict__ v) {
  if (indices) {
    for(CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
        node < nelem*RESTRICTION_ELEMSIZE;
        node += blockDim.x * gridDim.x) {
      const CeedInt ind = indices[node];
      const CeedInt locNode = node%RESTRICTION_ELEMSIZE;
      const CeedInt e = node/RESTRICTION_ELEMSIZE;
      for(CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp) {
        v[locNode + comp*RESTRICTION_ELEMSIZE + e*RESTRICTION_ELEMSIZE*RESTRICTION_NCOMP] =
          u[ind + RESTRICTION_NNODES * comp];
      }
    }
  } else {
    for(CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
        node < nelem*RESTRICTION_ELEMSIZE;
        node += blockDim.x * gridDim.x) {
      const CeedInt ind = node;
      const CeedInt locNode = node%RESTRICTION_ELEMSIZE;
      const CeedInt e = node/RESTRICTION_ELEMSIZE;
      for(CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp) {
        v[locNode + comp*RESTRICTION_ELEMSIZE + e*RESTRICTION_ELEMSIZE*RESTRICTION_NCOMP] =
          u[ind + RESTRICTION_NNODES * comp];
      }
    }
  }
}

extern "C" __global__ void noTrNoTrInterleaved(const CeedInt nelem,
    const CeedInt *__restrict__ indices,
    const CeedScalar *__restrict__ u,
    CeedScalar *__restrict__ v) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  if (indices) {
    const CeedInt esize = RESTRICTION_ELEMSIZE * nelem;
    for(CeedInt e = blockIdx.x * blockDim.x + threadIdx.x;
        e < nelem;
        e += blockDim.x * gridDim.x) {
      for (CeedInt node = 0; node < RESTRICTION_ELEMSIZE; ++node) {
        const CeedInt ind = indices[e * RESTRICTION_ELEMSIZE + node];
        for(CeedInt d = 0; d < RESTRICTION_NCOMP; ++d) {
          v[tid + node*32 + bid*32*RESTRICTION_ELEMSIZE + d*esize] = u[ind +
              RESTRICTION_NNODES * d]; // TODO: make sure at least 32 elements
        }
      }
    }
  } else {
    const CeedInt esize = RESTRICTION_ELEMSIZE * nelem;
    for(CeedInt e = blockIdx.x * blockDim.x + threadIdx.x;
        e < nelem;
        e += blockDim.x * gridDim.x) {
      for (CeedInt node = 0; node < RESTRICTION_ELEMSIZE; ++node) {
        const CeedInt ind = e * RESTRICTION_ELEMSIZE + node;
        for(CeedInt d = 0; d < RESTRICTION_NCOMP; ++d) {
          v[tid + node*32 + bid*32*RESTRICTION_ELEMSIZE + d*esize] = u[ind +
              RESTRICTION_NNODES * d];
        }
      }
    }
  }
}

extern "C" __global__ void noTrTr(const CeedInt nelem,
                                  const CeedInt *__restrict__ indices,
                                  const CeedScalar *__restrict__ u,
                                  CeedScalar *__restrict__ v) {
  if (indices) {
    for(CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
      node < nelem*RESTRICTION_ELEMSIZE;
      node += blockDim.x * gridDim.x) {
      const CeedInt ind = indices[node];
      const CeedInt locNode = node%RESTRICTION_ELEMSIZE;
      const CeedInt e = node/RESTRICTION_ELEMSIZE;
      for(CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp) {
        v[locNode + comp*RESTRICTION_ELEMSIZE + e*RESTRICTION_ELEMSIZE*RESTRICTION_NCOMP] =
          u[ind * RESTRICTION_NCOMP + comp];
      }
    }
  } else {
    for(CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
      node < nelem*RESTRICTION_ELEMSIZE;
      node += blockDim.x * gridDim.x) {
      const CeedInt ind = node;
      const CeedInt locNode = node%RESTRICTION_ELEMSIZE;
      const CeedInt e = node/RESTRICTION_ELEMSIZE;
      for(CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp) {
        v[locNode + comp*RESTRICTION_ELEMSIZE + e*RESTRICTION_ELEMSIZE*RESTRICTION_NCOMP] =
          u[ind * RESTRICTION_NCOMP + comp];
      }
    }
  }
}

extern "C" __global__ void noTrTrInterleaved(const CeedInt nelem,
    const CeedInt *__restrict__ indices,
    const CeedScalar *__restrict__ u,
    CeedScalar *__restrict__ v) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  if (indices) {
    const CeedInt esize = RESTRICTION_ELEMSIZE * nelem;
    for(CeedInt e = blockIdx.x * blockDim.x + threadIdx.x;
        e < nelem;
        e += blockDim.x * gridDim.x) {
      for (CeedInt node = 0; node < RESTRICTION_ELEMSIZE; ++node) {
        const CeedInt ind = indices[e * RESTRICTION_ELEMSIZE + node];
        for(CeedInt d = 0; d < RESTRICTION_NCOMP; ++d) {
          v[tid + node*32 + bid*32*RESTRICTION_ELEMSIZE + d*esize] = u[ind *
              RESTRICTION_NCOMP + d]; // TODO: make sure at least 32 elements
        }
      }
    }
  } else {
    const CeedInt esize = RESTRICTION_ELEMSIZE * nelem;
    for(CeedInt e = blockIdx.x * blockDim.x + threadIdx.x;
        e < nelem;
        e += blockDim.x * gridDim.x) {
      for (CeedInt node = 0; node < RESTRICTION_ELEMSIZE; ++node) {
        const CeedInt ind = e * RESTRICTION_ELEMSIZE + node;
        for(CeedInt d = 0; d < RESTRICTION_NCOMP; ++d) {
          v[tid + node*32 + bid*32*RESTRICTION_ELEMSIZE + d*esize] = u[ind *
              RESTRICTION_NCOMP + d];
        }
      }
    }
  }
}

extern "C" __global__ void trNoTr(const CeedInt *__restrict__ tindices,
                                  const CeedInt *__restrict__ toffsets,
                                  const CeedScalar *__restrict__ u,
                                  CeedScalar *__restrict__ v) {
  CeedScalar value[RESTRICTION_NCOMP];
  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x;
       i < RESTRICTION_NNODES;
       i += blockDim.x * gridDim.x) {
    const CeedInt rng1 = toffsets[i];
    const CeedInt rngN = toffsets[i+1];
    for (CeedInt d = 0; d < RESTRICTION_NCOMP; ++d)
      value[d] = 0.0;
    for (CeedInt j=rng1; j<rngN; ++j) {
      const CeedInt tind = tindices[j];
      CeedInt n = tind % RESTRICTION_ELEMSIZE;
      CeedInt e = tind / RESTRICTION_ELEMSIZE;
      for (CeedInt d = 0; d < RESTRICTION_NCOMP; ++d)
        value[d] += u[(e*RESTRICTION_NCOMP + d)*RESTRICTION_ELEMSIZE + n];
    }
    for (CeedInt d = 0; d < RESTRICTION_NCOMP; ++d)
      v[d*RESTRICTION_NNODES+i] = value[d];
  }
}

extern "C" __global__ void trTr(const CeedInt *__restrict__ tindices,
                                const CeedInt *__restrict__ toffsets,
                                const CeedScalar *__restrict__ u,
                                CeedScalar *__restrict__ v) {
  double value[RESTRICTION_NCOMP];
  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x;
       i < RESTRICTION_NNODES;
       i += blockDim.x * gridDim.x) {
    const int rng1 = toffsets[i];
    const int rngN = toffsets[i+1];
    for (int d = 0; d < RESTRICTION_NCOMP; ++d)
      value[d] = 0.0;
    for (int j=rng1; j<rngN; ++j) {
      const int tind = tindices[j];
      int n = tind % RESTRICTION_ELEMSIZE;
      int e = tind / RESTRICTION_ELEMSIZE;
      for (int d = 0; d < RESTRICTION_NCOMP; ++d)
        value[d] += u[(e*RESTRICTION_NCOMP + d)*RESTRICTION_ELEMSIZE + n];
    }
    for (int d = 0; d < RESTRICTION_NCOMP; ++d)
      v[d+RESTRICTION_NCOMP*i] = value[d];
  }
}

extern "C" __global__ void trNoTrIdentity(const CeedInt nelem,
    const CeedScalar *__restrict__ u,
    CeedScalar *__restrict__ v) {
  const CeedInt esize = RESTRICTION_ELEMSIZE * RESTRICTION_NCOMP * nelem;
  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
       i += blockDim.x * gridDim.x) {
    const CeedInt e = i / (RESTRICTION_NCOMP * RESTRICTION_ELEMSIZE);
    const CeedInt d = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NCOMP;
    const CeedInt s = i % RESTRICTION_ELEMSIZE;

    v[s + RESTRICTION_ELEMSIZE * e + RESTRICTION_NNODES * d] = u[i];
  }
}

extern "C" __global__ void trTrIdentity(const CeedInt nelem,
                                        const CeedScalar *__restrict__ u,
                                        CeedScalar *__restrict__ v) {
  const CeedInt esize = RESTRICTION_ELEMSIZE * RESTRICTION_NCOMP * nelem;
  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
       i += blockDim.x * gridDim.x) {
    const CeedInt e = i / (RESTRICTION_NCOMP * RESTRICTION_ELEMSIZE);
    const CeedInt d = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NCOMP;
    const CeedInt s = i % RESTRICTION_ELEMSIZE;

    v [ RESTRICTION_NCOMP * (s + RESTRICTION_ELEMSIZE * e) + d ] = u[i];
  }
}

);

static int CeedElemRestrictionApply_Cuda_reg(CeedElemRestriction r,
    CeedTransposeMode tmode, CeedTransposeMode lmode,
    CeedVector u, CeedVector v, CeedRequest *request) {
  int ierr;
  CeedElemRestriction_Cuda_reg *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  Ceed_Cuda_reg *data;
  ierr = CeedGetData(ceed, (void *)&data); CeedChk(ierr);
  const CeedScalar *d_u;
  CeedScalar *d_v;
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChk(ierr);
  const CeedInt warpsize  = 32;
  const CeedInt blocksize = warpsize;
  CeedInt nelem, elemsize, nnodes;
  CeedElemRestrictionGetNumElements(r, &nelem);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumNodes(r, &nnodes); CeedChk(ierr);
  CUfunction kernel;
  if (tmode == CEED_NOTRANSPOSE) {
    if (lmode == CEED_NOTRANSPOSE) {
      kernel = impl->noTrNoTr;
    } else {
      kernel = impl->noTrTr;
    }
    CeedInt elemsize;
    ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
    void *args[] = {&nelem, &impl->d_ind, &d_u, &d_v};
    CeedInt blocksize = elemsize<1024?(elemsize>32?elemsize:32):1024;
    ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(nnodes, blocksize),
                      blocksize, args); CeedChk(ierr);
  } else {
    if (impl->d_ind) {
      if (lmode == CEED_NOTRANSPOSE) {
        kernel = impl->trNoTr;
      } else {
        kernel = impl->trTr;
      }
      void *args[] = {&impl->d_tindices, &impl->d_toffsets, &d_u, &d_v};
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(nnodes, blocksize),
                               blocksize, args); CeedChk(ierr);
    } else {
      if (lmode == CEED_NOTRANSPOSE) {
        kernel = impl->trNoTrIdentity;
      } else {
        kernel = impl->trTrIdentity;
      }
      void *args[] = {&nelem, &d_u, &d_v};
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(nnodes, blocksize),
                               blocksize, args); CeedChk(ierr);
    }
  }
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;

  ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChk(ierr);
  return 0;
}

static int CeedElemRestrictionDestroy_Cuda_reg(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Cuda_reg *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);

  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  ierr = cuModuleUnload(impl->module); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&impl->h_ind_allocated); CeedChk(ierr);
  ierr = cudaFree(impl->d_ind_allocated); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(impl->d_toffsets); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(impl->d_tindices); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

static int CeedElemRestrictionOffset_Cuda_reg(const CeedElemRestriction r,
    const CeedInt *indices) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedElemRestriction_Cuda_reg *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);
  CeedInt nelem, elemsize, nnodes;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumNodes(r, &nnodes); CeedChk(ierr);
  const CeedInt sizeOffsets = nnodes+1;
  CeedInt *toffsets;
  ierr = CeedMalloc(sizeOffsets, &toffsets); CeedChk(ierr);
  const CeedInt sizeIndices = nelem * elemsize;
  CeedInt *tindices;
  ierr = CeedMalloc(sizeIndices, &tindices); CeedChk(ierr);
  for (int i=0; i<=nnodes; ++i) toffsets[i]=0;
  for (int e=0; e < nelem; ++e)
    for (int i=0; i < elemsize; ++i)
      ++toffsets[indices[elemsize*e+i]+1];
  for (int i = 1; i <= nnodes; ++i)
    toffsets[i] += toffsets[i-1];
  for (int e = 0; e < nelem; ++e) {
    for (int i = 0; i < elemsize; ++i) {
      const int lid = elemsize*e+i;
      const int gid = indices[lid];
      tindices[toffsets[gid]++] = lid;
    }
  }
  for (int i = nnodes; i > 0; --i)
    toffsets[i] = toffsets[i - 1];
  toffsets[0] = 0;
  ierr = cudaMalloc((void **)&impl->d_toffsets, sizeOffsets*sizeof(CeedInt));
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(impl->d_toffsets, toffsets, sizeOffsets*sizeof(CeedInt),
                    cudaMemcpyHostToDevice);
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMalloc((void **)&impl->d_tindices, sizeIndices*sizeof(CeedInt));
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(impl->d_tindices, tindices, sizeIndices*sizeof(CeedInt),
                    cudaMemcpyHostToDevice);
  CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&toffsets); CeedChk(ierr);
  ierr = CeedFree(&tindices); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreate_Cuda_reg(CeedMemType mtype,
                                       CeedCopyMode cmode,
                                       const CeedInt *indices,
                                       CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedElemRestriction_Cuda_reg *impl;
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
  impl->d_tindices      = NULL;
  impl->d_toffsets      = NULL;
  ierr = CeedElemRestrictionSetData(r, (void *)&impl); CeedChk(ierr);

  if (mtype == CEED_MEM_HOST) {
    switch (cmode) {
    case CEED_OWN_POINTER:
      impl->h_ind_allocated = (CeedInt *)indices;
      impl->h_ind = (CeedInt *)indices;
      break;
    case CEED_USE_POINTER:
      impl->h_ind = (CeedInt *)indices;
      break;
    case CEED_COPY_VALUES:
      break;
    }
    if (indices != NULL) {
      ierr = cudaMalloc( (void **)&impl->d_ind, size * sizeof(CeedInt));
      CeedChk_Cu(ceed, ierr);
      impl->d_ind_allocated = impl->d_ind;//We own the device memory
      ierr = cudaMemcpy(impl->d_ind, indices, size * sizeof(CeedInt),
                        cudaMemcpyHostToDevice);
      CeedChk_Cu(ceed, ierr);
      ierr = CeedElemRestrictionOffset_Cuda_reg(r, indices); CeedChk(ierr);
    }
  } else if (mtype == CEED_MEM_DEVICE) {
    switch (cmode) {
    case CEED_COPY_VALUES:
      if (indices != NULL) {
        ierr = cudaMalloc( (void **)&impl->d_ind, size * sizeof(CeedInt));
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
    if (indices != NULL) {
      ierr = CeedElemRestrictionOffset_Cuda_reg(r, indices); CeedChk(ierr);
    }
  } else
    return CeedError(ceed, 1, "Only MemType = HOST or DEVICE supported");

  CeedInt ncomp, nnodes;
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumNodes(r, &nnodes); CeedChk(ierr);
  ierr = CeedCompileCuda(ceed, restrictionkernels, &impl->module, 3,
                         "RESTRICTION_ELEMSIZE", elemsize,
                         "RESTRICTION_NCOMP", ncomp,
                         "RESTRICTION_NNODES", nnodes); CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "noTrNoTr", &impl->noTrNoTr);
  CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "noTrTr", &impl->noTrTr);
  CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "trNoTr", &impl->trNoTr);
  CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "trTr", &impl->trTr);
  CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "trNoTrIdentity",
                           &impl->trNoTrIdentity);
  CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "trTrIdentity",
                           &impl->trTrIdentity);
  CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Cuda_reg); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Cuda_reg); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreateBlocked_Cuda_reg(const CeedMemType mtype,
    const CeedCopyMode cmode,
    const CeedInt *indices,
    const CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
}
