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

#include "ceed-cuda.h"

// *INDENT-OFF*
static const char *restrictionkernels = QUOTE(

#if __CUDA_ARCH__ < 600
//------------------------------------------------------------------------------
// Atomic add, if CUDA version is older
//------------------------------------------------------------------------------
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val +
                                     __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN
    // (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif // __CUDA_ARCH__ < 600

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void noTrStrided(const CeedInt nelem,
                                       const CeedInt *__restrict__ indices,
                                       const CeedScalar *__restrict__ u,
                                       CeedScalar *__restrict__ v) {
  const CeedInt esize = RESTRICTION_ELEMSIZE * RESTRICTION_NCOMP * nelem;

  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
       i += blockDim.x * gridDim.x) {
    const CeedInt e = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NELEM;
    const CeedInt c = i / (RESTRICTION_ELEMSIZE * RESTRICTION_NELEM);
    const CeedInt n = i % RESTRICTION_ELEMSIZE;

    v[i] = u[n*STRIDE_NODES + c*STRIDE_COMP + e*STRIDE_ELEM];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
extern "C" __global__ void noTrOffset(const CeedInt nelem,
                                      const CeedInt *__restrict__ indices,
                                      const CeedScalar *__restrict__ u,
                                      CeedScalar *__restrict__ v) {
  const CeedInt esize = RESTRICTION_ELEMSIZE * RESTRICTION_NCOMP * nelem;

  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
       i += blockDim.x * gridDim.x) {
    const CeedInt e = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NELEM;
    const CeedInt c = i / (RESTRICTION_ELEMSIZE * RESTRICTION_NELEM);
    const CeedInt n = i % RESTRICTION_ELEMSIZE;

    v[i] = u[indices[n + e*RESTRICTION_ELEMSIZE] + c*RESTRICTION_COMPSTRIDE];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void trStrided(const CeedInt nelem,
                                     const CeedInt *__restrict__ indices,
                                     const CeedScalar *__restrict__ u,
                                     CeedScalar *__restrict__ v) {
  const CeedInt esize = RESTRICTION_ELEMSIZE * RESTRICTION_NCOMP * nelem;

  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
       i += blockDim.x * gridDim.x) {
    const CeedInt e = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NELEM;
    const CeedInt c = i / (RESTRICTION_ELEMSIZE * RESTRICTION_NELEM);
    const CeedInt n = i % RESTRICTION_ELEMSIZE;

    v[n*STRIDE_NODES + c*STRIDE_COMP + e*STRIDE_ELEM] += u[i];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
extern "C" __global__ void trOffset(const CeedInt nelem,
                                   const CeedInt *__restrict__ indices,
                                    const CeedScalar *__restrict__ u,
                                    CeedScalar *__restrict__ v) {
  const CeedInt esize = RESTRICTION_ELEMSIZE * RESTRICTION_NCOMP * nelem;

  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < esize;
       i += blockDim.x * gridDim.x) {
    const CeedInt e = (i / RESTRICTION_ELEMSIZE) % RESTRICTION_NELEM;
    const CeedInt c = i / (RESTRICTION_ELEMSIZE * RESTRICTION_NELEM);
    const CeedInt n = i % RESTRICTION_ELEMSIZE;

    atomicAdd(v + (indices[n + e*RESTRICTION_ELEMSIZE] +
                   c*RESTRICTION_COMPSTRIDE), u[i]);
  }
}

);
// *INDENT-ON*

//------------------------------------------------------------------------------
// Apply restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Cuda(CeedElemRestriction r,
    CeedTransposeMode tmode, CeedVector u, CeedVector v, CeedRequest *request) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  Ceed_Cuda *data;
  ierr = CeedGetData(ceed, (void *)&data); CeedChk(ierr);

  // Get vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChk(ierr);

  // Restrict
  CUfunction kernel;
  if (tmode == CEED_NOTRANSPOSE) {
    // L-vector -> E-vector
    if (!impl->d_ind) {
      // -- Strided restriction
      kernel = impl->noTrStrided;
    } else {
      // -- Offsets provided
      kernel = impl->noTrOffset;
    }
  } else {
    // E-vector -> L-vector
    if (!impl->d_ind) {
      // -- Strided restriction
      kernel = impl->trStrided;
    } else {
      // -- Offsets provided
      kernel = impl->trOffset;
    }
  }
  const CeedInt blocksize = data->optblocksize;
  CeedInt nelem;
  CeedElemRestrictionGetNumElements(r, &nelem);
  void *args[] = {&nelem, &impl->d_ind, &d_u, &d_v};
  ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(nelem, blocksize),
                           blocksize, args); CeedChk(ierr);
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;

  // Restore arrays
  ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Blocked not supported
//------------------------------------------------------------------------------
int CeedElemRestrictionApplyBlock_Cuda(CeedElemRestriction r, CeedInt block,
                                       CeedTransposeMode tmode, CeedVector u,
                                       CeedVector v, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
}

//------------------------------------------------------------------------------
// Get offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Cuda(CeedElemRestriction rstr,
    CeedMemType mtype, const CeedInt **offsets) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(rstr, (void *)&impl); CeedChk(ierr);

  switch (mtype) {
  case CEED_MEM_HOST:
    *offsets = impl->h_ind;
    break;
  case CEED_MEM_DEVICE:
    *offsets = impl->d_ind;
    break;
  }
  return 0;
}

//------------------------------------------------------------------------------
// Destroy
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Cuda(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);

  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  ierr = cuModuleUnload(impl->module); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&impl->h_ind_allocated); CeedChk(ierr);
  ierr = cudaFree(impl->d_ind_allocated); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Create Restriction
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Cuda(CeedMemType mtype,
                                   CeedCopyMode cmode, const CeedInt *indices,
                                   CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedElemRestriction_Cuda *impl;
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  CeedInt nelem, elemsize;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  CeedInt size = nelem * elemsize;
  CeedInt strides[3] = {1, size, elemsize};
  CeedInt compstride = 1;

  // Stride data
  bool isStrided;
  ierr = CeedElemRestrictionIsStrided(r, &isStrided); CeedChk(ierr);
  if (isStrided) {
    bool backendstrides;
    ierr = CeedElemRestrictionHasBackendStrides(r, &backendstrides);
    CeedChk(ierr);
    if (!backendstrides) {
      ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChk(ierr);
    }
  } else {
    ierr = CeedElemRestrictionGetCompStride(r, &compstride); CeedChk(ierr);
  }

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  impl->h_ind           = NULL;
  impl->h_ind_allocated = NULL;
  impl->d_ind           = NULL;
  impl->d_ind_allocated = NULL;

  // Set up device indices/offset arrays
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
                        cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
    }
  } else if (mtype == CEED_MEM_DEVICE) {
    switch (cmode) {
    case CEED_COPY_VALUES:
      if (indices != NULL) {
        ierr = cudaMalloc( (void **)&impl->d_ind, size * sizeof(CeedInt));
        CeedChk_Cu(ceed, ierr);
        impl->d_ind_allocated = impl->d_ind;//We own the device memory
        ierr = cudaMemcpy(impl->d_ind, indices, size * sizeof(CeedInt),
                          cudaMemcpyDeviceToDevice); CeedChk_Cu(ceed, ierr);
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

  // Compile CUDA kernels
  CeedInt ncomp;
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  ierr = CeedCompileCuda(ceed, restrictionkernels, &impl->module, 7,
                         "RESTRICTION_ELEMSIZE", elemsize,
                         "RESTRICTION_NELEM", nelem,
                         "RESTRICTION_NCOMP", ncomp,
                         "RESTRICTION_COMPSTRIDE", compstride,
                         "STRIDE_NODES", strides[0],
                         "STRIDE_COMP", strides[1],
                         "STRIDE_ELEM", strides[2]); CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "noTrStrided",
                           &impl->noTrStrided); CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "noTrOffset", &impl->noTrOffset);
  CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "trStrided", &impl->trStrided);
  CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "trOffset", &impl->trOffset);
  CeedChk(ierr);

  // Register backend functions
  ierr = CeedElemRestrictionSetData(r, (void *)&impl); CeedChk(ierr);
  CeedInt layout[3] = {1, elemsize*nelem, elemsize};
  ierr = CeedElemRestrictionSetELayout(r, layout); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Cuda); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock",
                                CeedElemRestrictionApplyBlock_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOffsets",
                                CeedElemRestrictionGetOffsets_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Cuda); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// No blocked restrictions
//------------------------------------------------------------------------------
int CeedElemRestrictionCreateBlocked_Cuda(const CeedMemType mtype,
    const CeedCopyMode cmode, const CeedInt *indices,
    const CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
}
//------------------------------------------------------------------------------
