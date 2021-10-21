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

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <stddef.h>
#include "ceed-cuda.h"
#include "kernel-strings/cuda-restriction.h"

//------------------------------------------------------------------------------
// Apply restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Cuda(CeedElemRestriction r,
    CeedTransposeMode tmode, CeedVector u, CeedVector v, CeedRequest *request) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  Ceed_Cuda *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  const CeedInt warpsize  = 32;
  const CeedInt blocksize = warpsize;
  const CeedInt nnodes = impl->nnodes;
  CeedInt nelem, elemsize;
  CeedElemRestrictionGetNumElements(r, &nelem);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChkBackend(ierr);
  CUfunction kernel;

  // Get vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);

  // Restrict
  if (tmode == CEED_NOTRANSPOSE) {
    // L-vector -> E-vector
    if (impl->d_ind) {
      // -- Offsets provided
      kernel = impl->noTrOffset;
      void *args[] = {&nelem, &impl->d_ind, &d_u, &d_v};
      CeedInt blocksize = elemsize<1024?(elemsize>32?elemsize:32):1024;
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(nnodes, blocksize),
                               blocksize, args); CeedChkBackend(ierr);
    } else {
      // -- Strided restriction
      kernel = impl->noTrStrided;
      void *args[] = {&nelem, &d_u, &d_v};
      CeedInt blocksize = elemsize<1024?(elemsize>32?elemsize:32):1024;
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(nnodes, blocksize),
                               blocksize, args); CeedChkBackend(ierr);
    }
  } else {
    // E-vector -> L-vector
    if (impl->d_ind) {
      // -- Offsets provided
      kernel = impl->trOffset;
      void *args[] = {&impl->d_lvec_indices, &impl->d_tindices,
                      &impl->d_toffsets, &d_u, &d_v
                     };
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(nnodes, blocksize),
                               blocksize, args); CeedChkBackend(ierr);
    } else {
      // -- Strided restriction
      kernel = impl->trStrided;
      void *args[] = {&nelem, &d_u, &d_v};
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(nnodes, blocksize),
                               blocksize, args); CeedChkBackend(ierr);
    }
  }

  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;

  // Restore arrays
  ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Blocked not supported
//------------------------------------------------------------------------------
int CeedElemRestrictionApplyBlock_Cuda(CeedElemRestriction r, CeedInt block,
                                       CeedTransposeMode tmode, CeedVector u,
                                       CeedVector v, CeedRequest *request) {
  // LCOV_EXCL_START
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  return CeedError(ceed, CEED_ERROR_BACKEND,
                   "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP
}

//------------------------------------------------------------------------------
// Get offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Cuda(CeedElemRestriction rstr,
    CeedMemType mtype, const CeedInt **offsets) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(rstr, &impl); CeedChkBackend(ierr);

  switch (mtype) {
  case CEED_MEM_HOST:
    *offsets = impl->h_ind;
    break;
  case CEED_MEM_DEVICE:
    *offsets = impl->d_ind;
    break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Cuda(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);

  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  ierr = cuModuleUnload(impl->module); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&impl->h_ind_allocated); CeedChkBackend(ierr);
  ierr = cudaFree(impl->d_ind_allocated); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(impl->d_toffsets); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(impl->d_tindices); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(impl->d_lvec_indices); CeedChk_Cu(ceed, ierr);

  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create transpose offsets and indices
//------------------------------------------------------------------------------
static int CeedElemRestrictionOffset_Cuda(const CeedElemRestriction r,
    const CeedInt *indices) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);
  CeedInt nelem, elemsize, lsize, ncomp;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetLVectorSize(r, &lsize); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChkBackend(ierr);

  // Count nnodes
  bool *isNode;
  ierr = CeedCalloc(lsize, &isNode); CeedChkBackend(ierr);
  const CeedInt sizeIndices = nelem * elemsize;
  for (CeedInt i = 0; i < sizeIndices; i++)
    isNode[indices[i]] = 1;
  CeedInt nnodes = 0;
  for (CeedInt i = 0; i < lsize; i++)
    nnodes += isNode[i];
  impl->nnodes = nnodes;

  // L-vector offsets array
  CeedInt *ind_to_offset, *lvec_indices;
  ierr = CeedCalloc(lsize, &ind_to_offset); CeedChkBackend(ierr);
  ierr = CeedCalloc(nnodes, &lvec_indices); CeedChkBackend(ierr);
  CeedInt j = 0;
  for (CeedInt i = 0; i < lsize; i++)
    if (isNode[i]) {
      lvec_indices[j] = i;
      ind_to_offset[i] = j++;
    }
  ierr = CeedFree(&isNode); CeedChkBackend(ierr);

  // Compute transpose offsets and indices
  const CeedInt sizeOffsets = nnodes + 1;
  CeedInt *toffsets;
  ierr = CeedCalloc(sizeOffsets, &toffsets); CeedChkBackend(ierr);
  CeedInt *tindices;
  ierr = CeedMalloc(sizeIndices, &tindices); CeedChkBackend(ierr);
  // Count node multiplicity
  for (CeedInt e = 0; e < nelem; ++e)
    for (CeedInt i = 0; i < elemsize; ++i)
      ++toffsets[ind_to_offset[indices[elemsize*e + i]] + 1];
  // Convert to running sum
  for (CeedInt i = 1; i < sizeOffsets; ++i)
    toffsets[i] += toffsets[i-1];
  // List all E-vec indices associated with L-vec node
  for (CeedInt e = 0; e < nelem; ++e) {
    for (CeedInt i = 0; i < elemsize; ++i) {
      const CeedInt lid = elemsize*e + i;
      const CeedInt gid = indices[lid];
      tindices[toffsets[ind_to_offset[gid]]++] = lid;
    }
  }
  // Reset running sum
  for (int i = sizeOffsets - 1; i > 0; --i)
    toffsets[i] = toffsets[i - 1];
  toffsets[0] = 0;

  // Copy data to device
  // -- L-vector indices
  ierr = cudaMalloc((void **)&impl->d_lvec_indices, nnodes*sizeof(CeedInt));
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(impl->d_lvec_indices, lvec_indices,
                    nnodes*sizeof(CeedInt), cudaMemcpyHostToDevice);
  CeedChk_Cu(ceed, ierr);
  // -- Transpose offsets
  ierr = cudaMalloc((void **)&impl->d_toffsets, sizeOffsets*sizeof(CeedInt));
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(impl->d_toffsets, toffsets, sizeOffsets*sizeof(CeedInt),
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
  // -- Transpose indices
  ierr = cudaMalloc((void **)&impl->d_tindices, sizeIndices*sizeof(CeedInt));
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(impl->d_tindices, tindices, sizeIndices*sizeof(CeedInt),
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  // Cleanup
  ierr = CeedFree(&ind_to_offset); CeedChkBackend(ierr);
  ierr = CeedFree(&lvec_indices); CeedChkBackend(ierr);
  ierr = CeedFree(&toffsets); CeedChkBackend(ierr);
  ierr = CeedFree(&tindices); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create restriction
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Cuda(CeedMemType mtype, CeedCopyMode cmode,
                                   const CeedInt *indices,
                                   CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  CeedElemRestriction_Cuda *impl;
  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  CeedInt nelem, ncomp, elemsize;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChkBackend(ierr);
  CeedInt size = nelem * elemsize;
  CeedInt strides[3] = {1, size, elemsize};
  CeedInt compstride = 1;

  // Stride data
  bool isStrided;
  ierr = CeedElemRestrictionIsStrided(r, &isStrided); CeedChkBackend(ierr);
  if (isStrided) {
    bool backendstrides;
    ierr = CeedElemRestrictionHasBackendStrides(r, &backendstrides);
    CeedChkBackend(ierr);
    if (!backendstrides) {
      ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChkBackend(ierr);
    }
  } else {
    ierr = CeedElemRestrictionGetCompStride(r, &compstride); CeedChkBackend(ierr);
  }

  impl->h_ind           = NULL;
  impl->h_ind_allocated = NULL;
  impl->d_ind           = NULL;
  impl->d_ind_allocated = NULL;
  impl->d_tindices      = NULL;
  impl->d_toffsets      = NULL;
  impl->nnodes = size;
  ierr = CeedElemRestrictionSetData(r, impl); CeedChkBackend(ierr);
  CeedInt layout[3] = {1, elemsize*nelem, elemsize};
  ierr = CeedElemRestrictionSetELayout(r, layout); CeedChkBackend(ierr);

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
      impl->d_ind_allocated = impl->d_ind; // We own the device memory
      ierr = cudaMemcpy(impl->d_ind, indices, size * sizeof(CeedInt),
                        cudaMemcpyHostToDevice);
      CeedChk_Cu(ceed, ierr);
      ierr = CeedElemRestrictionOffset_Cuda(r, indices); CeedChkBackend(ierr);
    }
  } else if (mtype == CEED_MEM_DEVICE) {
    switch (cmode) {
    case CEED_COPY_VALUES:
      if (indices != NULL) {
        ierr = cudaMalloc( (void **)&impl->d_ind, size * sizeof(CeedInt));
        CeedChk_Cu(ceed, ierr);
        impl->d_ind_allocated = impl->d_ind; // We own the device memory
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
      ierr = CeedElemRestrictionOffset_Cuda(r, indices); CeedChkBackend(ierr);
    }
  } else {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Only MemType = HOST or DEVICE supported");
    // LCOV_EXCL_STOP
  }

  // Compile CUDA kernels
  CeedInt nnodes = impl->nnodes;
  ierr = CeedCompileCuda(ceed, restrictionkernels, &impl->module, 8,
                         "RESTRICTION_ELEMSIZE", elemsize,
                         "RESTRICTION_NELEM", nelem,
                         "RESTRICTION_NCOMP", ncomp,
                         "RESTRICTION_NNODES", nnodes,
                         "RESTRICTION_COMPSTRIDE", compstride,
                         "STRIDE_NODES", strides[0],
                         "STRIDE_COMP", strides[1],
                         "STRIDE_ELEM", strides[2]); CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "noTrStrided",
                           &impl->noTrStrided); CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "noTrOffset", &impl->noTrOffset);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "trStrided", &impl->trStrided);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "trOffset", &impl->trOffset);
  CeedChkBackend(ierr);

  // Register backend functions
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Cuda);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock",
                                CeedElemRestrictionApplyBlock_Cuda);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOffsets",
                                CeedElemRestrictionGetOffsets_Cuda);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Cuda);
  CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Blocked not supported
//------------------------------------------------------------------------------
int CeedElemRestrictionCreateBlocked_Cuda(const CeedMemType mtype,
    const CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  return CeedError(ceed, CEED_ERROR_BACKEND,
                   "Backend does not implement blocked restrictions");
}
//------------------------------------------------------------------------------
