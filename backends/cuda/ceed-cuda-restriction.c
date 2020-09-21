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

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void noTrStrided(const CeedInt nelem,
                                       const CeedScalar *__restrict__ u,
                                       CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
      node < nelem*RESTRICTION_ELEMSIZE;
      node += blockDim.x * gridDim.x) {
    const CeedInt locNode = node % RESTRICTION_ELEMSIZE;
    const CeedInt elem = node / RESTRICTION_ELEMSIZE;

    for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
      v[locNode + comp*RESTRICTION_ELEMSIZE*RESTRICTION_NELEM +
        elem*RESTRICTION_ELEMSIZE] =
          u[locNode*STRIDE_NODES + comp*STRIDE_COMP + elem*STRIDE_ELEM];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
extern "C" __global__ void noTrOffset(const CeedInt nelem,
                                      const CeedInt *__restrict__ indices,
                                      const CeedScalar *__restrict__ u,
                                      CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
      node < nelem*RESTRICTION_ELEMSIZE;
      node += blockDim.x * gridDim.x) {
    const CeedInt ind = indices[node];
    const CeedInt locNode = node % RESTRICTION_ELEMSIZE;
    const CeedInt elem = node / RESTRICTION_ELEMSIZE;

    for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
      v[locNode + comp*RESTRICTION_ELEMSIZE*RESTRICTION_NELEM +
        elem*RESTRICTION_ELEMSIZE] =
          u[ind + comp*RESTRICTION_COMPSTRIDE];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void trStrided(const CeedInt nelem,
    const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
      node < nelem*RESTRICTION_ELEMSIZE;
      node += blockDim.x * gridDim.x) {
    const CeedInt locNode = node % RESTRICTION_ELEMSIZE;
    const CeedInt elem = node / RESTRICTION_ELEMSIZE;

    for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
      v[locNode*STRIDE_NODES + comp*STRIDE_COMP + elem*STRIDE_ELEM] +=
          u[locNode + comp*RESTRICTION_ELEMSIZE*RESTRICTION_NELEM +
            elem*RESTRICTION_ELEMSIZE];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
extern "C" __global__ void trOffset(const CeedInt *__restrict__ lvec_indices,
                                    const CeedInt *__restrict__ tindices,
                                    const CeedInt *__restrict__ toffsets,
                                    const CeedScalar *__restrict__ u,
                                    CeedScalar *__restrict__ v) {
  CeedScalar value[RESTRICTION_NCOMP];

  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x;
       i < RESTRICTION_NNODES;
       i += blockDim.x * gridDim.x) {
    const CeedInt ind = lvec_indices[i];
    const CeedInt rng1 = toffsets[i];
    const CeedInt rngN = toffsets[i+1];

    for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
      value[comp] = 0.0;

    for (CeedInt j = rng1; j < rngN; ++j) {
      const CeedInt tind = tindices[j];
      CeedInt locNode = tind % RESTRICTION_ELEMSIZE;
      CeedInt elem = tind / RESTRICTION_ELEMSIZE;

      for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
        value[comp] += u[locNode + comp*RESTRICTION_ELEMSIZE*RESTRICTION_NELEM +
                         elem*RESTRICTION_ELEMSIZE];
    }

    for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
      v[ind + comp*RESTRICTION_COMPSTRIDE] += value[comp];
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
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  Ceed_Cuda *data;
  ierr = CeedGetData(ceed, &data); CeedChk(ierr);
  const CeedInt warpsize  = 32;
  const CeedInt blocksize = warpsize;
  const CeedInt nnodes = impl->nnodes;
  CeedInt nelem, elemsize;
  CeedElemRestrictionGetNumElements(r, &nelem);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  CUfunction kernel;

  // Get vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChk(ierr);

  // Restrict
  if (tmode == CEED_NOTRANSPOSE) {
    // L-vector -> E-vector
    if (impl->d_ind) {
      // -- Offsets provided
      kernel = impl->noTrOffset;
      void *args[] = {&nelem, &impl->d_ind, &d_u, &d_v};
      CeedInt blocksize = elemsize<1024?(elemsize>32?elemsize:32):1024;
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(nnodes, blocksize),
                               blocksize, args); CeedChk(ierr);
    } else {
      // -- Strided restriction
      kernel = impl->noTrStrided;
      void *args[] = {&nelem, &d_u, &d_v};
      CeedInt blocksize = elemsize<1024?(elemsize>32?elemsize:32):1024;
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(nnodes, blocksize),
                               blocksize, args); CeedChk(ierr);
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
                               blocksize, args); CeedChk(ierr);
    } else {
      // -- Strided restriction
      kernel = impl->trStrided;
      void *args[] = {&nelem, &d_u, &d_v};
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(nnodes, blocksize),
                               blocksize, args); CeedChk(ierr);
    }
  }

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
  // LCOV_EXCL_START
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP
}

//------------------------------------------------------------------------------
// Get offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Cuda(CeedElemRestriction rstr,
    CeedMemType mtype, const CeedInt **offsets) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(rstr, &impl); CeedChk(ierr);

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
// Destroy restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Cuda(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChk(ierr);

  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  ierr = cuModuleUnload(impl->module); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&impl->h_ind_allocated); CeedChk(ierr);
  ierr = cudaFree(impl->d_ind_allocated); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(impl->d_toffsets); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(impl->d_tindices); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(impl->d_lvec_indices); CeedChk_Cu(ceed, ierr);

  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Create transpose offsets and indices
//------------------------------------------------------------------------------
static int CeedElemRestrictionOffset_Cuda(const CeedElemRestriction r,
    const CeedInt *indices) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChk(ierr);
  CeedInt nelem, elemsize, lsize, ncomp;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetLVectorSize(r, &lsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);

  // Count nnodes
  bool *isNode;
  ierr = CeedCalloc(lsize, &isNode); CeedChk(ierr);
  const CeedInt sizeIndices = nelem * elemsize;
  for (CeedInt i = 0; i < sizeIndices; i++)
    isNode[indices[i]] = 1;
  CeedInt nnodes = 0;
  for (CeedInt i = 0; i < lsize; i++)
    nnodes += isNode[i];
  impl->nnodes = nnodes;

  // L-vector offsets array
  CeedInt *ind_to_offset, *lvec_indices;
  ierr = CeedCalloc(lsize, &ind_to_offset); CeedChk(ierr);
  ierr = CeedCalloc(nnodes, &lvec_indices); CeedChk(ierr);
  CeedInt j = 0;
  for (CeedInt i = 0; i < lsize; i++)
    if (isNode[i]) {
      lvec_indices[j] = i;
      ind_to_offset[i] = j++;
    }
  ierr = CeedFree(&isNode); CeedChk(ierr);

  // Compute transpose offsets and indices
  const CeedInt sizeOffsets = nnodes + 1;
  CeedInt *toffsets;
  ierr = CeedCalloc(sizeOffsets, &toffsets); CeedChk(ierr);
  CeedInt *tindices;
  ierr = CeedMalloc(sizeIndices, &tindices); CeedChk(ierr);
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
  ierr = CeedFree(&ind_to_offset); CeedChk(ierr);
  ierr = CeedFree(&lvec_indices); CeedChk(ierr);
  ierr = CeedFree(&toffsets); CeedChk(ierr);
  ierr = CeedFree(&tindices); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Create restriction
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Cuda(CeedMemType mtype, CeedCopyMode cmode,
                                   const CeedInt *indices,
                                   CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedElemRestriction_Cuda *impl;
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  CeedInt nelem, ncomp, elemsize;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
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

  impl->h_ind           = NULL;
  impl->h_ind_allocated = NULL;
  impl->d_ind           = NULL;
  impl->d_ind_allocated = NULL;
  impl->d_tindices      = NULL;
  impl->d_toffsets      = NULL;
  impl->nnodes = size;
  ierr = CeedElemRestrictionSetData(r, impl); CeedChk(ierr);
  CeedInt layout[3] = {1, elemsize*nelem, elemsize};
  ierr = CeedElemRestrictionSetELayout(r, layout); CeedChk(ierr);

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
      ierr = CeedElemRestrictionOffset_Cuda(r, indices); CeedChk(ierr);
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
      ierr = CeedElemRestrictionOffset_Cuda(r, indices); CeedChk(ierr);
    }
  } else {
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Only MemType = HOST or DEVICE supported");
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
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock",
                                CeedElemRestrictionApplyBlock_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOffsets",
                                CeedElemRestrictionGetOffsets_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Cuda);
  CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Blocked not supported
//------------------------------------------------------------------------------
int CeedElemRestrictionCreateBlocked_Cuda(const CeedMemType mtype,
    const CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
}
//------------------------------------------------------------------------------
