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
#include <ceed/jit-tools.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <stddef.h>
#include "ceed-cuda-ref.h"
#include "../cuda/ceed-cuda-compile.h"

//------------------------------------------------------------------------------
// Apply restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Cuda(CeedElemRestriction r,
    CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(r, &impl); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  Ceed_Cuda *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  const CeedInt warp_size  = 32;
  const CeedInt block_size = warp_size;
  const CeedInt num_nodes = impl->num_nodes;
  CeedInt num_elem, elem_size;
  CeedElemRestrictionGetNumElements(r, &num_elem);
  ierr = CeedElemRestrictionGetElementSize(r, &elem_size); CeedChkBackend(ierr);
  CUfunction kernel;

  // Get vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  if (t_mode == CEED_TRANSPOSE) {
    // Sum into for transpose mode, e-vec to l-vec
    ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);
  } else {
    // Overwrite for notranspose mode, l-vec to e-vec
    ierr = CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);
  }

  // Restrict
  if (t_mode == CEED_NOTRANSPOSE) {
    // L-vector -> E-vector
    if (impl->d_ind) {
      // -- Offsets provided
      kernel = impl->OffsetNoTranspose;
      void *args[] = {&num_elem, &impl->d_ind, &d_u, &d_v};
      CeedInt block_size = elem_size < 1024 ? (elem_size > 32 ? elem_size : 32) :
                           1024;
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size),
                               block_size, args); CeedChkBackend(ierr);
    } else {
      // -- Strided restriction
      kernel = impl->StridedNoTranspose;
      void *args[] = {&num_elem, &d_u, &d_v};
      CeedInt block_size = elem_size < 1024 ? (elem_size > 32 ? elem_size : 32) :
                           1024;
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size),
                               block_size, args); CeedChkBackend(ierr);
    }
  } else {
    // E-vector -> L-vector
    if (impl->d_ind) {
      // -- Offsets provided
      kernel = impl->OffsetTranspose;
      void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices,
                      &impl->d_t_offsets, &d_u, &d_v
                     };
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size),
                               block_size, args); CeedChkBackend(ierr);
    } else {
      // -- Strided restriction
      kernel = impl->StridedTranspose;
      void *args[] = {&num_elem, &d_u, &d_v};
      ierr = CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size),
                               block_size, args); CeedChkBackend(ierr);
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
                                       CeedTransposeMode t_mode, CeedVector u,
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
    CeedMemType m_type, const CeedInt **offsets) {
  int ierr;
  CeedElemRestriction_Cuda *impl;
  ierr = CeedElemRestrictionGetData(rstr, &impl); CeedChkBackend(ierr);

  switch (m_type) {
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
  ierr = cudaFree(impl->d_t_offsets); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(impl->d_t_indices); CeedChk_Cu(ceed, ierr);
  ierr = cudaFree(impl->d_l_vec_indices); CeedChk_Cu(ceed, ierr);
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
  CeedInt num_elem, elem_size, l_size, num_comp;
  ierr = CeedElemRestrictionGetNumElements(r, &num_elem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elem_size); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetLVectorSize(r, &l_size); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &num_comp); CeedChkBackend(ierr);

  // Count num_nodes
  bool *is_node;
  ierr = CeedCalloc(l_size, &is_node); CeedChkBackend(ierr);
  const CeedInt size_indices = num_elem * elem_size;
  for (CeedInt i = 0; i < size_indices; i++)
    is_node[indices[i]] = 1;
  CeedInt num_nodes = 0;
  for (CeedInt i = 0; i < l_size; i++)
    num_nodes += is_node[i];
  impl->num_nodes = num_nodes;

  // L-vector offsets array
  CeedInt *ind_to_offset, *l_vec_indices;
  ierr = CeedCalloc(l_size, &ind_to_offset); CeedChkBackend(ierr);
  ierr = CeedCalloc(num_nodes, &l_vec_indices); CeedChkBackend(ierr);
  CeedInt j = 0;
  for (CeedInt i = 0; i < l_size; i++)
    if (is_node[i]) {
      l_vec_indices[j] = i;
      ind_to_offset[i] = j++;
    }
  ierr = CeedFree(&is_node); CeedChkBackend(ierr);

  // Compute transpose offsets and indices
  const CeedInt size_offsets = num_nodes + 1;
  CeedInt *t_offsets;
  ierr = CeedCalloc(size_offsets, &t_offsets); CeedChkBackend(ierr);
  CeedInt *t_indices;
  ierr = CeedMalloc(size_indices, &t_indices); CeedChkBackend(ierr);
  // Count node multiplicity
  for (CeedInt e = 0; e < num_elem; ++e)
    for (CeedInt i = 0; i < elem_size; ++i)
      ++t_offsets[ind_to_offset[indices[elem_size*e + i]] + 1];
  // Convert to running sum
  for (CeedInt i = 1; i < size_offsets; ++i)
    t_offsets[i] += t_offsets[i-1];
  // List all E-vec indices associated with L-vec node
  for (CeedInt e = 0; e < num_elem; ++e) {
    for (CeedInt i = 0; i < elem_size; ++i) {
      const CeedInt lid = elem_size*e + i;
      const CeedInt gid = indices[lid];
      t_indices[t_offsets[ind_to_offset[gid]]++] = lid;
    }
  }
  // Reset running sum
  for (int i = size_offsets - 1; i > 0; --i)
    t_offsets[i] = t_offsets[i - 1];
  t_offsets[0] = 0;

  // Copy data to device
  // -- L-vector indices
  ierr = cudaMalloc((void **)&impl->d_l_vec_indices, num_nodes*sizeof(CeedInt));
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(impl->d_l_vec_indices, l_vec_indices,
                    num_nodes*sizeof(CeedInt), cudaMemcpyHostToDevice);
  CeedChk_Cu(ceed, ierr);
  // -- Transpose offsets
  ierr = cudaMalloc((void **)&impl->d_t_offsets, size_offsets*sizeof(CeedInt));
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(impl->d_t_offsets, t_offsets, size_offsets*sizeof(CeedInt),
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);
  // -- Transpose indices
  ierr = cudaMalloc((void **)&impl->d_t_indices, size_indices*sizeof(CeedInt));
  CeedChk_Cu(ceed, ierr);
  ierr = cudaMemcpy(impl->d_t_indices, t_indices, size_indices*sizeof(CeedInt),
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed, ierr);

  // Cleanup
  ierr = CeedFree(&ind_to_offset); CeedChkBackend(ierr);
  ierr = CeedFree(&l_vec_indices); CeedChkBackend(ierr);
  ierr = CeedFree(&t_offsets); CeedChkBackend(ierr);
  ierr = CeedFree(&t_indices); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create restriction
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Cuda(CeedMemType m_type, CeedCopyMode copy_mode,
                                   const CeedInt *indices, CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  CeedElemRestriction_Cuda *impl;
  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  CeedInt num_elem, num_comp, elem_size;
  ierr = CeedElemRestrictionGetNumElements(r, &num_elem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &num_comp); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elem_size); CeedChkBackend(ierr);
  CeedInt size = num_elem * elem_size;
  CeedInt strides[3] = {1, size, elem_size};
  CeedInt comp_stride = 1;

  // Stride data
  bool is_strided;
  ierr = CeedElemRestrictionIsStrided(r, &is_strided); CeedChkBackend(ierr);
  if (is_strided) {
    bool has_backend_strides;
    ierr = CeedElemRestrictionHasBackendStrides(r, &has_backend_strides);
    CeedChkBackend(ierr);
    if (!has_backend_strides) {
      ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChkBackend(ierr);
    }
  } else {
    ierr = CeedElemRestrictionGetCompStride(r, &comp_stride); CeedChkBackend(ierr);
  }

  impl->h_ind           = NULL;
  impl->h_ind_allocated = NULL;
  impl->d_ind           = NULL;
  impl->d_ind_allocated = NULL;
  impl->d_t_indices     = NULL;
  impl->d_t_offsets     = NULL;
  impl->num_nodes = size;
  ierr = CeedElemRestrictionSetData(r, impl); CeedChkBackend(ierr);
  CeedInt layout[3] = {1, elem_size*num_elem, elem_size};
  ierr = CeedElemRestrictionSetELayout(r, layout); CeedChkBackend(ierr);

  // Set up device indices/offset arrays
  if (m_type == CEED_MEM_HOST) {
    switch (copy_mode) {
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
  } else if (m_type == CEED_MEM_DEVICE) {
    switch (copy_mode) {
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
  CeedInt num_nodes = impl->num_nodes;
  char *restriction_kernel_path, *restriction_kernel_source;
  ierr = CeedPathConcatenate(ceed, __FILE__, "kernels/cuda-ref-restriction.h",
                             &restriction_kernel_path); CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Restriction Kernel Source -----\n");
  ierr = CeedLoadSourceToBuffer(ceed, restriction_kernel_path,
                                &restriction_kernel_source);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2,
               "----- Loading Restriction Kernel Source Complete! -----\n");
  ierr = CeedCompileCuda(ceed, restriction_kernel_source, &impl->module, 8,
                         "RESTR_ELEM_SIZE", elem_size,
                         "RESTR_NUM_ELEM", num_elem,
                         "RESTR_NUM_COMP", num_comp,
                         "RESTR_NUM_NODES", num_nodes,
                         "RESTR_COMP_STRIDE", comp_stride,
                         "RESTR_STRIDE_NODES", strides[0],
                         "RESTR_STRIDE_COMP", strides[1],
                         "RESTR_STRIDE_ELEM", strides[2]); CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "StridedTranspose",
                           &impl->StridedTranspose); CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "StridedNoTranspose",
                           &impl->StridedNoTranspose); CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "OffsetTranspose",
                           &impl->OffsetTranspose); CeedChkBackend(ierr);
  ierr = CeedGetKernelCuda(ceed, impl->module, "OffsetNoTranspose",
                           &impl->OffsetNoTranspose); CeedChkBackend(ierr);
  ierr = CeedFree(&restriction_kernel_path); CeedChkBackend(ierr);
  ierr = CeedFree(&restriction_kernel_source); CeedChkBackend(ierr);

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
int CeedElemRestrictionCreateBlocked_Cuda(const CeedMemType m_type,
    const CeedCopyMode copy_mode, const CeedInt *indices, CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
  return CeedError(ceed, CEED_ERROR_BACKEND,
                   "Backend does not implement blocked restrictions");
}
//------------------------------------------------------------------------------
