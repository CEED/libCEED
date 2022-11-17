// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "../cuda/ceed-cuda-compile.h"
#include "ceed-cuda-ref.h"

//------------------------------------------------------------------------------
// Apply restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Cuda(CeedElemRestriction r, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  CeedElemRestriction_Cuda *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  Ceed_Cuda *data;
  CeedCallBackend(CeedGetData(ceed, &data));
  const CeedInt warp_size  = 32;
  const CeedInt block_size = warp_size;
  const CeedInt num_nodes  = impl->num_nodes;
  CeedInt       num_elem, elem_size;
  CeedElemRestrictionGetNumElements(r, &num_elem);
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  CUfunction kernel;

  // Get vectors
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  if (t_mode == CEED_TRANSPOSE) {
    // Sum into for transpose mode, e-vec to l-vec
    CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  } else {
    // Overwrite for notranspose mode, l-vec to e-vec
    CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));
  }

  // Restrict
  if (t_mode == CEED_NOTRANSPOSE) {
    // L-vector -> E-vector
    if (impl->d_ind) {
      // -- Offsets provided
      kernel             = impl->OffsetNoTranspose;
      void   *args[]     = {&num_elem, &impl->d_ind, &d_u, &d_v};
      CeedInt block_size = elem_size < 1024 ? (elem_size > 32 ? elem_size : 32) : 1024;
      CeedCallBackend(CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
    } else {
      // -- Strided restriction
      kernel             = impl->StridedNoTranspose;
      void   *args[]     = {&num_elem, &d_u, &d_v};
      CeedInt block_size = elem_size < 1024 ? (elem_size > 32 ? elem_size : 32) : 1024;
      CeedCallBackend(CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
    }
  } else {
    // E-vector -> L-vector
    if (impl->d_ind) {
      // -- Offsets provided
      kernel       = impl->OffsetTranspose;
      void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &d_u, &d_v};
      CeedCallBackend(CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
    } else {
      // -- Strided restriction
      kernel       = impl->StridedTranspose;
      void *args[] = {&num_elem, &d_u, &d_v};
      CeedCallBackend(CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
    }
  }

  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED) *request = NULL;

  // Restore arrays
  CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Blocked not supported
//------------------------------------------------------------------------------
int CeedElemRestrictionApplyBlock_Cuda(CeedElemRestriction r, CeedInt block, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                       CeedRequest *request) {
  // LCOV_EXCL_START
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement blocked restrictions");
  // LCOV_EXCL_STOP
}

//------------------------------------------------------------------------------
// Get offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Cuda(CeedElemRestriction rstr, CeedMemType m_type, const CeedInt **offsets) {
  CeedElemRestriction_Cuda *impl;
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));

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
  CeedElemRestriction_Cuda *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedCallCuda(ceed, cuModuleUnload(impl->module));
  CeedCallBackend(CeedFree(&impl->h_ind_allocated));
  CeedCallCuda(ceed, cudaFree(impl->d_ind_allocated));
  CeedCallCuda(ceed, cudaFree(impl->d_t_offsets));
  CeedCallCuda(ceed, cudaFree(impl->d_t_indices));
  CeedCallCuda(ceed, cudaFree(impl->d_l_vec_indices));
  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create transpose offsets and indices
//------------------------------------------------------------------------------
static int CeedElemRestrictionOffset_Cuda(const CeedElemRestriction r, const CeedInt *indices) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedElemRestriction_Cuda *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  CeedSize l_size;
  CeedInt  num_elem, elem_size, num_comp;
  CeedCallBackend(CeedElemRestrictionGetNumElements(r, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  CeedCallBackend(CeedElemRestrictionGetLVectorSize(r, &l_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));

  // Count num_nodes
  bool *is_node;
  CeedCallBackend(CeedCalloc(l_size, &is_node));
  const CeedInt size_indices = num_elem * elem_size;
  for (CeedInt i = 0; i < size_indices; i++) is_node[indices[i]] = 1;
  CeedInt num_nodes = 0;
  for (CeedInt i = 0; i < l_size; i++) num_nodes += is_node[i];
  impl->num_nodes = num_nodes;

  // L-vector offsets array
  CeedInt *ind_to_offset, *l_vec_indices;
  CeedCallBackend(CeedCalloc(l_size, &ind_to_offset));
  CeedCallBackend(CeedCalloc(num_nodes, &l_vec_indices));
  CeedInt j = 0;
  for (CeedInt i = 0; i < l_size; i++) {
    if (is_node[i]) {
      l_vec_indices[j] = i;
      ind_to_offset[i] = j++;
    }
  }
  CeedCallBackend(CeedFree(&is_node));

  // Compute transpose offsets and indices
  const CeedInt size_offsets = num_nodes + 1;
  CeedInt      *t_offsets;
  CeedCallBackend(CeedCalloc(size_offsets, &t_offsets));
  CeedInt *t_indices;
  CeedCallBackend(CeedMalloc(size_indices, &t_indices));
  // Count node multiplicity
  for (CeedInt e = 0; e < num_elem; ++e) {
    for (CeedInt i = 0; i < elem_size; ++i) ++t_offsets[ind_to_offset[indices[elem_size * e + i]] + 1];
  }
  // Convert to running sum
  for (CeedInt i = 1; i < size_offsets; ++i) t_offsets[i] += t_offsets[i - 1];
  // List all E-vec indices associated with L-vec node
  for (CeedInt e = 0; e < num_elem; ++e) {
    for (CeedInt i = 0; i < elem_size; ++i) {
      const CeedInt lid                          = elem_size * e + i;
      const CeedInt gid                          = indices[lid];
      t_indices[t_offsets[ind_to_offset[gid]]++] = lid;
    }
  }
  // Reset running sum
  for (int i = size_offsets - 1; i > 0; --i) t_offsets[i] = t_offsets[i - 1];
  t_offsets[0] = 0;

  // Copy data to device
  // -- L-vector indices
  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_l_vec_indices, num_nodes * sizeof(CeedInt)));
  CeedCallCuda(ceed, cudaMemcpy(impl->d_l_vec_indices, l_vec_indices, num_nodes * sizeof(CeedInt), cudaMemcpyHostToDevice));
  // -- Transpose offsets
  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_t_offsets, size_offsets * sizeof(CeedInt)));
  CeedCallCuda(ceed, cudaMemcpy(impl->d_t_offsets, t_offsets, size_offsets * sizeof(CeedInt), cudaMemcpyHostToDevice));
  // -- Transpose indices
  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_t_indices, size_indices * sizeof(CeedInt)));
  CeedCallCuda(ceed, cudaMemcpy(impl->d_t_indices, t_indices, size_indices * sizeof(CeedInt), cudaMemcpyHostToDevice));

  // Cleanup
  CeedCallBackend(CeedFree(&ind_to_offset));
  CeedCallBackend(CeedFree(&l_vec_indices));
  CeedCallBackend(CeedFree(&t_offsets));
  CeedCallBackend(CeedFree(&t_indices));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create restriction
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Cuda(CeedMemType m_type, CeedCopyMode copy_mode, const CeedInt *indices, CeedElemRestriction r) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedElemRestriction_Cuda *impl;
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedInt num_elem, num_comp, elem_size;
  CeedCallBackend(CeedElemRestrictionGetNumElements(r, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  CeedInt size        = num_elem * elem_size;
  CeedInt strides[3]  = {1, size, elem_size};
  CeedInt comp_stride = 1;

  // Stride data
  bool is_strided;
  CeedCallBackend(CeedElemRestrictionIsStrided(r, &is_strided));
  if (is_strided) {
    bool has_backend_strides;
    CeedCallBackend(CeedElemRestrictionHasBackendStrides(r, &has_backend_strides));
    if (!has_backend_strides) {
      CeedCallBackend(CeedElemRestrictionGetStrides(r, &strides));
    }
  } else {
    CeedCallBackend(CeedElemRestrictionGetCompStride(r, &comp_stride));
  }

  impl->h_ind           = NULL;
  impl->h_ind_allocated = NULL;
  impl->d_ind           = NULL;
  impl->d_ind_allocated = NULL;
  impl->d_t_indices     = NULL;
  impl->d_t_offsets     = NULL;
  impl->num_nodes       = size;
  CeedCallBackend(CeedElemRestrictionSetData(r, impl));
  CeedInt layout[3] = {1, elem_size * num_elem, elem_size};
  CeedCallBackend(CeedElemRestrictionSetELayout(r, layout));

  // Set up device indices/offset arrays
  if (m_type == CEED_MEM_HOST) {
    switch (copy_mode) {
      case CEED_OWN_POINTER:
        impl->h_ind_allocated = (CeedInt *)indices;
        impl->h_ind           = (CeedInt *)indices;
        break;
      case CEED_USE_POINTER:
        impl->h_ind = (CeedInt *)indices;
        break;
      case CEED_COPY_VALUES:
        if (indices != NULL) {
          CeedCallBackend(CeedMalloc(elem_size * num_elem, &impl->h_ind_allocated));
          memcpy(impl->h_ind_allocated, indices, elem_size * num_elem * sizeof(CeedInt));
          impl->h_ind = impl->h_ind_allocated;
        }
        break;
    }
    if (indices != NULL) {
      CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_ind, size * sizeof(CeedInt)));
      impl->d_ind_allocated = impl->d_ind;  // We own the device memory
      CeedCallCuda(ceed, cudaMemcpy(impl->d_ind, indices, size * sizeof(CeedInt), cudaMemcpyHostToDevice));
      CeedCallBackend(CeedElemRestrictionOffset_Cuda(r, indices));
    }
  } else if (m_type == CEED_MEM_DEVICE) {
    switch (copy_mode) {
      case CEED_COPY_VALUES:
        if (indices != NULL) {
          CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_ind, size * sizeof(CeedInt)));
          impl->d_ind_allocated = impl->d_ind;  // We own the device memory
          CeedCallCuda(ceed, cudaMemcpy(impl->d_ind, indices, size * sizeof(CeedInt), cudaMemcpyDeviceToDevice));
        }
        break;
      case CEED_OWN_POINTER:
        impl->d_ind           = (CeedInt *)indices;
        impl->d_ind_allocated = impl->d_ind;
        break;
      case CEED_USE_POINTER:
        impl->d_ind = (CeedInt *)indices;
    }
    if (indices != NULL) {
      CeedCallBackend(CeedMalloc(elem_size * num_elem, &impl->h_ind_allocated));
      CeedCallCuda(ceed, cudaMemcpy(impl->h_ind_allocated, impl->d_ind, elem_size * num_elem * sizeof(CeedInt), cudaMemcpyDeviceToHost));
      impl->h_ind = impl->h_ind_allocated;
      CeedCallBackend(CeedElemRestrictionOffset_Cuda(r, indices));
    }
  } else {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Only MemType = HOST or DEVICE supported");
    // LCOV_EXCL_STOP
  }

  // Compile CUDA kernels
  CeedInt num_nodes = impl->num_nodes;
  char   *restriction_kernel_path, *restriction_kernel_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-restriction.h", &restriction_kernel_path));
  CeedDebug256(ceed, 2, "----- Loading Restriction Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, restriction_kernel_path, &restriction_kernel_source));
  CeedDebug256(ceed, 2, "----- Loading Restriction Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompileCuda(ceed, restriction_kernel_source, &impl->module, 8, "RESTR_ELEM_SIZE", elem_size, "RESTR_NUM_ELEM", num_elem,
                                  "RESTR_NUM_COMP", num_comp, "RESTR_NUM_NODES", num_nodes, "RESTR_COMP_STRIDE", comp_stride, "RESTR_STRIDE_NODES",
                                  strides[0], "RESTR_STRIDE_COMP", strides[1], "RESTR_STRIDE_ELEM", strides[2]));
  CeedCallBackend(CeedGetKernelCuda(ceed, impl->module, "StridedTranspose", &impl->StridedTranspose));
  CeedCallBackend(CeedGetKernelCuda(ceed, impl->module, "StridedNoTranspose", &impl->StridedNoTranspose));
  CeedCallBackend(CeedGetKernelCuda(ceed, impl->module, "OffsetTranspose", &impl->OffsetTranspose));
  CeedCallBackend(CeedGetKernelCuda(ceed, impl->module, "OffsetNoTranspose", &impl->OffsetNoTranspose));
  CeedCallBackend(CeedFree(&restriction_kernel_path));
  CeedCallBackend(CeedFree(&restriction_kernel_source));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply", CeedElemRestrictionApply_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock", CeedElemRestrictionApplyBlock_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOffsets", CeedElemRestrictionGetOffsets_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy", CeedElemRestrictionDestroy_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Blocked not supported
//------------------------------------------------------------------------------
int CeedElemRestrictionCreateBlocked_Cuda(const CeedMemType m_type, const CeedCopyMode copy_mode, const CeedInt *indices, CeedElemRestriction r) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement blocked restrictions");
}
//------------------------------------------------------------------------------
