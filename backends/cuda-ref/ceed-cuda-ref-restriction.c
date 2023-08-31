// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "../cuda/ceed-cuda-common.h"
#include "../cuda/ceed-cuda-compile.h"
#include "ceed-cuda-ref.h"

//------------------------------------------------------------------------------
// Apply restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Cuda(CeedElemRestriction r, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  Ceed                      ceed;
  Ceed_Cuda                *data;
  CUfunction                kernel;
  CeedInt                   num_elem, elem_size;
  const CeedScalar         *d_u;
  CeedScalar               *d_v;
  CeedElemRestriction_Cuda *impl;

  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedElemRestrictionGetNumElements(r, &num_elem);
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  const CeedInt num_nodes = impl->num_nodes;

  // Get vectors
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

      CeedCallBackend(CeedRunKernel_Cuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
    } else {
      // -- Strided restriction
      kernel             = impl->StridedNoTranspose;
      void   *args[]     = {&num_elem, &d_u, &d_v};
      CeedInt block_size = elem_size < 1024 ? (elem_size > 32 ? elem_size : 32) : 1024;

      CeedCallBackend(CeedRunKernel_Cuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
    }
  } else {
    // E-vector -> L-vector
    if (impl->d_ind) {
      // -- Offsets provided
      CeedInt block_size = 32;

      if (impl->OffsetTranspose) {
        kernel       = impl->OffsetTranspose;
        void *args[] = {&num_elem, &impl->d_ind, &d_u, &d_v};

        CeedCallBackend(CeedRunKernel_Cuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
      } else {
        kernel       = impl->OffsetTransposeDet;
        void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &d_u, &d_v};

        CeedCallBackend(CeedRunKernel_Cuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
      }
    } else {
      // -- Strided restriction
      kernel             = impl->StridedTranspose;
      void   *args[]     = {&num_elem, &d_u, &d_v};
      CeedInt block_size = 32;

      CeedCallBackend(CeedRunKernel_Cuda(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
    }
  }

  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED) *request = NULL;

  // Restore arrays
  CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Cuda(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt **offsets) {
  CeedElemRestriction_Cuda *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  switch (mem_type) {
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
  Ceed                      ceed;
  CeedElemRestriction_Cuda *impl;

  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
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
  Ceed                      ceed;
  bool                     *is_node;
  CeedSize                  l_size;
  CeedInt                   num_elem, elem_size, num_comp, num_nodes = 0;
  CeedInt                  *ind_to_offset, *l_vec_indices, *t_offsets, *t_indices;
  CeedElemRestriction_Cuda *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  CeedCallBackend(CeedElemRestrictionGetNumElements(r, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  CeedCallBackend(CeedElemRestrictionGetLVectorSize(r, &l_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));
  const CeedInt size_indices = num_elem * elem_size;

  // Count num_nodes
  CeedCallBackend(CeedCalloc(l_size, &is_node));

  for (CeedInt i = 0; i < size_indices; i++) is_node[indices[i]] = 1;
  for (CeedInt i = 0; i < l_size; i++) num_nodes += is_node[i];
  impl->num_nodes = num_nodes;

  // L-vector offsets array
  CeedCallBackend(CeedCalloc(l_size, &ind_to_offset));
  CeedCallBackend(CeedCalloc(num_nodes, &l_vec_indices));
  for (CeedInt i = 0, j = 0; i < l_size; i++) {
    if (is_node[i]) {
      l_vec_indices[j] = i;
      ind_to_offset[i] = j++;
    }
  }
  CeedCallBackend(CeedFree(&is_node));

  // Compute transpose offsets and indices
  const CeedInt size_offsets = num_nodes + 1;

  CeedCallBackend(CeedCalloc(size_offsets, &t_offsets));
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
      const CeedInt lid = elem_size * e + i;
      const CeedInt gid = indices[lid];

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
int CeedElemRestrictionCreate_Cuda(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *indices, const bool *orients,
                                   const CeedInt8 *curl_orients, CeedElemRestriction r) {
  Ceed                      ceed, ceed_parent;
  bool                      is_deterministic, is_strided;
  CeedInt                   num_elem, num_comp, elem_size, comp_stride = 1;
  CeedRestrictionType       rstr_type;
  CeedElemRestriction_Cuda *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedGetParent(ceed, &ceed_parent));
  CeedCallBackend(CeedIsDeterministic(ceed_parent, &is_deterministic));
  CeedCallBackend(CeedElemRestrictionGetNumElements(r, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  const CeedInt size       = num_elem * elem_size;
  CeedInt       strides[3] = {1, size, elem_size};
  CeedInt       layout[3]  = {1, elem_size * num_elem, elem_size};

  CeedCallBackend(CeedElemRestrictionGetType(r, &rstr_type));
  CeedCheck(rstr_type != CEED_RESTRICTION_ORIENTED && rstr_type != CEED_RESTRICTION_CURL_ORIENTED, ceed, CEED_ERROR_BACKEND,
            "Backend does not implement CeedElemRestrictionCreateOriented or CeedElemRestrictionCreateCurlOriented");

  // Stride data
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
  CeedCallBackend(CeedElemRestrictionSetELayout(r, layout));

  // Set up device indices/offset arrays
  switch (mem_type) {
    case CEED_MEM_HOST: {
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
        if (is_deterministic) CeedCallBackend(CeedElemRestrictionOffset_Cuda(r, indices));
      }
      break;
    }
    case CEED_MEM_DEVICE: {
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
        if (is_deterministic) CeedCallBackend(CeedElemRestrictionOffset_Cuda(r, indices));
      }
      break;
    }
    // LCOV_EXCL_START
    default:
      return CeedError(ceed, CEED_ERROR_BACKEND, "Only MemType = HOST or DEVICE supported");
      // LCOV_EXCL_STOP
  }

  // Compile CUDA kernels (add atomicAdd function for old NVidia architectures)
  CeedInt num_nodes = impl->num_nodes;
  char   *restriction_kernel_path, *restriction_kernel_source = NULL;

  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-restriction.h", &restriction_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source -----\n");
  if (!is_deterministic) {
    struct cudaDeviceProp prop;
    Ceed_Cuda            *ceed_data;

    CeedCallBackend(CeedGetData(ceed, &ceed_data));
    CeedCallBackend(cudaGetDeviceProperties(&prop, ceed_data->device_id));
    if ((prop.major < 6) && (CEED_SCALAR_TYPE != CEED_SCALAR_FP32)) {
      char *atomic_add_path;

      CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-atomic-add-fallback.h", &atomic_add_path));
      CeedCallBackend(CeedLoadSourceToBuffer(ceed, atomic_add_path, &restriction_kernel_source));
      CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, restriction_kernel_path, &restriction_kernel_source));
      CeedCallBackend(CeedFree(&atomic_add_path));
    }
  }
  if (!restriction_kernel_source) {
    CeedCallBackend(CeedLoadSourceToBuffer(ceed, restriction_kernel_path, &restriction_kernel_source));
  }
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 8, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
                                   "RSTR_NUM_COMP", num_comp, "RSTR_NUM_NODES", num_nodes, "RSTR_COMP_STRIDE", comp_stride, "RSTR_STRIDE_NODES",
                                   strides[0], "RSTR_STRIDE_COMP", strides[1], "RSTR_STRIDE_ELEM", strides[2]));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "StridedNoTranspose", &impl->StridedNoTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "StridedTranspose", &impl->StridedTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->OffsetNoTranspose));
  if (!is_deterministic) {
    CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetTranspose", &impl->OffsetTranspose));
  } else {
    CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetTransposeDet", &impl->OffsetTransposeDet));
  }
  CeedCallBackend(CeedFree(&restriction_kernel_path));
  CeedCallBackend(CeedFree(&restriction_kernel_source));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply", CeedElemRestrictionApply_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyUnsigned", CeedElemRestrictionApply_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyUnoriented", CeedElemRestrictionApply_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOffsets", CeedElemRestrictionGetOffsets_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy", CeedElemRestrictionDestroy_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
