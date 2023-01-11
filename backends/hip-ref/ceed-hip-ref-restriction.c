// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>
#include <hip/hip_runtime.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Apply restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Hip(CeedElemRestriction r, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  CeedElemRestriction_Hip *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  Ceed_Hip *data;
  CeedCallBackend(CeedGetData(ceed, &data));
  const CeedInt block_size = 64;
  const CeedInt num_nodes  = impl->num_nodes;
  CeedInt       num_elem, elem_size;
  CeedElemRestrictionGetNumElements(r, &num_elem);
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  hipFunction_t kernel;

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
      CeedInt block_size = elem_size < 256 ? (elem_size > 64 ? elem_size : 64) : 256;
      CeedCallBackend(CeedRunKernelHip(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
    } else {
      // -- Strided restriction
      kernel             = impl->StridedNoTranspose;
      void   *args[]     = {&num_elem, &d_u, &d_v};
      CeedInt block_size = elem_size < 256 ? (elem_size > 64 ? elem_size : 64) : 256;
      CeedCallBackend(CeedRunKernelHip(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
    }
  } else {
    // E-vector -> L-vector
    if (impl->d_ind) {
      // -- Offsets provided
      kernel       = impl->OffsetTranspose;
      void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &d_u, &d_v};
      CeedCallBackend(CeedRunKernelHip(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
    } else {
      // -- Strided restriction
      kernel       = impl->StridedTranspose;
      void *args[] = {&num_elem, &d_u, &d_v};
      CeedCallBackend(CeedRunKernelHip(ceed, kernel, CeedDivUpInt(num_nodes, block_size), block_size, args));
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
int CeedElemRestrictionApplyBlock_Hip(CeedElemRestriction r, CeedInt block, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
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
static int CeedElemRestrictionGetOffsets_Hip(CeedElemRestriction rstr, CeedMemType mtype, const CeedInt **offsets) {
  CeedElemRestriction_Hip *impl;
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));

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
static int CeedElemRestrictionDestroy_Hip(CeedElemRestriction r) {
  CeedElemRestriction_Hip *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedCallHip(ceed, hipModuleUnload(impl->module));
  CeedCallBackend(CeedFree(&impl->h_ind_allocated));
  CeedCallHip(ceed, hipFree(impl->d_ind_allocated));
  CeedCallHip(ceed, hipFree(impl->d_t_offsets));
  CeedCallHip(ceed, hipFree(impl->d_t_indices));
  CeedCallHip(ceed, hipFree(impl->d_l_vec_indices));
  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create transpose offsets and indices
//------------------------------------------------------------------------------
static int CeedElemRestrictionOffset_Hip(const CeedElemRestriction r, const CeedInt *indices) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedElemRestriction_Hip *impl;
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
  CeedCallHip(ceed, hipMalloc((void **)&impl->d_l_vec_indices, num_nodes * sizeof(CeedInt)));
  CeedCallHip(ceed, hipMemcpy(impl->d_l_vec_indices, l_vec_indices, num_nodes * sizeof(CeedInt), hipMemcpyHostToDevice));
  // -- Transpose offsets
  CeedCallHip(ceed, hipMalloc((void **)&impl->d_t_offsets, size_offsets * sizeof(CeedInt)));
  CeedCallHip(ceed, hipMemcpy(impl->d_t_offsets, t_offsets, size_offsets * sizeof(CeedInt), hipMemcpyHostToDevice));
  // -- Transpose indices
  CeedCallHip(ceed, hipMalloc((void **)&impl->d_t_indices, size_indices * sizeof(CeedInt)));
  CeedCallHip(ceed, hipMemcpy(impl->d_t_indices, t_indices, size_indices * sizeof(CeedInt), hipMemcpyHostToDevice));

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
int CeedElemRestrictionCreate_Hip(CeedMemType mtype, CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction r) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedElemRestriction_Hip *impl;
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
  if (mtype == CEED_MEM_HOST) {
    switch (cmode) {
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
      CeedCallHip(ceed, hipMalloc((void **)&impl->d_ind, size * sizeof(CeedInt)));
      impl->d_ind_allocated = impl->d_ind;  // We own the device memory
      CeedCallHip(ceed, hipMemcpy(impl->d_ind, indices, size * sizeof(CeedInt), hipMemcpyHostToDevice));
      CeedCallBackend(CeedElemRestrictionOffset_Hip(r, indices));
    }
  } else if (mtype == CEED_MEM_DEVICE) {
    switch (cmode) {
      case CEED_COPY_VALUES:
        if (indices != NULL) {
          CeedCallHip(ceed, hipMalloc((void **)&impl->d_ind, size * sizeof(CeedInt)));
          impl->d_ind_allocated = impl->d_ind;  // We own the device memory
          CeedCallHip(ceed, hipMemcpy(impl->d_ind, indices, size * sizeof(CeedInt), hipMemcpyDeviceToDevice));
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
      CeedCallHip(ceed, hipMemcpy(impl->h_ind_allocated, impl->d_ind, elem_size * num_elem * sizeof(CeedInt), hipMemcpyDeviceToHost));
      impl->h_ind = impl->h_ind_allocated;
      CeedCallBackend(CeedElemRestrictionOffset_Hip(r, indices));
    }
  } else {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Only MemType = HOST or DEVICE supported");
    // LCOV_EXCL_STOP
  }

  // Compile HIP kernels
  CeedInt num_nodes = impl->num_nodes;
  char   *restriction_kernel_path, *restriction_kernel_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-restriction.h", &restriction_kernel_path));
  CeedDebug256(ceed, 2, "----- Loading Restriction Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, restriction_kernel_path, &restriction_kernel_source));
  CeedDebug256(ceed, 2, "----- Loading Restriction Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompileHip(ceed, restriction_kernel_source, &impl->module, 8, "RESTR_ELEM_SIZE", elem_size, "RESTR_NUM_ELEM", num_elem,
                                 "RESTR_NUM_COMP", num_comp, "RESTR_NUM_NODES", num_nodes, "RESTR_COMP_STRIDE", comp_stride, "RESTR_STRIDE_NODES",
                                 strides[0], "RESTR_STRIDE_COMP", strides[1], "RESTR_STRIDE_ELEM", strides[2]));
  CeedCallBackend(CeedGetKernelHip(ceed, impl->module, "StridedNoTranspose", &impl->StridedNoTranspose));
  CeedCallBackend(CeedGetKernelHip(ceed, impl->module, "OffsetNoTranspose", &impl->OffsetNoTranspose));
  CeedCallBackend(CeedGetKernelHip(ceed, impl->module, "StridedTranspose", &impl->StridedTranspose));
  CeedCallBackend(CeedGetKernelHip(ceed, impl->module, "OffsetTranspose", &impl->OffsetTranspose));
  CeedCallBackend(CeedFree(&restriction_kernel_path));
  CeedCallBackend(CeedFree(&restriction_kernel_source));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply", CeedElemRestrictionApply_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock", CeedElemRestrictionApplyBlock_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOffsets", CeedElemRestrictionGetOffsets_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy", CeedElemRestrictionDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Blocked not supported
//------------------------------------------------------------------------------
int CeedElemRestrictionCreateBlocked_Hip(const CeedMemType mtype, const CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction r) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement blocked restrictions");
}
//------------------------------------------------------------------------------
