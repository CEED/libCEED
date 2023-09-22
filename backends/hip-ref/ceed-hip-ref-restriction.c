// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <hip/hip_runtime.h>

#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Core apply restriction code
//------------------------------------------------------------------------------
static inline int CeedElemRestrictionApply_Hip_Core(CeedElemRestriction rstr, CeedTransposeMode t_mode, bool use_signs, bool use_orients,
                                                    CeedVector u, CeedVector v, CeedRequest *request) {
  Ceed                     ceed;
  CeedInt                  num_elem, elem_size;
  CeedRestrictionType      rstr_type;
  const CeedScalar        *d_u;
  CeedScalar              *d_v;
  CeedElemRestriction_Hip *impl;
  hipFunction_t            kernel;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
  CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));
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
    const CeedInt block_size = elem_size < 256 ? (elem_size > 64 ? elem_size : 64) : 256;
    const CeedInt grid       = CeedDivUpInt(num_nodes, block_size);

    switch (rstr_type) {
      case CEED_RESTRICTION_STRIDED: {
        kernel       = impl->StridedNoTranspose;
        void *args[] = {&num_elem, &d_u, &d_v};

        CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
      } break;
      case CEED_RESTRICTION_STANDARD: {
        kernel       = impl->OffsetNoTranspose;
        void *args[] = {&num_elem, &impl->d_ind, &d_u, &d_v};

        CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
      } break;
      case CEED_RESTRICTION_ORIENTED: {
        if (use_signs) {
          kernel       = impl->OrientedNoTranspose;
          void *args[] = {&num_elem, &impl->d_ind, &impl->d_orients, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
        } else {
          kernel       = impl->OffsetNoTranspose;
          void *args[] = {&num_elem, &impl->d_ind, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
        }
      } break;
      case CEED_RESTRICTION_CURL_ORIENTED: {
        if (use_signs && use_orients) {
          kernel       = impl->CurlOrientedNoTranspose;
          void *args[] = {&num_elem, &impl->d_ind, &impl->d_curl_orients, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
        } else if (use_orients) {
          kernel       = impl->CurlOrientedUnsignedNoTranspose;
          void *args[] = {&num_elem, &impl->d_ind, &impl->d_curl_orients, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
        } else {
          kernel       = impl->OffsetNoTranspose;
          void *args[] = {&num_elem, &impl->d_ind, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
        }
      } break;
      case CEED_RESTRICTION_POINTS: {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement restriction CeedElemRestrictionAtPoints");
        // LCOV_EXCL_STOP
      } break;
    }
  } else {
    // E-vector -> L-vector
    const CeedInt block_size = 64;
    const CeedInt grid       = CeedDivUpInt(num_nodes, block_size);

    switch (rstr_type) {
      case CEED_RESTRICTION_STRIDED: {
        kernel       = impl->StridedTranspose;
        void *args[] = {&num_elem, &d_u, &d_v};

        CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
      } break;
      case CEED_RESTRICTION_STANDARD: {
        if (impl->OffsetTranspose) {
          kernel       = impl->OffsetTranspose;
          void *args[] = {&num_elem, &impl->d_ind, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
        } else {
          kernel       = impl->OffsetTransposeDet;
          void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
        }
      } break;
      case CEED_RESTRICTION_ORIENTED: {
        if (use_signs) {
          kernel       = impl->OrientedTranspose;
          void *args[] = {&num_elem, &impl->d_ind, &impl->d_orients, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
        } else {
          if (impl->OffsetTranspose) {
            kernel       = impl->OffsetTranspose;
            void *args[] = {&num_elem, &impl->d_ind, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
          } else {
            kernel       = impl->OffsetTransposeDet;
            void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
          }
        }
      } break;
      case CEED_RESTRICTION_CURL_ORIENTED: {
        if (use_signs && use_orients) {
          kernel       = impl->CurlOrientedTranspose;
          void *args[] = {&num_elem, &impl->d_ind, &impl->d_curl_orients, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
        } else if (use_orients) {
          kernel       = impl->CurlOrientedUnsignedTranspose;
          void *args[] = {&num_elem, &impl->d_ind, &impl->d_curl_orients, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
        } else {
          if (impl->OffsetTranspose) {
            kernel       = impl->OffsetTranspose;
            void *args[] = {&num_elem, &impl->d_ind, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
          } else {
            kernel       = impl->OffsetTransposeDet;
            void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Hip(ceed, kernel, grid, block_size, args));
          }
        }
      } break;
      case CEED_RESTRICTION_POINTS: {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement restriction CeedElemRestrictionAtPoints");
        // LCOV_EXCL_STOP
      } break;
    }
  }

  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED) *request = NULL;

  // Restore arrays
  CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Hip(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Hip_Core(rstr, t_mode, true, true, u, v, request);
}

//------------------------------------------------------------------------------
// Apply unsigned restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyUnsigned_Hip(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                                CeedRequest *request) {
  return CeedElemRestrictionApply_Hip_Core(rstr, t_mode, false, true, u, v, request);
}

//------------------------------------------------------------------------------
// Apply unoriented restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyUnoriented_Hip(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                                  CeedRequest *request) {
  return CeedElemRestrictionApply_Hip_Core(rstr, t_mode, false, false, u, v, request);
}

//------------------------------------------------------------------------------
// Get offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Hip(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt **offsets) {
  CeedElemRestriction_Hip *impl;

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
// Get orientations
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOrientations_Hip(CeedElemRestriction rstr, CeedMemType mem_type, const bool **orients) {
  CeedElemRestriction_Hip *impl;
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));

  switch (mem_type) {
    case CEED_MEM_HOST:
      *orients = impl->h_orients;
      break;
    case CEED_MEM_DEVICE:
      *orients = impl->d_orients;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get curl-conforming orientations
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetCurlOrientations_Hip(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt8 **curl_orients) {
  CeedElemRestriction_Hip *impl;
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));

  switch (mem_type) {
    case CEED_MEM_HOST:
      *curl_orients = impl->h_curl_orients;
      break;
    case CEED_MEM_DEVICE:
      *curl_orients = impl->d_curl_orients;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Hip(CeedElemRestriction rstr) {
  Ceed                     ceed;
  CeedElemRestriction_Hip *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCallHip(ceed, hipModuleUnload(impl->module));
  CeedCallBackend(CeedFree(&impl->h_ind_allocated));
  CeedCallHip(ceed, hipFree(impl->d_ind_allocated));
  CeedCallHip(ceed, hipFree(impl->d_t_offsets));
  CeedCallHip(ceed, hipFree(impl->d_t_indices));
  CeedCallHip(ceed, hipFree(impl->d_l_vec_indices));
  CeedCallBackend(CeedFree(&impl->h_orients_allocated));
  CeedCallHip(ceed, hipFree(impl->d_orients_allocated));
  CeedCallBackend(CeedFree(&impl->h_curl_orients_allocated));
  CeedCallHip(ceed, hipFree(impl->d_curl_orients_allocated));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create transpose offsets and indices
//------------------------------------------------------------------------------
static int CeedElemRestrictionOffset_Hip(const CeedElemRestriction rstr, const CeedInt *indices) {
  Ceed                     ceed;
  bool                    *is_node;
  CeedSize                 l_size;
  CeedInt                  num_elem, elem_size, num_comp, num_nodes = 0;
  CeedInt                 *ind_to_offset, *l_vec_indices, *t_offsets, *t_indices;
  CeedElemRestriction_Hip *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
  CeedCallBackend(CeedElemRestrictionGetLVectorSize(rstr, &l_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
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
int CeedElemRestrictionCreate_Hip(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *indices, const bool *orients,
                                  const CeedInt8 *curl_orients, CeedElemRestriction rstr) {
  Ceed                     ceed, ceed_parent;
  bool                     is_deterministic;
  CeedInt                  num_elem, num_comp, elem_size, comp_stride = 1;
  CeedRestrictionType      rstr_type;
  char                    *restriction_kernel_path, *restriction_kernel_source;
  CeedElemRestriction_Hip *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCallBackend(CeedGetParent(ceed, &ceed_parent));
  CeedCallBackend(CeedIsDeterministic(ceed_parent, &is_deterministic));
  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
  const CeedInt size       = num_elem * elem_size;
  CeedInt       strides[3] = {1, size, elem_size};
  CeedInt       layout[3]  = {1, elem_size * num_elem, elem_size};

  // Stride data
  CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));
  if (rstr_type == CEED_RESTRICTION_STRIDED) {
    bool has_backend_strides;

    CeedCallBackend(CeedElemRestrictionHasBackendStrides(rstr, &has_backend_strides));
    if (!has_backend_strides) {
      CeedCallBackend(CeedElemRestrictionGetStrides(rstr, &strides));
    }
  } else {
    CeedCallBackend(CeedElemRestrictionGetCompStride(rstr, &comp_stride));
  }

  CeedCallBackend(CeedCalloc(1, &impl));
  impl->num_nodes                = size;
  impl->h_ind                    = NULL;
  impl->h_ind_allocated          = NULL;
  impl->d_ind                    = NULL;
  impl->d_ind_allocated          = NULL;
  impl->d_t_indices              = NULL;
  impl->d_t_offsets              = NULL;
  impl->h_orients                = NULL;
  impl->h_orients_allocated      = NULL;
  impl->d_orients                = NULL;
  impl->d_orients_allocated      = NULL;
  impl->h_curl_orients           = NULL;
  impl->h_curl_orients_allocated = NULL;
  impl->d_curl_orients           = NULL;
  impl->d_curl_orients_allocated = NULL;
  CeedCallBackend(CeedElemRestrictionSetData(rstr, impl));
  CeedCallBackend(CeedElemRestrictionSetELayout(rstr, layout));

  // Set up device offset/orientation arrays
  if (rstr_type != CEED_RESTRICTION_STRIDED) {
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
            CeedCallBackend(CeedMalloc(size, &impl->h_ind_allocated));
            memcpy(impl->h_ind_allocated, indices, size * sizeof(CeedInt));
            impl->h_ind = impl->h_ind_allocated;
            break;
        }
        CeedCallHip(ceed, hipMalloc((void **)&impl->d_ind, size * sizeof(CeedInt)));
        impl->d_ind_allocated = impl->d_ind;  // We own the device memory
        CeedCallHip(ceed, hipMemcpy(impl->d_ind, indices, size * sizeof(CeedInt), hipMemcpyHostToDevice));
        if (is_deterministic) CeedCallBackend(CeedElemRestrictionOffset_Hip(rstr, indices));
      } break;
      case CEED_MEM_DEVICE: {
        switch (copy_mode) {
          case CEED_COPY_VALUES:
            CeedCallHip(ceed, hipMalloc((void **)&impl->d_ind, size * sizeof(CeedInt)));
            impl->d_ind_allocated = impl->d_ind;  // We own the device memory
            CeedCallHip(ceed, hipMemcpy(impl->d_ind, indices, size * sizeof(CeedInt), hipMemcpyDeviceToDevice));
            break;
          case CEED_OWN_POINTER:
            impl->d_ind           = (CeedInt *)indices;
            impl->d_ind_allocated = impl->d_ind;
            break;
          case CEED_USE_POINTER:
            impl->d_ind = (CeedInt *)indices;
            break;
        }
        CeedCallBackend(CeedMalloc(size, &impl->h_ind_allocated));
        CeedCallHip(ceed, hipMemcpy(impl->h_ind_allocated, impl->d_ind, size * sizeof(CeedInt), hipMemcpyDeviceToHost));
        impl->h_ind = impl->h_ind_allocated;
        if (is_deterministic) CeedCallBackend(CeedElemRestrictionOffset_Hip(rstr, indices));
      } break;
    }

    // Orientation data
    if (rstr_type == CEED_RESTRICTION_ORIENTED) {
      switch (mem_type) {
        case CEED_MEM_HOST: {
          switch (copy_mode) {
            case CEED_OWN_POINTER:
              impl->h_orients_allocated = (bool *)orients;
              impl->h_orients           = (bool *)orients;
              break;
            case CEED_USE_POINTER:
              impl->h_orients = (bool *)orients;
              break;
            case CEED_COPY_VALUES:
              CeedCallBackend(CeedMalloc(size, &impl->h_orients_allocated));
              memcpy(impl->h_orients_allocated, orients, size * sizeof(bool));
              impl->h_orients = impl->h_orients_allocated;
              break;
          }
          CeedCallHip(ceed, hipMalloc((void **)&impl->d_orients, size * sizeof(bool)));
          impl->d_orients_allocated = impl->d_orients;  // We own the device memory
          CeedCallHip(ceed, hipMemcpy(impl->d_orients, orients, size * sizeof(bool), hipMemcpyHostToDevice));
        } break;
        case CEED_MEM_DEVICE: {
          switch (copy_mode) {
            case CEED_COPY_VALUES:
              CeedCallHip(ceed, hipMalloc((void **)&impl->d_orients, size * sizeof(bool)));
              impl->d_orients_allocated = impl->d_orients;  // We own the device memory
              CeedCallHip(ceed, hipMemcpy(impl->d_orients, orients, size * sizeof(bool), hipMemcpyDeviceToDevice));
              break;
            case CEED_OWN_POINTER:
              impl->d_orients           = (bool *)orients;
              impl->d_orients_allocated = impl->d_orients;
              break;
            case CEED_USE_POINTER:
              impl->d_orients = (bool *)orients;
              break;
          }
          CeedCallBackend(CeedMalloc(size, &impl->h_orients_allocated));
          CeedCallHip(ceed, hipMemcpy(impl->h_orients_allocated, impl->d_orients, size * sizeof(bool), hipMemcpyDeviceToHost));
          impl->h_orients = impl->h_orients_allocated;
        } break;
      }
    } else if (rstr_type == CEED_RESTRICTION_CURL_ORIENTED) {
      switch (mem_type) {
        case CEED_MEM_HOST: {
          switch (copy_mode) {
            case CEED_OWN_POINTER:
              impl->h_curl_orients_allocated = (CeedInt8 *)curl_orients;
              impl->h_curl_orients           = (CeedInt8 *)curl_orients;
              break;
            case CEED_USE_POINTER:
              impl->h_curl_orients = (CeedInt8 *)curl_orients;
              break;
            case CEED_COPY_VALUES:
              CeedCallBackend(CeedMalloc(3 * size, &impl->h_curl_orients_allocated));
              memcpy(impl->h_curl_orients_allocated, curl_orients, 3 * size * sizeof(CeedInt8));
              impl->h_curl_orients = impl->h_curl_orients_allocated;
              break;
          }
          CeedCallHip(ceed, hipMalloc((void **)&impl->d_curl_orients, 3 * size * sizeof(CeedInt8)));
          impl->d_curl_orients_allocated = impl->d_curl_orients;  // We own the device memory
          CeedCallHip(ceed, hipMemcpy(impl->d_curl_orients, curl_orients, 3 * size * sizeof(CeedInt8), hipMemcpyHostToDevice));
        } break;
        case CEED_MEM_DEVICE: {
          switch (copy_mode) {
            case CEED_COPY_VALUES:
              CeedCallHip(ceed, hipMalloc((void **)&impl->d_curl_orients, 3 * size * sizeof(CeedInt8)));
              impl->d_curl_orients_allocated = impl->d_curl_orients;  // We own the device memory
              CeedCallHip(ceed, hipMemcpy(impl->d_curl_orients, curl_orients, 3 * size * sizeof(CeedInt8), hipMemcpyDeviceToDevice));
              break;
            case CEED_OWN_POINTER:
              impl->d_curl_orients           = (CeedInt8 *)curl_orients;
              impl->d_curl_orients_allocated = impl->d_curl_orients;
              break;
            case CEED_USE_POINTER:
              impl->d_curl_orients = (CeedInt8 *)curl_orients;
              break;
          }
          CeedCallBackend(CeedMalloc(3 * size, &impl->h_curl_orients_allocated));
          CeedCallHip(ceed, hipMemcpy(impl->h_curl_orients_allocated, impl->d_curl_orients, 3 * size * sizeof(CeedInt8), hipMemcpyDeviceToHost));
          impl->h_curl_orients = impl->h_curl_orients_allocated;
        } break;
      }
    }
  }

  // Compile HIP kernels
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-restriction.h", &restriction_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, restriction_kernel_path, &restriction_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Hip(ceed, restriction_kernel_source, &impl->module, 8, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
                                  "RSTR_NUM_COMP", num_comp, "RSTR_NUM_NODES", impl->num_nodes, "RSTR_COMP_STRIDE", comp_stride, "RSTR_STRIDE_NODES",
                                  strides[0], "RSTR_STRIDE_COMP", strides[1], "RSTR_STRIDE_ELEM", strides[2]));
  CeedCallBackend(CeedGetKernel_Hip(ceed, impl->module, "StridedNoTranspose", &impl->StridedNoTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, impl->module, "StridedTranspose", &impl->StridedTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, impl->module, "OffsetNoTranspose", &impl->OffsetNoTranspose));
  if (!is_deterministic) {
    CeedCallBackend(CeedGetKernel_Hip(ceed, impl->module, "OffsetTranspose", &impl->OffsetTranspose));
  } else {
    CeedCallBackend(CeedGetKernel_Hip(ceed, impl->module, "OffsetTransposeDet", &impl->OffsetTransposeDet));
  }
  CeedCallBackend(CeedGetKernel_Hip(ceed, impl->module, "OrientedNoTranspose", &impl->OrientedNoTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, impl->module, "OrientedTranspose", &impl->OrientedTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, impl->module, "CurlOrientedNoTranspose", &impl->CurlOrientedNoTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, impl->module, "CurlOrientedUnsignedNoTranspose", &impl->CurlOrientedUnsignedNoTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, impl->module, "CurlOrientedTranspose", &impl->CurlOrientedTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, impl->module, "CurlOrientedUnsignedTranspose", &impl->CurlOrientedUnsignedTranspose));
  CeedCallBackend(CeedFree(&restriction_kernel_path));
  CeedCallBackend(CeedFree(&restriction_kernel_source));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "Apply", CeedElemRestrictionApply_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyUnsigned", CeedElemRestrictionApplyUnsigned_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyUnoriented", CeedElemRestrictionApplyUnoriented_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetOffsets", CeedElemRestrictionGetOffsets_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetOrientations", CeedElemRestrictionGetOrientations_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetCurlOrientations", CeedElemRestrictionGetCurlOrientations_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "Destroy", CeedElemRestrictionDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
