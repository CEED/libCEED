// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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
// Compile restriction kernels
//------------------------------------------------------------------------------
static inline int CeedElemRestrictionSetupCompile_Cuda(CeedElemRestriction rstr) {
  Ceed                      ceed;
  bool                      is_deterministic;
  char                     *restriction_kernel_source;
  const char               *restriction_kernel_path;
  CeedInt                   num_elem, num_comp, elem_size, comp_stride;
  CeedRestrictionType       rstr_type;
  CeedElemRestriction_Cuda *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));
  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(rstr, &comp_stride));
  if (rstr_type == CEED_RESTRICTION_POINTS) {
    CeedCallBackend(CeedElemRestrictionGetMaxPointsInElement(rstr, &elem_size));
  } else {
    CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
  }
  is_deterministic = impl->d_l_vec_indices != NULL;

  // Compile CUDA kernels
  switch (rstr_type) {
    case CEED_RESTRICTION_STRIDED: {
      bool    has_backend_strides;
      CeedInt strides[3] = {1, num_elem * elem_size, elem_size};

      CeedCallBackend(CeedElemRestrictionHasBackendStrides(rstr, &has_backend_strides));
      if (!has_backend_strides) {
        CeedCallBackend(CeedElemRestrictionGetStrides(rstr, strides));
      }

      CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-restriction-strided.h", &restriction_kernel_path));
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source -----\n");
      CeedCallBackend(CeedLoadSourceToBuffer(ceed, restriction_kernel_path, &restriction_kernel_source));
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source Complete! -----\n");
      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
                                       "RSTR_NUM_COMP", num_comp, "RSTR_STRIDE_NODES", strides[0], "RSTR_STRIDE_COMP", strides[1], "RSTR_STRIDE_ELEM",
                                       strides[2]));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "StridedNoTranspose", &impl->ApplyNoTranspose));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "StridedTranspose", &impl->ApplyTranspose));
    } break;
    case CEED_RESTRICTION_POINTS: {
      const char *offset_kernel_path;
      char      **file_paths     = NULL;
      CeedInt     num_file_paths = 0;

      CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-restriction-at-points.h", &restriction_kernel_path));
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source -----\n");
      CeedCallBackend(CeedLoadSourceAndInitializeBuffer(ceed, restriction_kernel_path, &num_file_paths, &file_paths, &restriction_kernel_source));
      CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-restriction-offset.h", &offset_kernel_path));
      CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, offset_kernel_path, &num_file_paths, &file_paths, &restriction_kernel_source));
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source Complete! -----\n");
      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
                                       "RSTR_NUM_COMP", num_comp, "RSTR_NUM_NODES", impl->num_nodes, "RSTR_COMP_STRIDE", comp_stride,
                                       "USE_DETERMINISTIC", is_deterministic ? 1 : 0));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyNoTranspose));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "AtPointsTranspose", &impl->ApplyTranspose));
    } break;
    case CEED_RESTRICTION_STANDARD: {
      CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-restriction-offset.h", &restriction_kernel_path));
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source -----\n");
      CeedCallBackend(CeedLoadSourceToBuffer(ceed, restriction_kernel_path, &restriction_kernel_source));
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source Complete! -----\n");
      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
                                       "RSTR_NUM_COMP", num_comp, "RSTR_NUM_NODES", impl->num_nodes, "RSTR_COMP_STRIDE", comp_stride,
                                       "USE_DETERMINISTIC", is_deterministic ? 1 : 0));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyNoTranspose));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetTranspose", &impl->ApplyTranspose));
    } break;
    case CEED_RESTRICTION_ORIENTED: {
      const char *offset_kernel_path;
      char      **file_paths     = NULL;
      CeedInt     num_file_paths = 0;

      CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-restriction-oriented.h", &restriction_kernel_path));
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source -----\n");
      CeedCallBackend(CeedLoadSourceAndInitializeBuffer(ceed, restriction_kernel_path, &num_file_paths, &file_paths, &restriction_kernel_source));
      CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-restriction-offset.h", &offset_kernel_path));
      CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, offset_kernel_path, &num_file_paths, &file_paths, &restriction_kernel_source));
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source Complete! -----\n");
      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
                                       "RSTR_NUM_COMP", num_comp, "RSTR_NUM_NODES", impl->num_nodes, "RSTR_COMP_STRIDE", comp_stride,
                                       "USE_DETERMINISTIC", is_deterministic ? 1 : 0));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OrientedNoTranspose", &impl->ApplyNoTranspose));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyUnsignedNoTranspose));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OrientedTranspose", &impl->ApplyTranspose));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetTranspose", &impl->ApplyUnsignedTranspose));
      // Cleanup
      CeedCallBackend(CeedFree(&offset_kernel_path));
      for (CeedInt i = 0; i < num_file_paths; i++) CeedCall(CeedFree(&file_paths[i]));
      CeedCall(CeedFree(&file_paths));
    } break;
    case CEED_RESTRICTION_CURL_ORIENTED: {
      const char *offset_kernel_path;
      char      **file_paths     = NULL;
      CeedInt     num_file_paths = 0;

      CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-restriction-curl-oriented.h", &restriction_kernel_path));
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source -----\n");
      CeedCallBackend(CeedLoadSourceAndInitializeBuffer(ceed, restriction_kernel_path, &num_file_paths, &file_paths, &restriction_kernel_source));
      CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-restriction-offset.h", &offset_kernel_path));
      CeedCallBackend(CeedLoadSourceToInitializedBuffer(ceed, offset_kernel_path, &num_file_paths, &file_paths, &restriction_kernel_source));
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Restriction Kernel Source Complete! -----\n");
      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
                                       "RSTR_NUM_COMP", num_comp, "RSTR_NUM_NODES", impl->num_nodes, "RSTR_COMP_STRIDE", comp_stride,
                                       "USE_DETERMINISTIC", is_deterministic ? 1 : 0));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedNoTranspose", &impl->ApplyNoTranspose));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedUnsignedNoTranspose", &impl->ApplyUnsignedNoTranspose));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyUnorientedNoTranspose));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedTranspose", &impl->ApplyTranspose));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedUnsignedTranspose", &impl->ApplyUnsignedTranspose));
      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetTranspose", &impl->ApplyUnorientedTranspose));
      // Cleanup
      CeedCallBackend(CeedFree(&offset_kernel_path));
      for (CeedInt i = 0; i < num_file_paths; i++) CeedCall(CeedFree(&file_paths[i]));
      CeedCall(CeedFree(&file_paths));
    } break;
  }
  CeedCallBackend(CeedFree(&restriction_kernel_path));
  CeedCallBackend(CeedFree(&restriction_kernel_source));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core apply restriction code
//------------------------------------------------------------------------------
static inline int CeedElemRestrictionApply_Cuda_Core(CeedElemRestriction rstr, CeedTransposeMode t_mode, bool use_signs, bool use_orients,
                                                     CeedVector u, CeedVector v, CeedRequest *request) {
  Ceed                      ceed;
  CeedRestrictionType       rstr_type;
  const CeedScalar         *d_u;
  CeedScalar               *d_v;
  CeedElemRestriction_Cuda *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));

  // Assemble kernel if needed
  if (!impl->module) {
    CeedCallBackend(CeedElemRestrictionSetupCompile_Cuda(rstr));
  }

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
    CeedInt elem_size;

    CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
    const CeedInt block_size = elem_size < 1024 ? (elem_size > 32 ? elem_size : 32) : 1024;
    const CeedInt grid       = CeedDivUpInt(impl->num_nodes, block_size);

    switch (rstr_type) {
      case CEED_RESTRICTION_STRIDED: {
        void *args[] = {&d_u, &d_v};

        CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
      } break;
      case CEED_RESTRICTION_POINTS:
      case CEED_RESTRICTION_STANDARD: {
        void *args[] = {&impl->d_offsets, &d_u, &d_v};

        CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
      } break;
      case CEED_RESTRICTION_ORIENTED: {
        if (use_signs) {
          void *args[] = {&impl->d_offsets, &impl->d_orients, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
        } else {
          void *args[] = {&impl->d_offsets, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedNoTranspose, grid, block_size, args));
        }
      } break;
      case CEED_RESTRICTION_CURL_ORIENTED: {
        if (use_signs && use_orients) {
          void *args[] = {&impl->d_offsets, &impl->d_curl_orients, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
        } else if (use_orients) {
          void *args[] = {&impl->d_offsets, &impl->d_curl_orients, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedNoTranspose, grid, block_size, args));
        } else {
          void *args[] = {&impl->d_offsets, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnorientedNoTranspose, grid, block_size, args));
        }
      } break;
    }
  } else {
    // E-vector -> L-vector
    const bool    is_deterministic = impl->d_l_vec_indices != NULL;
    const CeedInt block_size       = 32;
    const CeedInt grid             = CeedDivUpInt(impl->num_nodes, block_size);

    switch (rstr_type) {
      case CEED_RESTRICTION_STRIDED: {
        void *args[] = {&d_u, &d_v};

        CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
      } break;
      case CEED_RESTRICTION_POINTS: {
        if (!is_deterministic) {
          void *args[] = {&impl->d_offsets, &impl->d_points_per_elem, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
        } else {
          void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_points_per_elem, &impl->d_t_offsets, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
        }
      } break;
      case CEED_RESTRICTION_STANDARD: {
        if (!is_deterministic) {
          void *args[] = {&impl->d_offsets, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
        } else {
          void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &d_u, &d_v};

          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
        }
      } break;
      case CEED_RESTRICTION_ORIENTED: {
        if (use_signs) {
          if (!is_deterministic) {
            void *args[] = {&impl->d_offsets, &impl->d_orients, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
          } else {
            void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &impl->d_orients, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
          }
        } else {
          if (!is_deterministic) {
            void *args[] = {&impl->d_offsets, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
          } else {
            void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
          }
        }
      } break;
      case CEED_RESTRICTION_CURL_ORIENTED: {
        if (use_signs && use_orients) {
          if (!is_deterministic) {
            void *args[] = {&impl->d_offsets, &impl->d_curl_orients, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
          } else {
            void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &impl->d_curl_orients, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
          }
        } else if (use_orients) {
          if (!is_deterministic) {
            void *args[] = {&impl->d_offsets, &impl->d_curl_orients, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
          } else {
            void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &impl->d_curl_orients, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
          }
        } else {
          if (!is_deterministic) {
            void *args[] = {&impl->d_offsets, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnorientedTranspose, grid, block_size, args));
          } else {
            void *args[] = {&impl->d_l_vec_indices, &impl->d_t_indices, &impl->d_t_offsets, &d_u, &d_v};

            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnorientedTranspose, grid, block_size, args));
          }
        }
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
static int CeedElemRestrictionApply_Cuda(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Cuda_Core(rstr, t_mode, true, true, u, v, request);
}

//------------------------------------------------------------------------------
// Apply unsigned restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyUnsigned_Cuda(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                                 CeedRequest *request) {
  return CeedElemRestrictionApply_Cuda_Core(rstr, t_mode, false, true, u, v, request);
}

//------------------------------------------------------------------------------
// Apply unoriented restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyUnoriented_Cuda(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                                   CeedRequest *request) {
  return CeedElemRestrictionApply_Cuda_Core(rstr, t_mode, false, false, u, v, request);
}

//------------------------------------------------------------------------------
// Get offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Cuda(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt **offsets) {
  CeedElemRestriction_Cuda *impl;
  CeedRestrictionType       rstr_type;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));
  switch (mem_type) {
    case CEED_MEM_HOST:
      *offsets = rstr_type == CEED_RESTRICTION_POINTS ? impl->h_offsets_at_points : impl->h_offsets;
      break;
    case CEED_MEM_DEVICE:
      *offsets = rstr_type == CEED_RESTRICTION_POINTS ? impl->d_offsets_at_points : impl->d_offsets;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get orientations
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOrientations_Cuda(CeedElemRestriction rstr, CeedMemType mem_type, const bool **orients) {
  CeedElemRestriction_Cuda *impl;
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
static int CeedElemRestrictionGetCurlOrientations_Cuda(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt8 **curl_orients) {
  CeedElemRestriction_Cuda *impl;
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
// Get offset for padded AtPoints E-layout
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetAtPointsElementOffset_Cuda(CeedElemRestriction rstr, CeedInt elem, CeedSize *elem_offset) {
  CeedInt layout[3];

  CeedCallBackend(CeedElemRestrictionGetELayout(rstr, layout));
  *elem_offset = 0 * layout[0] + 0 * layout[1] + elem * layout[2];
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Cuda(CeedElemRestriction rstr) {
  Ceed                      ceed;
  CeedElemRestriction_Cuda *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  if (impl->module) {
    CeedCallCuda(ceed, cuModuleUnload(impl->module));
  }
  CeedCallBackend(CeedFree(&impl->h_offsets_owned));
  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_offsets_owned));
  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_t_offsets));
  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_t_indices));
  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_l_vec_indices));
  CeedCallBackend(CeedFree(&impl->h_orients_owned));
  CeedCallCuda(ceed, cudaFree((bool *)impl->d_orients_owned));
  CeedCallBackend(CeedFree(&impl->h_curl_orients_owned));
  CeedCallCuda(ceed, cudaFree((CeedInt8 *)impl->d_curl_orients_owned));
  CeedCallBackend(CeedFree(&impl->h_offsets_at_points_owned));
  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_offsets_at_points_owned));
  CeedCallBackend(CeedFree(&impl->h_points_per_elem_owned));
  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_points_per_elem_owned));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create transpose offsets and indices
//------------------------------------------------------------------------------
static int CeedElemRestrictionOffset_Cuda(const CeedElemRestriction rstr, const CeedInt elem_size, const CeedInt *indices) {
  Ceed                      ceed;
  bool                     *is_node;
  CeedSize                  l_size;
  CeedInt                   num_elem, num_comp, num_nodes = 0;
  CeedInt                  *ind_to_offset, *l_vec_indices, *t_offsets, *t_indices;
  CeedRestrictionType       rstr_type;
  CeedElemRestriction_Cuda *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));
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
  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_l_vec_indices, num_nodes * sizeof(CeedInt)));
  CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_l_vec_indices, l_vec_indices, num_nodes * sizeof(CeedInt), cudaMemcpyHostToDevice));
  // -- Transpose offsets
  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_t_offsets, size_offsets * sizeof(CeedInt)));
  CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_t_offsets, t_offsets, size_offsets * sizeof(CeedInt), cudaMemcpyHostToDevice));
  // -- Transpose indices
  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_t_indices, size_indices * sizeof(CeedInt)));
  CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_t_indices, t_indices, size_indices * sizeof(CeedInt), cudaMemcpyHostToDevice));

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
int CeedElemRestrictionCreate_Cuda(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orients,
                                   const CeedInt8 *curl_orients, CeedElemRestriction rstr) {
  Ceed                      ceed, ceed_parent;
  bool                      is_deterministic;
  CeedInt                   num_elem, num_comp, elem_size;
  CeedRestrictionType       rstr_type;
  CeedElemRestriction_Cuda *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCallBackend(CeedGetParent(ceed, &ceed_parent));
  CeedCallBackend(CeedIsDeterministic(ceed_parent, &is_deterministic));
  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
  CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));
  // Use max number of points as elem size for AtPoints restrictions
  if (rstr_type == CEED_RESTRICTION_POINTS) {
    CeedInt max_points = 0;

    for (CeedInt i = 0; i < num_elem; i++) {
      max_points = CeedIntMax(max_points, offsets[i + 1] - offsets[i]);
    }
    elem_size = max_points;
  }
  const CeedInt size = num_elem * elem_size;

  CeedCallBackend(CeedCalloc(1, &impl));
  impl->num_nodes = size;
  CeedCallBackend(CeedElemRestrictionSetData(rstr, impl));

  // Set layouts
  {
    bool    has_backend_strides;
    CeedInt layout[3] = {1, size, elem_size};

    CeedCallBackend(CeedElemRestrictionSetELayout(rstr, layout));
    if (rstr_type == CEED_RESTRICTION_STRIDED) {
      CeedCallBackend(CeedElemRestrictionHasBackendStrides(rstr, &has_backend_strides));
      if (has_backend_strides) {
        CeedCallBackend(CeedElemRestrictionSetLLayout(rstr, layout));
      }
    }
  }

  // Pad AtPoints indices
  if (rstr_type == CEED_RESTRICTION_POINTS) {
    CeedSize offsets_len = elem_size * num_elem, at_points_size = num_elem + 1;
    CeedInt  max_points = elem_size, *offsets_padded, *points_per_elem;

    CeedCheck(mem_type == CEED_MEM_HOST, ceed, CEED_ERROR_BACKEND, "only MemType Host supported when creating AtPoints restriction");
    CeedCallBackend(CeedMalloc(offsets_len, &offsets_padded));
    CeedCallBackend(CeedMalloc(num_elem, &points_per_elem));
    for (CeedInt i = 0; i < num_elem; i++) {
      CeedInt num_points = offsets[i + 1] - offsets[i];

      points_per_elem[i] = num_points;
      at_points_size += num_points;
      // -- Copy all points in element
      for (CeedInt j = 0; j < num_points; j++) {
        offsets_padded[i * max_points + j] = offsets[offsets[i] + j] * num_comp;
      }
      // -- Replicate out last point in element
      for (CeedInt j = num_points; j < max_points; j++) {
        offsets_padded[i * max_points + j] = offsets[offsets[i] + num_points - 1] * num_comp;
      }
    }
    CeedCallBackend(CeedSetHostCeedIntArray(offsets, copy_mode, at_points_size, &impl->h_offsets_at_points_owned, &impl->h_offsets_at_points_borrowed,
                                            &impl->h_offsets_at_points));
    CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_offsets_at_points_owned, at_points_size * sizeof(CeedInt)));
    CeedCallCuda(ceed, cudaMemcpy((CeedInt **)impl->d_offsets_at_points_owned, impl->h_offsets_at_points, at_points_size * sizeof(CeedInt),
                                  cudaMemcpyHostToDevice));
    impl->d_offsets_at_points = (CeedInt *)impl->d_offsets_at_points_owned;

    // -- Use padded offsets for the rest of the setup
    offsets   = (const CeedInt *)offsets_padded;
    copy_mode = CEED_OWN_POINTER;
    CeedCallBackend(CeedElemRestrictionSetAtPointsEVectorSize(rstr, at_points_size * num_comp));

    // -- Points per element
    CeedCallBackend(CeedSetHostCeedIntArray(points_per_elem, CEED_OWN_POINTER, num_elem, &impl->h_points_per_elem_owned,
                                            &impl->h_points_per_elem_borrowed, &impl->h_points_per_elem));
    CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_points_per_elem_owned, num_elem * sizeof(CeedInt)));
    CeedCallCuda(ceed,
                 cudaMemcpy((CeedInt **)impl->d_points_per_elem_owned, impl->h_points_per_elem, num_elem * sizeof(CeedInt), cudaMemcpyHostToDevice));
    impl->d_points_per_elem = (CeedInt *)impl->d_points_per_elem_owned;
  }

  // Set up device offset/orientation arrays
  if (rstr_type != CEED_RESTRICTION_STRIDED) {
    switch (mem_type) {
      case CEED_MEM_HOST: {
        CeedCallBackend(CeedSetHostCeedIntArray(offsets, copy_mode, size, &impl->h_offsets_owned, &impl->h_offsets_borrowed, &impl->h_offsets));
        CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_offsets_owned, size * sizeof(CeedInt)));
        CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_offsets_owned, impl->h_offsets, size * sizeof(CeedInt), cudaMemcpyHostToDevice));
        impl->d_offsets = (CeedInt *)impl->d_offsets_owned;
        if (is_deterministic) CeedCallBackend(CeedElemRestrictionOffset_Cuda(rstr, elem_size, offsets));
      } break;
      case CEED_MEM_DEVICE: {
        CeedCallBackend(CeedSetDeviceCeedIntArray_Cuda(ceed, offsets, copy_mode, size, &impl->d_offsets_owned, &impl->d_offsets_borrowed,
                                                       (const CeedInt **)&impl->d_offsets));
        CeedCallBackend(CeedMalloc(size, &impl->h_offsets_owned));
        CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->h_offsets_owned, impl->d_offsets, size * sizeof(CeedInt), cudaMemcpyDeviceToHost));
        impl->h_offsets = impl->h_offsets_owned;
        if (is_deterministic) CeedCallBackend(CeedElemRestrictionOffset_Cuda(rstr, elem_size, offsets));
      } break;
    }

    // Orientation data
    if (rstr_type == CEED_RESTRICTION_ORIENTED) {
      switch (mem_type) {
        case CEED_MEM_HOST: {
          CeedCallBackend(CeedSetHostBoolArray(orients, copy_mode, size, &impl->h_orients_owned, &impl->h_orients_borrowed, &impl->h_orients));
          CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_orients_owned, size * sizeof(bool)));
          CeedCallCuda(ceed, cudaMemcpy((bool *)impl->d_orients_owned, impl->h_orients, size * sizeof(bool), cudaMemcpyHostToDevice));
          impl->d_orients = impl->d_orients_owned;
        } break;
        case CEED_MEM_DEVICE: {
          CeedCallBackend(CeedSetDeviceBoolArray_Cuda(ceed, orients, copy_mode, size, &impl->d_orients_owned, &impl->d_orients_borrowed,
                                                      (const bool **)&impl->d_orients));
          CeedCallBackend(CeedMalloc(size, &impl->h_orients_owned));
          CeedCallCuda(ceed, cudaMemcpy((bool *)impl->h_orients_owned, impl->d_orients, size * sizeof(bool), cudaMemcpyDeviceToHost));
          impl->h_orients = impl->h_orients_owned;
        } break;
      }
    } else if (rstr_type == CEED_RESTRICTION_CURL_ORIENTED) {
      switch (mem_type) {
        case CEED_MEM_HOST: {
          CeedCallBackend(CeedSetHostCeedInt8Array(curl_orients, copy_mode, 3 * size, &impl->h_curl_orients_owned, &impl->h_curl_orients_borrowed,
                                                   &impl->h_curl_orients));
          CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_curl_orients_owned, 3 * size * sizeof(CeedInt8)));
          CeedCallCuda(ceed,
                       cudaMemcpy((CeedInt8 *)impl->d_curl_orients_owned, impl->h_curl_orients, 3 * size * sizeof(CeedInt8), cudaMemcpyHostToDevice));
          impl->d_curl_orients = impl->d_curl_orients_owned;
        } break;
        case CEED_MEM_DEVICE: {
          CeedCallBackend(CeedSetDeviceCeedInt8Array_Cuda(ceed, curl_orients, copy_mode, 3 * size, &impl->d_curl_orients_owned,
                                                          &impl->d_curl_orients_borrowed, (const CeedInt8 **)&impl->d_curl_orients));
          CeedCallBackend(CeedMalloc(3 * size, &impl->h_curl_orients_owned));
          CeedCallCuda(ceed,
                       cudaMemcpy((CeedInt8 *)impl->h_curl_orients_owned, impl->d_curl_orients, 3 * size * sizeof(CeedInt8), cudaMemcpyDeviceToHost));
          impl->h_curl_orients = impl->h_curl_orients_owned;
        } break;
      }
    }
  }

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "Apply", CeedElemRestrictionApply_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyUnsigned", CeedElemRestrictionApplyUnsigned_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyUnoriented", CeedElemRestrictionApplyUnoriented_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetOffsets", CeedElemRestrictionGetOffsets_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetOrientations", CeedElemRestrictionGetOrientations_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetCurlOrientations", CeedElemRestrictionGetCurlOrientations_Cuda));
  if (rstr_type == CEED_RESTRICTION_POINTS) {
    CeedCallBackend(
        CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetAtPointsElementOffset", CeedElemRestrictionGetAtPointsElementOffset_Cuda));
  }
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "Destroy", CeedElemRestrictionDestroy_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
