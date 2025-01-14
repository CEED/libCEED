// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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
#include "ceed-hip-shared.h"

//------------------------------------------------------------------------------
// Compute a block size based on required minimum threads
//------------------------------------------------------------------------------
static CeedInt ComputeBlockSizeFromRequirement(const CeedInt required) {
  CeedInt maxSize     = 1024;  // Max total threads per block
  CeedInt currentSize = 64;    // Start with one group

  while (currentSize < maxSize) {
    if (currentSize > required) break;
    else currentSize = currentSize * 2;
  }
  return currentSize;
}

//------------------------------------------------------------------------------
// Compute required thread block sizes for basis kernels given P, Q, dim, and
// num_comp (num_comp not currently used, but may be again in other basis
// parallelization options)
//------------------------------------------------------------------------------
static int ComputeBasisThreadBlockSizes(const CeedInt dim, const CeedInt P_1d, const CeedInt Q_1d, const CeedInt num_comp, CeedInt *block_sizes) {
  // Note that this will use the same block sizes for all dimensions when compiling,
  // but as each basis object is defined for a particular dimension, we will never
  // call any kernels except the ones for the dimension for which we have computed the
  // block sizes.
  const CeedInt thread_1d = CeedIntMax(P_1d, Q_1d);

  switch (dim) {
    case 1: {
      // Interp kernels:
      block_sizes[0] = 256;

      // Grad kernels:
      block_sizes[1] = 256;

      // Weight kernels:
      block_sizes[2] = 256;
    } break;
    case 2: {
      // Interp kernels:
      CeedInt required = thread_1d * thread_1d;

      block_sizes[0] = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));

      // Grad kernels: currently use same required minimum threads
      block_sizes[1] = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));

      // Weight kernels:
      required       = CeedIntMax(64, Q_1d * Q_1d);
      block_sizes[2] = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));

    } break;
    case 3: {
      // Interp kernels:
      CeedInt required = thread_1d * thread_1d;

      block_sizes[0] = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));

      // Grad kernels: currently use same required minimum threads
      block_sizes[1] = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));

      // Weight kernels:
      required       = Q_1d * Q_1d * Q_1d;
      block_sizes[2] = CeedIntMax(256, ComputeBlockSizeFromRequirement(required));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply basis
//------------------------------------------------------------------------------
static int CeedBasisApplyTensorCore_Hip_shared(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode,
                                               CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  Ceed                  ceed;
  Ceed_Hip             *ceed_Hip;
  CeedInt               dim, num_comp;
  const CeedScalar     *d_u;
  CeedScalar           *d_v;
  CeedBasis_Hip_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_Hip));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));

  // Get read/write access to u, v
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(eval_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  if (apply_add) CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  else CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Apply basis operation
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      CeedInt P_1d, Q_1d;
      CeedInt block_size = data->block_sizes[0];

      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      CeedInt thread_1d     = CeedIntMax(Q_1d, P_1d);
      void   *interp_args[] = {(void *)&num_elem, &data->d_interp_1d, &d_u, &d_v};

      if (dim == 1) {
        CeedInt elems_per_block = 64 * thread_1d > 256 ? 256 / thread_1d : 64;
        elems_per_block         = elems_per_block > 0 ? elems_per_block : 1;
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, 1,
                                                     elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, data->Interp, grid, thread_1d, 1, elems_per_block, shared_mem, interp_args));
        }
      } else if (dim == 2) {
        // Check if required threads is small enough to do multiple elems
        const CeedInt elems_per_block = CeedIntMax(block_size / (thread_1d * thread_1d), 1);
        CeedInt       grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, thread_1d,
                                                     elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
        }
      } else if (dim == 3) {
        const CeedInt elems_per_block = CeedIntMax(block_size / (thread_1d * thread_1d), 1);
        CeedInt       grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, thread_1d,
                                                     elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
        }
      }
    } break;
    case CEED_EVAL_GRAD: {
      CeedInt P_1d, Q_1d;
      CeedInt block_size = data->block_sizes[1];

      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      CeedInt     thread_1d = CeedIntMax(Q_1d, P_1d);
      CeedScalar *d_grad_1d = data->d_grad_1d;

      if (data->d_collo_grad_1d) {
        d_grad_1d = data->d_collo_grad_1d;
      }
      void *grad_args[] = {(void *)&num_elem, &data->d_interp_1d, &d_grad_1d, &d_u, &d_v};

      if (dim == 1) {
        CeedInt elems_per_block = 64 * thread_1d > 256 ? 256 / thread_1d : 64;
        elems_per_block         = elems_per_block > 0 ? elems_per_block : 1;
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, 1,
                                                     elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, data->Grad, grid, thread_1d, 1, elems_per_block, shared_mem, grad_args));
        }
      } else if (dim == 2) {
        // Check if required threads is small enough to do multiple elems
        const CeedInt elems_per_block = CeedIntMax(block_size / (thread_1d * thread_1d), 1);
        CeedInt       grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, thread_1d,
                                                     elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        }
      } else if (dim == 3) {
        const CeedInt elems_per_block = CeedIntMax(block_size / (thread_1d * thread_1d), 1);
        CeedInt       grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, thread_1d,
                                                     elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        }
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedInt Q_1d;
      CeedInt block_size = data->block_sizes[2];

      CeedCheck(data->d_q_weight_1d, ceed, CEED_ERROR_BACKEND, "%s not supported; q_weights_1d not set", CeedEvalModes[eval_mode]);
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      void *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight_1d, &d_v};

      if (dim == 1) {
        const CeedInt opt_elems       = block_size / Q_1d;
        const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
        const CeedInt grid_size       = num_elem / elems_per_block + (num_elem % elems_per_block > 0);

        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->Weight, grid_size, Q_1d, elems_per_block, 1, weight_args));
      } else if (dim == 2) {
        const CeedInt opt_elems       = block_size / (Q_1d * Q_1d);
        const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
        const CeedInt grid_size       = num_elem / elems_per_block + (num_elem % elems_per_block > 0);

        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->Weight, grid_size, Q_1d, Q_1d, elems_per_block, weight_args));
      } else if (dim == 3) {
        const CeedInt opt_elems       = block_size / (Q_1d * Q_1d);
        const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
        const CeedInt grid_size       = num_elem / elems_per_block + (num_elem % elems_per_block > 0);

        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->Weight, grid_size, Q_1d, Q_1d, elems_per_block, weight_args));
      }
    } break;
    case CEED_EVAL_NONE: /* handled separately below */
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      return CeedError(ceed, CEED_ERROR_BACKEND, "%s not supported", CeedEvalModes[eval_mode]);
      // LCOV_EXCL_STOP
  }

  // Restore vectors, cover CEED_EVAL_NONE
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  if (eval_mode == CEED_EVAL_NONE) CeedCallBackend(CeedVectorSetArray(v, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)d_u));
  if (eval_mode != CEED_EVAL_WEIGHT) CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisApplyTensor_Hip_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                    CeedVector v) {
  CeedCallBackend(CeedBasisApplyTensorCore_Hip_shared(basis, false, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisApplyAddTensor_Hip_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                       CeedVector v) {
  CeedCallBackend(CeedBasisApplyTensorCore_Hip_shared(basis, true, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - tensor AtPoints
//------------------------------------------------------------------------------
static int CeedBasisApplyAtPointsCore_Hip_shared(CeedBasis basis, bool apply_add, const CeedInt num_elem, const CeedInt *num_points,
                                                 CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  Ceed                  ceed;
  CeedInt               Q_1d, dim, max_num_points = num_points[0];
  const CeedInt         is_transpose = t_mode == CEED_TRANSPOSE;
  const CeedScalar     *d_x, *d_u;
  CeedScalar           *d_v;
  CeedBasis_Hip_shared *data;

  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));

  // Weight handled separately
  if (eval_mode == CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorSetValue(v, 1.0));
    return CEED_ERROR_SUCCESS;
  }

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  // Check padded to uniform number of points per elem
  for (CeedInt i = 1; i < num_elem; i++) max_num_points = CeedIntMax(max_num_points, num_points[i]);
  {
    CeedInt  num_comp, q_comp;
    CeedSize len, len_required;

    CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, eval_mode, &q_comp));
    CeedCallBackend(CeedVectorGetLength(is_transpose ? u : v, &len));
    len_required = (CeedSize)num_comp * (CeedSize)q_comp * (CeedSize)num_elem * (CeedSize)max_num_points;
    CeedCheck(len >= len_required, ceed, CEED_ERROR_BACKEND,
              "Vector at points must be padded to the same number of points in each element for BasisApplyAtPoints on GPU backends."
              " Found %" CeedSize_FMT ", Required %" CeedSize_FMT,
              len, len_required);
  }

  // Move num_points array to device
  if (is_transpose) {
    const CeedInt num_bytes = num_elem * sizeof(CeedInt);

    if (num_elem != data->num_elem_at_points) {
      data->num_elem_at_points = num_elem;

      if (data->d_points_per_elem) CeedCallHip(ceed, hipFree(data->d_points_per_elem));
      CeedCallHip(ceed, hipMalloc((void **)&data->d_points_per_elem, num_bytes));
      CeedCallBackend(CeedFree(&data->h_points_per_elem));
      CeedCallBackend(CeedCalloc(num_elem, &data->h_points_per_elem));
    }
    if (memcmp(data->h_points_per_elem, num_points, num_bytes)) {
      memcpy(data->h_points_per_elem, num_points, num_bytes);
      CeedCallHip(ceed, hipMemcpy(data->d_points_per_elem, num_points, num_bytes, hipMemcpyHostToDevice));
    }
  }

  // Build kernels if needed
  if (data->num_points != max_num_points) {
    CeedInt P_1d;

    CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
    data->num_points = max_num_points;

    // -- Create interp matrix to Chebyshev coefficients
    if (!data->d_chebyshev_interp_1d) {
      CeedSize    interp_bytes;
      CeedScalar *chebyshev_interp_1d;

      interp_bytes = P_1d * Q_1d * sizeof(CeedScalar);
      CeedCallBackend(CeedCalloc(P_1d * Q_1d, &chebyshev_interp_1d));
      CeedCallBackend(CeedBasisGetChebyshevInterp1D(basis, chebyshev_interp_1d));
      CeedCallHip(ceed, hipMalloc((void **)&data->d_chebyshev_interp_1d, interp_bytes));
      CeedCallHip(ceed, hipMemcpy(data->d_chebyshev_interp_1d, chebyshev_interp_1d, interp_bytes, hipMemcpyHostToDevice));
      CeedCallBackend(CeedFree(&chebyshev_interp_1d));
    }

    // -- Compile kernels
    const char basis_kernel_source[] = "// AtPoints basis source\n#include <ceed/jit-source/hip/hip-shared-basis-tensor-at-points.h>\n";
    CeedInt    num_comp;

    if (data->moduleAtPoints) CeedCallHip(ceed, hipModuleUnload(data->moduleAtPoints));
    CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
    CeedCallBackend(CeedCompile_Hip(ceed, basis_kernel_source, &data->moduleAtPoints, 9, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "T_1D",
                                    CeedIntMax(Q_1d, P_1d), "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp, "BASIS_NUM_NODES", CeedIntPow(P_1d, dim),
                                    "BASIS_NUM_QPTS", CeedIntPow(Q_1d, dim), "BASIS_NUM_PTS", max_num_points, "BASIS_INTERP_BLOCK_SIZE",
                                    data->block_sizes[0]));
    CeedCallBackend(CeedGetKernel_Hip(ceed, data->moduleAtPoints, "InterpAtPoints", &data->InterpAtPoints));
    CeedCallBackend(CeedGetKernel_Hip(ceed, data->moduleAtPoints, "InterpTransposeAtPoints", &data->InterpTransposeAtPoints));
    CeedCallBackend(CeedGetKernel_Hip(ceed, data->moduleAtPoints, "GradAtPoints", &data->GradAtPoints));
    CeedCallBackend(CeedGetKernel_Hip(ceed, data->moduleAtPoints, "GradTransposeAtPoints", &data->GradTransposeAtPoints));
  }

  // Get read/write access to u, v
  CeedCallBackend(CeedVectorGetArrayRead(x_ref, CEED_MEM_DEVICE, &d_x));
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(eval_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  if (apply_add) CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  else CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (is_transpose && !apply_add) {
    CeedInt  num_comp, q_comp, num_nodes;
    CeedSize length;

    CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
    CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, eval_mode, &q_comp));
    CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
    length =
        (CeedSize)num_elem * (CeedSize)num_comp * (t_mode == CEED_TRANSPOSE ? (CeedSize)num_nodes : ((CeedSize)max_num_points * (CeedSize)q_comp));
    CeedCallHip(ceed, hipMemset(d_v, 0, length * sizeof(CeedScalar)));
  }

  // Basis action
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      CeedInt P_1d, Q_1d;
      CeedInt block_size = data->block_sizes[0];

      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      CeedInt thread_1d     = CeedIntMax(Q_1d, P_1d);
      void   *interp_args[] = {(void *)&num_elem, &data->d_chebyshev_interp_1d, &data->d_points_per_elem, &d_x, &d_u, &d_v};

      if (dim == 1) {
        CeedInt elems_per_block = 64 * thread_1d > 256 ? 256 / thread_1d : 64;
        elems_per_block         = elems_per_block > 0 ? elems_per_block : 1;
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, is_transpose ? data->InterpTransposeAtPoints : data->InterpAtPoints, grid, thread_1d, 1,
                                                   elems_per_block, shared_mem, interp_args));
      } else if (dim == 2) {
        // Check if required threads is small enough to do multiple elems
        const CeedInt elems_per_block = CeedIntMax(block_size / (thread_1d * thread_1d), 1);
        CeedInt       grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, is_transpose ? data->InterpTransposeAtPoints : data->InterpAtPoints, grid, thread_1d,
                                                   thread_1d, elems_per_block, shared_mem, interp_args));
      } else if (dim == 3) {
        const CeedInt elems_per_block = 1;
        CeedInt       grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, is_transpose ? data->InterpTransposeAtPoints : data->InterpAtPoints, grid, thread_1d,
                                                   thread_1d, elems_per_block, shared_mem, interp_args));
      }
    } break;
    case CEED_EVAL_GRAD: {
      CeedInt P_1d, Q_1d;
      CeedInt block_size = data->block_sizes[0];

      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      CeedInt thread_1d   = CeedIntMax(Q_1d, P_1d);
      void   *grad_args[] = {(void *)&num_elem, &data->d_chebyshev_interp_1d, &data->d_points_per_elem, &d_x, &d_u, &d_v};

      if (dim == 1) {
        CeedInt elems_per_block = 64 * thread_1d > 256 ? 256 / thread_1d : 64;
        elems_per_block         = elems_per_block > 0 ? elems_per_block : 1;
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, is_transpose ? data->GradTransposeAtPoints : data->GradAtPoints, grid, thread_1d, 1,
                                                   elems_per_block, shared_mem, grad_args));
      } else if (dim == 2) {
        // Check if required threads is small enough to do multiple elems
        const CeedInt elems_per_block = CeedIntMax(block_size / (thread_1d * thread_1d), 1);
        CeedInt       grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, is_transpose ? data->GradTransposeAtPoints : data->GradAtPoints, grid, thread_1d, thread_1d,
                                                   elems_per_block, shared_mem, grad_args));
      } else if (dim == 3) {
        const CeedInt elems_per_block = 1;
        CeedInt       grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt       shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, is_transpose ? data->GradTransposeAtPoints : data->GradAtPoints, grid, thread_1d, thread_1d,
                                                   elems_per_block, shared_mem, grad_args));
      }
    } break;
    case CEED_EVAL_WEIGHT:
    case CEED_EVAL_NONE: /* handled separately below */
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      return CeedError(ceed, CEED_ERROR_BACKEND, "%s not supported", CeedEvalModes[eval_mode]);
      // LCOV_EXCL_STOP
  }

  // Restore vectors, cover CEED_EVAL_NONE
  CeedCallBackend(CeedVectorRestoreArrayRead(x_ref, &d_x));
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  if (eval_mode == CEED_EVAL_NONE) CeedCallBackend(CeedVectorSetArray(v, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)d_u));
  if (eval_mode != CEED_EVAL_WEIGHT) CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAtPoints_Hip_shared(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
                                             CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyAtPointsCore_Hip_shared(basis, false, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAddAtPoints_Hip_shared(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
                                                CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyAtPointsCore_Hip_shared(basis, true, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply basis
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorCore_Hip_shared(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode,
                                                  CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  Ceed                  ceed;
  Ceed_Hip             *ceed_Hip;
  CeedInt               dim, num_comp;
  const CeedScalar     *d_u;
  CeedScalar           *d_v;
  CeedBasis_Hip_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_Hip));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));

  // Get read/write access to u, v
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(eval_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  if (apply_add) CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  else CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Apply basis operation
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      CeedInt P, Q;

      CeedCallBackend(CeedBasisGetNumNodes(basis, &P));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &Q));
      CeedInt thread        = CeedIntMax(Q, P);
      void   *interp_args[] = {(void *)&num_elem, &data->d_interp_1d, &d_u, &d_v};

      {
        CeedInt elems_per_block = 64 * thread > 256 ? 256 / thread : 64;
        elems_per_block         = elems_per_block > 0 ? elems_per_block : 1;
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread, 1,
                                                     elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, data->Interp, grid, thread, 1, elems_per_block, shared_mem, interp_args));
        }
      }
    } break;
    case CEED_EVAL_GRAD: {
      CeedInt P, Q;

      CeedCallBackend(CeedBasisGetNumNodes(basis, &P));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &Q));
      CeedInt thread      = CeedIntMax(Q, P);
      void   *grad_args[] = {(void *)&num_elem, &data->d_grad_1d, &d_u, &d_v};

      {
        CeedInt elems_per_block = 64 * thread > 256 ? 256 / thread : 64;
        elems_per_block         = elems_per_block > 0 ? elems_per_block : 1;
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread, 1, elems_per_block,
                                                     shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, data->Grad, grid, thread, 1, elems_per_block, shared_mem, grad_args));
        }
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedInt Q;
      CeedInt block_size = data->block_sizes[2];

      CeedCheck(data->d_q_weight_1d, ceed, CEED_ERROR_BACKEND, "%s not supported; q_weights_1d not set", CeedEvalModes[eval_mode]);
      CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &Q));
      void *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight_1d, &d_v};

      {
        const CeedInt opt_elems       = block_size / Q;
        const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
        const CeedInt grid_size       = num_elem / elems_per_block + (num_elem % elems_per_block > 0);

        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->Weight, grid_size, Q, elems_per_block, 1, weight_args));
      }
    } break;
    case CEED_EVAL_NONE: /* handled separately below */
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      return CeedError(ceed, CEED_ERROR_BACKEND, "%s not supported", CeedEvalModes[eval_mode]);
      // LCOV_EXCL_STOP
  }

  // Restore vectors, cover CEED_EVAL_NONE
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  if (eval_mode == CEED_EVAL_NONE) CeedCallBackend(CeedVectorSetArray(v, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)d_u));
  if (eval_mode != CEED_EVAL_WEIGHT) CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisApplyNonTensor_Hip_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                       CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Hip_shared(basis, false, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisApplyAddNonTensor_Hip_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                          CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Hip_shared(basis, true, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Hip_shared(CeedBasis basis) {
  Ceed                  ceed;
  CeedBasis_Hip_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallHip(ceed, hipModuleUnload(data->module));
  if (data->moduleAtPoints) CeedCallHip(ceed, hipModuleUnload(data->moduleAtPoints));
  if (data->d_q_weight_1d) CeedCallHip(ceed, hipFree(data->d_q_weight_1d));
  CeedCallBackend(CeedFree(&data->h_points_per_elem));
  if (data->d_points_per_elem) CeedCallHip(ceed, hipFree(data->d_points_per_elem));
  CeedCallHip(ceed, hipFree(data->d_interp_1d));
  CeedCallHip(ceed, hipFree(data->d_grad_1d));
  CeedCallHip(ceed, hipFree(data->d_collo_grad_1d));
  CeedCallHip(ceed, hipFree(data->d_chebyshev_interp_1d));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor basis
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Hip_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                       const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed                  ceed;
  CeedInt               num_comp;
  const CeedInt         q_bytes      = Q_1d * sizeof(CeedScalar);
  const CeedInt         interp_bytes = q_bytes * P_1d;
  CeedBasis_Hip_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  if (q_weight_1d) {
    CeedCallHip(ceed, hipMalloc((void **)&data->d_q_weight_1d, q_bytes));
    CeedCallHip(ceed, hipMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes, hipMemcpyHostToDevice));
  }
  CeedCallHip(ceed, hipMalloc((void **)&data->d_interp_1d, interp_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_interp_1d, interp_1d, interp_bytes, hipMemcpyHostToDevice));
  CeedCallHip(ceed, hipMalloc((void **)&data->d_grad_1d, interp_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_grad_1d, grad_1d, interp_bytes, hipMemcpyHostToDevice));

  // Compute collocated gradient and copy to GPU
  data->d_collo_grad_1d    = NULL;
  bool has_collocated_grad = dim == 3 && Q_1d >= P_1d;

  if (has_collocated_grad) {
    CeedScalar *collo_grad_1d;

    CeedCallBackend(CeedMalloc(Q_1d * Q_1d, &collo_grad_1d));
    CeedCallBackend(CeedBasisGetCollocatedGrad(basis, collo_grad_1d));
    CeedCallHip(ceed, hipMalloc((void **)&data->d_collo_grad_1d, q_bytes * Q_1d));
    CeedCallHip(ceed, hipMemcpy(data->d_collo_grad_1d, collo_grad_1d, q_bytes * Q_1d, hipMemcpyHostToDevice));
    CeedCallBackend(CeedFree(&collo_grad_1d));
  }

  // Set number of threads per block for basis kernels
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(ComputeBasisThreadBlockSizes(dim, P_1d, Q_1d, num_comp, data->block_sizes));

  // Compile basis kernels
  const char basis_kernel_source[] = "// Tensor basis source\n#include <ceed/jit-source/hip/hip-shared-basis-tensor.h>\n";

  CeedCallBackend(CeedCompile_Hip(ceed, basis_kernel_source, &data->module, 11, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "T_1D",
                                  CeedIntMax(Q_1d, P_1d), "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp, "BASIS_NUM_NODES", CeedIntPow(P_1d, dim),
                                  "BASIS_NUM_QPTS", CeedIntPow(Q_1d, dim), "BASIS_INTERP_BLOCK_SIZE", data->block_sizes[0], "BASIS_GRAD_BLOCK_SIZE",
                                  data->block_sizes[1], "BASIS_WEIGHT_BLOCK_SIZE", data->block_sizes[2], "BASIS_HAS_COLLOCATED_GRAD",
                                  has_collocated_grad));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "InterpTransposeAdd", &data->InterpTransposeAdd));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Grad", &data->Grad));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "GradTranspose", &data->GradTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "GradTransposeAdd", &data->GradTransposeAdd));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Weight", &data->Weight));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyTensor_Hip_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddTensor_Hip_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPoints_Hip_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAddAtPoints", CeedBasisApplyAddAtPoints_Hip_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Hip_shared));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor basis
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Hip_shared(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                 const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                  ceed;
  CeedInt               num_comp, q_comp_interp, q_comp_grad;
  const CeedInt         q_bytes = num_qpts * sizeof(CeedScalar);
  CeedBasis_Hip_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  // Check shared memory size
  {
    Ceed_Hip *hip_data;

    CeedCallBackend(CeedGetData(ceed, &hip_data));
    if (((size_t)num_nodes * (size_t)num_qpts * (size_t)dim + (size_t)CeedIntMax(num_nodes, num_qpts)) * sizeof(CeedScalar) >
        hip_data->device_prop.sharedMemPerBlock) {
      CeedCallBackend(CeedBasisCreateH1Fallback(ceed, topo, dim, num_nodes, num_qpts, interp, grad, q_ref, q_weight, basis));
      return CEED_ERROR_SUCCESS;
    }
  }

  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_GRAD, &q_comp_grad));
  if (q_weight) {
    CeedCallHip(ceed, hipMalloc((void **)&data->d_q_weight_1d, q_bytes));
    CeedCallHip(ceed, hipMemcpy(data->d_q_weight_1d, q_weight, q_bytes, hipMemcpyHostToDevice));
  }
  if (interp) {
    const CeedInt interp_bytes = q_bytes * num_nodes * q_comp_interp;

    CeedCallHip(ceed, hipMalloc((void **)&data->d_interp_1d, interp_bytes));
    CeedCallHip(ceed, hipMemcpy(data->d_interp_1d, interp, interp_bytes, hipMemcpyHostToDevice));
  }
  if (grad) {
    const CeedInt grad_bytes = q_bytes * num_nodes * q_comp_grad;

    CeedCallHip(ceed, hipMalloc((void **)&data->d_grad_1d, grad_bytes));
    CeedCallHip(ceed, hipMemcpy(data->d_grad_1d, grad, grad_bytes, hipMemcpyHostToDevice));
  }

  // Compile basis kernels
  const char basis_kernel_source[] = "// Non-tensor basis source\n#include <ceed/jit-source/hip/hip-shared-basis-nontensor.h>\n";

  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(ComputeBasisThreadBlockSizes(dim, num_nodes, num_qpts, num_comp, data->block_sizes));
  CeedCallBackend(CeedCompile_Hip(ceed, basis_kernel_source, &data->module, 6, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "T_1D",
                                  CeedIntMax(num_qpts, num_nodes), "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp, "BASIS_INTERP_BLOCK_SIZE",
                                  data->block_sizes[0]));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "InterpTransposeAdd", &data->InterpTransposeAdd));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Grad", &data->Grad));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "GradTranspose", &data->GradTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "GradTransposeAdd", &data->GradTransposeAdd));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Weight", &data->Weight));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Hip_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Hip_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Hip_shared));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
