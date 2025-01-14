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
#include "ceed-cuda-shared.h"

//------------------------------------------------------------------------------
// Apply tensor basis
//------------------------------------------------------------------------------
static int CeedBasisApplyTensorCore_Cuda_shared(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode,
                                                CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  Ceed                   ceed;
  Ceed_Cuda             *ceed_Cuda;
  CeedInt                dim, num_comp;
  const CeedScalar      *d_u;
  CeedScalar            *d_v;
  CeedBasis_Cuda_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_Cuda));
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

      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);

      void *interp_args[] = {(void *)&num_elem, &data->d_interp_1d, &d_u, &d_v};

      if (dim == 1) {
        // avoid >512 total threads
        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread_1d, 1));
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, 1,
                                                      elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread_1d, 1, elems_per_block, shared_mem, interp_args));
        }
      } else if (dim == 2) {
        const CeedInt opt_elems[7] = {0, 32, 8, 6, 4, 2, 8};
        // elems_per_block must be at least 1
        CeedInt elems_per_block = CeedIntMax(thread_1d < 7 ? opt_elems[thread_1d] / num_comp : 1, 1);
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, thread_1d,
                                                      elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
        }
      } else if (dim == 3) {
        CeedInt elems_per_block = 1;
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, thread_1d,
                                                      elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
        }
      }
    } break;
    case CEED_EVAL_GRAD: {
      CeedInt P_1d, Q_1d;

      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      CeedInt     thread_1d = CeedIntMax(Q_1d, P_1d);
      CeedScalar *d_grad_1d = data->d_grad_1d;

      if (data->d_collo_grad_1d) {
        d_grad_1d = data->d_collo_grad_1d;
      }
      void *grad_args[] = {(void *)&num_elem, &data->d_interp_1d, &d_grad_1d, &d_u, &d_v};

      if (dim == 1) {
        // avoid >512 total threads
        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread_1d, 1));
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, 1,
                                                      elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, 1, elems_per_block, shared_mem, grad_args));
        }
      } else if (dim == 2) {
        const CeedInt opt_elems[7] = {0, 32, 8, 6, 4, 2, 8};
        // elems_per_block must be at least 1
        CeedInt elems_per_block = CeedIntMax(thread_1d < 7 ? opt_elems[thread_1d] / num_comp : 1, 1);
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, thread_1d,
                                                      elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        }
      } else if (dim == 3) {
        CeedInt elems_per_block = 1;
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, thread_1d,
                                                      elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        }
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedInt Q_1d;
      CeedInt block_size = 32;

      CeedCheck(data->d_q_weight_1d, ceed, CEED_ERROR_BACKEND, "%s not supported; q_weights_1d not set", CeedEvalModes[eval_mode]);
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      void *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight_1d, &d_v};
      if (dim == 1) {
        const CeedInt elems_per_block = block_size / Q_1d;
        const CeedInt grid_size       = num_elem / elems_per_block + (num_elem % elems_per_block > 0);

        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid_size, Q_1d, elems_per_block, 1, weight_args));
      } else if (dim == 2) {
        const CeedInt opt_elems       = block_size / (Q_1d * Q_1d);
        const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
        const CeedInt grid_size       = num_elem / elems_per_block + (num_elem % elems_per_block > 0);

        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid_size, Q_1d, Q_1d, elems_per_block, weight_args));
      } else if (dim == 3) {
        const CeedInt opt_elems       = block_size / (Q_1d * Q_1d);
        const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
        const CeedInt grid_size       = num_elem / elems_per_block + (num_elem % elems_per_block > 0);

        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid_size, Q_1d, Q_1d, elems_per_block, weight_args));
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

static int CeedBasisApplyTensor_Cuda_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                            CeedVector v) {
  CeedCallBackend(CeedBasisApplyTensorCore_Cuda_shared(basis, false, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAddTensor_Cuda_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                                               CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyTensorCore_Cuda_shared(basis, true, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - tensor AtPoints
//------------------------------------------------------------------------------
static int CeedBasisApplyAtPointsCore_Cuda_shared(CeedBasis basis, bool apply_add, const CeedInt num_elem, const CeedInt *num_points,
                                                  CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  Ceed                   ceed;
  Ceed_Cuda             *ceed_Cuda;
  CeedInt                Q_1d, dim, num_comp, max_num_points = num_points[0];
  const CeedInt          is_transpose = t_mode == CEED_TRANSPOSE;
  const CeedScalar      *d_x, *d_u;
  CeedScalar            *d_v;
  CeedBasis_Cuda_shared *data;

  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));

  // Weight handled separately
  if (eval_mode == CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorSetValue(v, 1.0));
    return CEED_ERROR_SUCCESS;
  }

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_Cuda));

  // Check padded to uniform number of points per elem
  for (CeedInt i = 1; i < num_elem; i++) max_num_points = CeedIntMax(max_num_points, num_points[i]);
  {
    CeedInt  q_comp;
    CeedSize len, len_required;
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

      if (data->d_points_per_elem) CeedCallCuda(ceed, cudaFree(data->d_points_per_elem));
      CeedCallCuda(ceed, cudaMalloc((void **)&data->d_points_per_elem, num_bytes));
      CeedCallBackend(CeedFree(&data->h_points_per_elem));
      CeedCallBackend(CeedCalloc(num_elem, &data->h_points_per_elem));
    }
    if (memcmp(data->h_points_per_elem, num_points, num_bytes)) {
      memcpy(data->h_points_per_elem, num_points, num_bytes);
      CeedCallCuda(ceed, cudaMemcpy(data->d_points_per_elem, num_points, num_bytes, cudaMemcpyHostToDevice));
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
      CeedCallCuda(ceed, cudaMalloc((void **)&data->d_chebyshev_interp_1d, interp_bytes));
      CeedCallCuda(ceed, cudaMemcpy(data->d_chebyshev_interp_1d, chebyshev_interp_1d, interp_bytes, cudaMemcpyHostToDevice));
      CeedCallBackend(CeedFree(&chebyshev_interp_1d));
    }

    // -- Compile kernels
    const char basis_kernel_source[] = "// AtPoints basis source\n#include <ceed/jit-source/cuda/cuda-shared-basis-tensor-at-points.h>\n";
    CeedInt    num_comp;

    if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
    CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
    CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->moduleAtPoints, 8, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "T_1D",
                                     CeedIntMax(Q_1d, P_1d), "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp, "BASIS_NUM_NODES", CeedIntPow(P_1d, dim),
                                     "BASIS_NUM_QPTS", CeedIntPow(Q_1d, dim), "BASIS_NUM_PTS", max_num_points));
    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "InterpAtPoints", &data->InterpAtPoints));
    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "InterpTransposeAtPoints", &data->InterpTransposeAtPoints));
    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "GradAtPoints", &data->GradAtPoints));
    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "GradTransposeAtPoints", &data->GradTransposeAtPoints));
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
    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
  }

  // Basis action
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      CeedInt P_1d, Q_1d;

      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);

      void *interp_args[] = {(void *)&num_elem, &data->d_chebyshev_interp_1d, &data->d_points_per_elem, &d_x, &d_u, &d_v};

      if (dim == 1) {
        // avoid >512 total threads
        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread_1d, 1));
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, is_transpose ? data->InterpTransposeAtPoints : data->InterpAtPoints, grid, thread_1d, 1,
                                                    elems_per_block, shared_mem, interp_args));
      } else if (dim == 2) {
        const CeedInt opt_elems[7] = {0, 32, 8, 6, 4, 2, 8};
        // elems_per_block must be at least 1
        CeedInt elems_per_block = CeedIntMax(thread_1d < 7 ? opt_elems[thread_1d] / num_comp : 1, 1);
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, is_transpose ? data->InterpTransposeAtPoints : data->InterpAtPoints, grid, thread_1d,
                                                    thread_1d, elems_per_block, shared_mem, interp_args));
      } else if (dim == 3) {
        CeedInt elems_per_block = 1;
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, is_transpose ? data->InterpTransposeAtPoints : data->InterpAtPoints, grid, thread_1d,
                                                    thread_1d, elems_per_block, shared_mem, interp_args));
      }
    } break;
    case CEED_EVAL_GRAD: {
      CeedInt P_1d, Q_1d;

      CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);

      void *grad_args[] = {(void *)&num_elem, &data->d_chebyshev_interp_1d, &data->d_points_per_elem, &d_x, &d_u, &d_v};

      if (dim == 1) {
        // avoid >512 total threads
        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread_1d, 1));
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, is_transpose ? data->GradTransposeAtPoints : data->GradAtPoints, grid, thread_1d, 1,
                                                    elems_per_block, shared_mem, grad_args));
      } else if (dim == 2) {
        const CeedInt opt_elems[7] = {0, 32, 8, 6, 4, 2, 8};
        // elems_per_block must be at least 1
        CeedInt elems_per_block = CeedIntMax(thread_1d < 7 ? opt_elems[thread_1d] / num_comp : 1, 1);
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, is_transpose ? data->GradTransposeAtPoints : data->GradAtPoints, grid, thread_1d, thread_1d,
                                                    elems_per_block, shared_mem, grad_args));
      } else if (dim == 3) {
        CeedInt elems_per_block = 1;
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, is_transpose ? data->GradTransposeAtPoints : data->GradAtPoints, grid, thread_1d, thread_1d,
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
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAtPoints_Cuda_shared(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
                                              CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda_shared(basis, false, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAddAtPoints_Cuda_shared(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
                                                 CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda_shared(basis, true, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorCore_Cuda_shared(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode,
                                                   CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  Ceed                   ceed;
  Ceed_Cuda             *ceed_Cuda;
  CeedInt                dim;
  const CeedScalar      *d_u;
  CeedScalar            *d_v;
  CeedBasis_Cuda_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_Cuda));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));

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
      CeedInt thread = CeedIntMax(Q, P);

      void *interp_args[] = {(void *)&num_elem, &data->d_interp_1d, &d_u, &d_v};

      {
        // avoid >512 total threads
        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread, 1));
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread, 1,
                                                      elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread, 1, elems_per_block, shared_mem, interp_args));
        }
      }
    } break;
    case CEED_EVAL_GRAD: {
      CeedInt P, Q;

      CeedCallBackend(CeedBasisGetNumNodes(basis, &P));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &Q));
      CeedInt thread = CeedIntMax(Q, P);

      void *grad_args[] = {(void *)&num_elem, &data->d_grad_1d, &d_u, &d_v};

      {
        // avoid >512 total threads
        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread, 1));
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);
        CeedInt shared_mem      = elems_per_block * thread * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread, 1,
                                                      elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread, 1, elems_per_block, shared_mem, grad_args));
        }
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedInt Q;

      CeedCheck(data->d_q_weight_1d, ceed, CEED_ERROR_BACKEND, "%s not supported; q_weights_1d not set", CeedEvalModes[eval_mode]);
      CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &Q));
      void *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight_1d, &d_v};

      {
        // avoid >512 total threads
        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / Q, 1));
        CeedInt grid            = num_elem / elems_per_block + (num_elem % elems_per_block > 0);

        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid, Q, elems_per_block, 1, weight_args));
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

static int CeedBasisApplyNonTensor_Cuda_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                                               CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Cuda_shared(basis, false, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAddNonTensor_Cuda_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                                                  CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Cuda_shared(basis, true, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Cuda_shared(CeedBasis basis) {
  Ceed                   ceed;
  CeedBasis_Cuda_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallCuda(ceed, cuModuleUnload(data->module));
  if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
  if (data->d_q_weight_1d) CeedCallCuda(ceed, cudaFree(data->d_q_weight_1d));
  CeedCallBackend(CeedFree(&data->h_points_per_elem));
  if (data->d_points_per_elem) CeedCallCuda(ceed, cudaFree(data->d_points_per_elem));
  CeedCallCuda(ceed, cudaFree(data->d_interp_1d));
  CeedCallCuda(ceed, cudaFree(data->d_grad_1d));
  CeedCallCuda(ceed, cudaFree(data->d_collo_grad_1d));
  CeedCallCuda(ceed, cudaFree(data->d_chebyshev_interp_1d));
  CeedCallBackend(CeedFree(&data));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor basis
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                        const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed                   ceed;
  CeedInt                num_comp;
  const CeedInt          q_bytes      = Q_1d * sizeof(CeedScalar);
  const CeedInt          interp_bytes = q_bytes * P_1d;
  CeedBasis_Cuda_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  if (q_weight_1d) {
    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight_1d, q_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes, cudaMemcpyHostToDevice));
  }
  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp_1d, interp_bytes));
  CeedCallCuda(ceed, cudaMemcpy(data->d_interp_1d, interp_1d, interp_bytes, cudaMemcpyHostToDevice));
  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad_1d, interp_bytes));
  CeedCallCuda(ceed, cudaMemcpy(data->d_grad_1d, grad_1d, interp_bytes, cudaMemcpyHostToDevice));

  // Compute collocated gradient and copy to GPU
  data->d_collo_grad_1d    = NULL;
  bool has_collocated_grad = dim == 3 && Q_1d >= P_1d;

  if (has_collocated_grad) {
    CeedScalar *collo_grad_1d;

    CeedCallBackend(CeedMalloc(Q_1d * Q_1d, &collo_grad_1d));
    CeedCallBackend(CeedBasisGetCollocatedGrad(basis, collo_grad_1d));
    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_collo_grad_1d, q_bytes * Q_1d));
    CeedCallCuda(ceed, cudaMemcpy(data->d_collo_grad_1d, collo_grad_1d, q_bytes * Q_1d, cudaMemcpyHostToDevice));
    CeedCallBackend(CeedFree(&collo_grad_1d));
  }

  // Compile basis kernels
  const char basis_kernel_source[] = "// Tensor basis source\n#include <ceed/jit-source/cuda/cuda-shared-basis-tensor.h>\n";

  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 8, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "T_1D",
                                   CeedIntMax(Q_1d, P_1d), "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp, "BASIS_NUM_NODES", CeedIntPow(P_1d, dim),
                                   "BASIS_NUM_QPTS", CeedIntPow(Q_1d, dim), "BASIS_HAS_COLLOCATED_GRAD", has_collocated_grad));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTransposeAdd", &data->InterpTransposeAdd));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Grad", &data->Grad));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "GradTranspose", &data->GradTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "GradTransposeAdd", &data->GradTransposeAdd));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyTensor_Cuda_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddTensor_Cuda_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPoints_Cuda_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAddAtPoints", CeedBasisApplyAddAtPoints_Cuda_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda_shared));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor basis
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Cuda_shared(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                  const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                   ceed;
  CeedInt                num_comp, q_comp_interp, q_comp_grad;
  const CeedInt          q_bytes = num_qpts * sizeof(CeedScalar);
  CeedBasis_Cuda_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  // Check shared memory size
  {
    Ceed_Cuda *cuda_data;

    CeedCallBackend(CeedGetData(ceed, &cuda_data));
    if (((size_t)num_nodes * (size_t)num_qpts * (size_t)dim + (size_t)CeedIntMax(num_nodes, num_qpts)) * sizeof(CeedScalar) >
        cuda_data->device_prop.sharedMemPerBlock) {
      CeedCallBackend(CeedBasisCreateH1Fallback(ceed, topo, dim, num_nodes, num_qpts, interp, grad, q_ref, q_weight, basis));
      CeedCallBackend(CeedDestroy(&ceed));
      return CEED_ERROR_SUCCESS;
    }
  }

  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_GRAD, &q_comp_grad));
  if (q_weight) {
    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight_1d, q_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight_1d, q_weight, q_bytes, cudaMemcpyHostToDevice));
  }
  if (interp) {
    const CeedInt interp_bytes = q_bytes * num_nodes * q_comp_interp;

    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp_1d, interp_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_interp_1d, interp, interp_bytes, cudaMemcpyHostToDevice));
  }
  if (grad) {
    const CeedInt grad_bytes = q_bytes * num_nodes * q_comp_grad;

    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad_1d, grad_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_grad_1d, grad, grad_bytes, cudaMemcpyHostToDevice));
  }

  // Compile basis kernels
  const char basis_kernel_source[] = "// Non-tensor basis source\n#include <ceed/jit-source/cuda/cuda-shared-basis-nontensor.h>\n";

  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "T_1D",
                                   CeedIntMax(num_qpts, num_nodes), "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTransposeAdd", &data->InterpTransposeAdd));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Grad", &data->Grad));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "GradTranspose", &data->GradTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "GradTransposeAdd", &data->GradTransposeAdd));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda_shared));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
