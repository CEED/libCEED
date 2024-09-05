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
#include <string.h>

#include "../cuda/ceed-cuda-common.h"
#include "../cuda/ceed-cuda-compile.h"
#include "ceed-cuda-ref.h"

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyCore_Cuda(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                                   CeedVector u, CeedVector v) {
  Ceed              ceed;
  CeedInt           Q_1d, dim;
  const CeedInt     is_transpose   = t_mode == CEED_TRANSPOSE;
  const int         max_block_size = 32;
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  CeedBasis_Cuda   *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));

  // Get read/write access to u, v
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(eval_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  if (apply_add) CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  else CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (is_transpose && !apply_add) {
    CeedSize length;

    CeedCallBackend(CeedVectorGetLength(v, &length));
    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
  }
  CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));

  // Basis action
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      void         *interp_args[] = {(void *)&num_elem, (void *)&is_transpose, &data->d_interp_1d, &d_u, &d_v};
      const CeedInt block_size    = CeedIntMin(CeedIntPow(Q_1d, dim), max_block_size);

      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->Interp, num_elem, block_size, interp_args));
    } break;
    case CEED_EVAL_GRAD: {
      void         *grad_args[] = {(void *)&num_elem, (void *)&is_transpose, &data->d_interp_1d, &data->d_grad_1d, &d_u, &d_v};
      const CeedInt block_size  = max_block_size;

      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->Grad, num_elem, block_size, grad_args));
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedCheck(data->d_q_weight_1d, ceed, CEED_ERROR_BACKEND, "%s not supported; q_weights_1d not set", CeedEvalModes[eval_mode]);
      void     *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight_1d, &d_v};
      const int block_size_x  = Q_1d;
      const int block_size_y  = dim >= 2 ? Q_1d : 1;

      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, num_elem, block_size_x, block_size_y, 1, weight_args));
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
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApply_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                               CeedVector v) {
  CeedCallBackend(CeedBasisApplyCore_Cuda(basis, false, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAdd_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                  CeedVector v) {
  CeedCallBackend(CeedBasisApplyCore_Cuda(basis, true, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - tensor AtPoints
//------------------------------------------------------------------------------
static int CeedBasisApplyAtPointsCore_Cuda(CeedBasis basis, bool apply_add, const CeedInt num_elem, const CeedInt *num_points,
                                           CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  Ceed              ceed;
  CeedInt           Q_1d, dim, max_num_points = num_points[0];
  const CeedInt     is_transpose   = t_mode == CEED_TRANSPOSE;
  const int         max_block_size = 32;
  const CeedScalar *d_x, *d_u;
  CeedScalar       *d_v;
  CeedBasis_Cuda   *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));

  // Weight handled separately
  if (eval_mode == CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorSetValue(v, 1.0));
    return CEED_ERROR_SUCCESS;
  }

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
    char       *basis_kernel_source;
    const char *basis_kernel_path;
    CeedInt     num_comp;

    if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
    CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
    CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-basis-tensor-at-points.h", &basis_kernel_path));
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
    CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
    CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->moduleAtPoints, 9, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "BASIS_BUF_LEN",
                                     Q_1d * CeedIntPow(Q_1d > P_1d ? Q_1d : P_1d, dim - 1), "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp,
                                     "BASIS_NUM_NODES", CeedIntPow(P_1d, dim), "BASIS_NUM_QPTS", CeedIntPow(Q_1d, dim), "BASIS_NUM_PTS",
                                     max_num_points, "POINTS_BUFF_LEN", CeedIntPow(Q_1d, dim - 1)));
    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "InterpAtPoints", &data->InterpAtPoints));
    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "GradAtPoints", &data->GradAtPoints));
    CeedCallBackend(CeedFree(&basis_kernel_path));
    CeedCallBackend(CeedFree(&basis_kernel_source));
  }

  // Get read/write access to u, v
  CeedCallBackend(CeedVectorGetArrayRead(x_ref, CEED_MEM_DEVICE, &d_x));
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(eval_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  if (apply_add) CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  else CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (is_transpose && !apply_add) {
    CeedSize length;

    CeedCallBackend(CeedVectorGetLength(v, &length));
    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
  }

  // Basis action
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      void *interp_args[]      = {(void *)&num_elem, (void *)&is_transpose, &data->d_chebyshev_interp_1d, &data->d_points_per_elem, &d_x, &d_u, &d_v};
      const CeedInt block_size = CeedIntMin(CeedIntPow(Q_1d, dim), max_block_size);

      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->InterpAtPoints, num_elem, block_size, interp_args));
    } break;
    case CEED_EVAL_GRAD: {
      void *grad_args[]        = {(void *)&num_elem, (void *)&is_transpose, &data->d_chebyshev_interp_1d, &data->d_points_per_elem, &d_x, &d_u, &d_v};
      const CeedInt block_size = CeedIntMin(CeedIntPow(Q_1d, dim), max_block_size);

      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->GradAtPoints, num_elem, block_size, grad_args));
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

static int CeedBasisApplyAtPoints_Cuda(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
                                       CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda(basis, false, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAddAtPoints_Cuda(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
                                          CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda(basis, true, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - non-tensor
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorCore_Cuda(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                                            CeedVector u, CeedVector v) {
  Ceed                     ceed;
  CeedInt                  num_nodes, num_qpts;
  const CeedInt            is_transpose    = t_mode == CEED_TRANSPOSE;
  const int                elems_per_block = 1;
  const int                grid            = CeedDivUpInt(num_elem, elems_per_block);
  const CeedScalar        *d_u;
  CeedScalar              *d_v;
  CeedBasisNonTensor_Cuda *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));

  // Get read/write access to u, v
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(eval_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  if (apply_add) CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  else CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (is_transpose && !apply_add) {
    CeedSize length;

    CeedCallBackend(CeedVectorGetLength(v, &length));
    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
  }

  // Apply basis operation
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      void     *interp_args[] = {(void *)&num_elem, &data->d_interp, &d_u, &d_v};
      const int block_size_x  = is_transpose ? num_nodes : num_qpts;

      if (is_transpose) {
        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->InterpTranspose, grid, block_size_x, 1, elems_per_block, interp_args));
      } else {
        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Interp, grid, block_size_x, 1, elems_per_block, interp_args));
      }
    } break;
    case CEED_EVAL_GRAD: {
      void     *grad_args[]  = {(void *)&num_elem, &data->d_grad, &d_u, &d_v};
      const int block_size_x = is_transpose ? num_nodes : num_qpts;

      if (is_transpose) {
        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, grad_args));
      } else {
        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, grad_args));
      }
    } break;
    case CEED_EVAL_DIV: {
      void     *div_args[]   = {(void *)&num_elem, &data->d_div, &d_u, &d_v};
      const int block_size_x = is_transpose ? num_nodes : num_qpts;

      if (is_transpose) {
        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, div_args));
      } else {
        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, div_args));
      }
    } break;
    case CEED_EVAL_CURL: {
      void     *curl_args[]  = {(void *)&num_elem, &data->d_curl, &d_u, &d_v};
      const int block_size_x = is_transpose ? num_nodes : num_qpts;

      if (is_transpose) {
        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, curl_args));
      } else {
        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, curl_args));
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedCheck(data->d_q_weight, ceed, CEED_ERROR_BACKEND, "%s not supported; q_weights not set", CeedEvalModes[eval_mode]);
      void *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight, &d_v};

      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid, num_qpts, 1, elems_per_block, weight_args));
    } break;
    case CEED_EVAL_NONE: /* handled separately below */
      break;
  }

  // Restore vectors, cover CEED_EVAL_NONE
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  if (eval_mode == CEED_EVAL_NONE) CeedCallBackend(CeedVectorSetArray(v, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)d_u));
  if (eval_mode != CEED_EVAL_WEIGHT) CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyNonTensor_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                        CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Cuda(basis, false, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAddNonTensor_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                           CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Cuda(basis, true, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Cuda(CeedBasis basis) {
  Ceed            ceed;
  CeedBasis_Cuda *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallCuda(ceed, cuModuleUnload(data->module));
  if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
  if (data->d_q_weight_1d) CeedCallCuda(ceed, cudaFree(data->d_q_weight_1d));
  CeedCallBackend(CeedFree(&data->h_points_per_elem));
  if (data->d_points_per_elem) CeedCallCuda(ceed, cudaFree(data->d_points_per_elem));
  CeedCallCuda(ceed, cudaFree(data->d_interp_1d));
  CeedCallCuda(ceed, cudaFree(data->d_grad_1d));
  CeedCallCuda(ceed, cudaFree(data->d_chebyshev_interp_1d));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Cuda(CeedBasis basis) {
  Ceed                     ceed;
  CeedBasisNonTensor_Cuda *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallCuda(ceed, cuModuleUnload(data->module));
  if (data->d_q_weight) CeedCallCuda(ceed, cudaFree(data->d_q_weight));
  CeedCallCuda(ceed, cudaFree(data->d_interp));
  CeedCallCuda(ceed, cudaFree(data->d_grad));
  CeedCallCuda(ceed, cudaFree(data->d_div));
  CeedCallCuda(ceed, cudaFree(data->d_curl));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                 const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed            ceed;
  char           *basis_kernel_source;
  const char     *basis_kernel_path;
  CeedInt         num_comp;
  const CeedInt   q_bytes      = Q_1d * sizeof(CeedScalar);
  const CeedInt   interp_bytes = q_bytes * P_1d;
  CeedBasis_Cuda *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy data to GPU
  if (q_weight_1d) {
    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight_1d, q_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes, cudaMemcpyHostToDevice));
  }
  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp_1d, interp_bytes));
  CeedCallCuda(ceed, cudaMemcpy(data->d_interp_1d, interp_1d, interp_bytes, cudaMemcpyHostToDevice));
  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad_1d, interp_bytes));
  CeedCallCuda(ceed, cudaMemcpy(data->d_grad_1d, grad_1d, interp_bytes, cudaMemcpyHostToDevice));

  // Compile basis kernels
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-basis-tensor.h", &basis_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 7, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "BASIS_BUF_LEN",
                                   Q_1d * CeedIntPow(Q_1d > P_1d ? Q_1d : P_1d, dim - 1), "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp,
                                   "BASIS_NUM_NODES", CeedIntPow(P_1d, dim), "BASIS_NUM_QPTS", CeedIntPow(Q_1d, dim)));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Grad", &data->Grad));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAdd_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPoints_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAddAtPoints", CeedBasisApplyAddAtPoints_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H^1
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                           const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                     ceed;
  char                    *basis_kernel_source;
  const char              *basis_kernel_path;
  CeedInt                  num_comp, q_comp_interp, q_comp_grad;
  const CeedInt            q_bytes = num_qpts * sizeof(CeedScalar);
  CeedBasisNonTensor_Cuda *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_GRAD, &q_comp_grad));
  if (q_weight) {
    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight, q_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight, q_weight, q_bytes, cudaMemcpyHostToDevice));
  }
  if (interp) {
    const CeedInt interp_bytes = q_bytes * num_nodes * q_comp_interp;

    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp, interp_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_interp, interp, interp_bytes, cudaMemcpyHostToDevice));
  }
  if (grad) {
    const CeedInt grad_bytes = q_bytes * num_nodes * q_comp_grad;

    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad, grad_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_grad, grad, grad_bytes, cudaMemcpyHostToDevice));
  }

  // Compile basis kernels
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-basis-nontensor.h", &basis_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
                                   q_comp_interp, "BASIS_Q_COMP_DERIV", q_comp_grad, "BASIS_NUM_COMP", num_comp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Deriv", &data->Deriv));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H(div)
//------------------------------------------------------------------------------
int CeedBasisCreateHdiv_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *div,
                             const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                     ceed;
  char                    *basis_kernel_source;
  const char              *basis_kernel_path;
  CeedInt                  num_comp, q_comp_interp, q_comp_div;
  const CeedInt            q_bytes = num_qpts * sizeof(CeedScalar);
  CeedBasisNonTensor_Cuda *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_DIV, &q_comp_div));
  if (q_weight) {
    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight, q_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight, q_weight, q_bytes, cudaMemcpyHostToDevice));
  }
  if (interp) {
    const CeedInt interp_bytes = q_bytes * num_nodes * q_comp_interp;

    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp, interp_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_interp, interp, interp_bytes, cudaMemcpyHostToDevice));
  }
  if (div) {
    const CeedInt div_bytes = q_bytes * num_nodes * q_comp_div;

    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_div, div_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_div, div, div_bytes, cudaMemcpyHostToDevice));
  }

  // Compile basis kernels
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-basis-nontensor.h", &basis_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
                                   q_comp_interp, "BASIS_Q_COMP_DERIV", q_comp_div, "BASIS_NUM_COMP", num_comp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Deriv", &data->Deriv));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H(curl)
//------------------------------------------------------------------------------
int CeedBasisCreateHcurl_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                              const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                     ceed;
  char                    *basis_kernel_source;
  const char              *basis_kernel_path;
  CeedInt                  num_comp, q_comp_interp, q_comp_curl;
  const CeedInt            q_bytes = num_qpts * sizeof(CeedScalar);
  CeedBasisNonTensor_Cuda *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_CURL, &q_comp_curl));
  if (q_weight) {
    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight, q_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight, q_weight, q_bytes, cudaMemcpyHostToDevice));
  }
  if (interp) {
    const CeedInt interp_bytes = q_bytes * num_nodes * q_comp_interp;

    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp, interp_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_interp, interp, interp_bytes, cudaMemcpyHostToDevice));
  }
  if (curl) {
    const CeedInt curl_bytes = q_bytes * num_nodes * q_comp_curl;

    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_curl, curl_bytes));
    CeedCallCuda(ceed, cudaMemcpy(data->d_curl, curl, curl_bytes, cudaMemcpyHostToDevice));
  }

  // Compile basis kernels
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-basis-nontensor.h", &basis_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
                                   q_comp_interp, "BASIS_Q_COMP_DERIV", q_comp_curl, "BASIS_NUM_COMP", num_comp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Deriv", &data->Deriv));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
