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

#include "../cuda/ceed-cuda-common.h"
#include "../cuda/ceed-cuda-compile.h"
#include "ceed-cuda-shared.h"

//------------------------------------------------------------------------------
// Device initalization
//------------------------------------------------------------------------------
int CeedInit_CudaInterp(CeedScalar *d_B, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B);
int CeedInit_CudaGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr);
int CeedInit_CudaCollocatedGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr);

//------------------------------------------------------------------------------
// Apply basis
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

      CeedCallBackend(CeedInit_CudaInterp(data->d_interp_1d, P_1d, Q_1d, &data->c_B));
      void *interp_args[] = {(void *)&num_elem, &data->c_B, &d_u, &d_v};

      if (dim == 1) {
        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread_1d,
                                                                                                 1));  // avoid >512 total threads
        CeedInt grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
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
        CeedInt grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, thread_1d,
                                                      elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
        }
      } else if (dim == 3) {
        CeedInt elems_per_block = 1;
        CeedInt grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
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
      CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);

      if (data->d_collo_grad_1d) {
        CeedCallBackend(CeedInit_CudaCollocatedGrad(data->d_interp_1d, data->d_collo_grad_1d, P_1d, Q_1d, &data->c_B, &data->c_G));
      } else {
        CeedCallBackend(CeedInit_CudaGrad(data->d_interp_1d, data->d_grad_1d, P_1d, Q_1d, &data->c_B, &data->c_G));
      }
      void *grad_args[] = {(void *)&num_elem, &data->c_B, &data->c_G, &d_u, &d_v};
      if (dim == 1) {
        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread_1d,
                                                                                                 1));  // avoid >512 total threads
        CeedInt grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
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
        CeedInt grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, thread_1d,
                                                      elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        }
      } else if (dim == 3) {
        CeedInt elems_per_block = 1;
        CeedInt grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
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
        const CeedInt grid_size       = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);

        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid_size, Q_1d, elems_per_block, 1, weight_args));
      } else if (dim == 2) {
        const CeedInt opt_elems       = block_size / (Q_1d * Q_1d);
        const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
        const CeedInt grid_size       = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);

        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid_size, Q_1d, Q_1d, elems_per_block, weight_args));
      } else if (dim == 3) {
        const CeedInt opt_elems       = block_size / (Q_1d * Q_1d);
        const CeedInt elems_per_block = opt_elems > 0 ? opt_elems : 1;
        const CeedInt grid_size       = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);

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
  CeedInt                Q_1d, dim, max_num_points = num_points[0];
  const CeedInt          is_transpose   = t_mode == CEED_TRANSPOSE;
  const int              max_block_size = 32;
  const CeedScalar      *d_x, *d_u;
  CeedScalar            *d_v;
  CeedBasis_Cuda_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));

  // Check uniform number of points per elem
  for (CeedInt i = 1; i < num_elem; i++) {
    CeedCheck(max_num_points == num_points[i], ceed, CEED_ERROR_BACKEND,
              "BasisApplyAtPoints only supported for the same number of points in each element");
  }

  // Weight handled separately
  if (eval_mode == CEED_EVAL_WEIGHT) {
    CeedCall(CeedVectorSetValue(v, 1.0));
    return CEED_ERROR_SUCCESS;
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
      CeedCall(CeedBasisGetChebyshevInterp1D(basis, chebyshev_interp_1d));
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
      void         *interp_args[] = {(void *)&num_elem, (void *)&is_transpose, &data->d_chebyshev_interp_1d, &d_x, &d_u, &d_v};
      const CeedInt block_size    = CeedIntMin(CeedIntPow(Q_1d, dim), max_block_size);

      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->InterpAtPoints, num_elem, block_size, interp_args));
    } break;
    case CEED_EVAL_GRAD: {
      void         *grad_args[] = {(void *)&num_elem, (void *)&is_transpose, &data->d_chebyshev_interp_1d, &d_x, &d_u, &d_v};
      const CeedInt block_size  = CeedIntMin(CeedIntPow(Q_1d, dim), max_block_size);

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
  CeedCallCuda(ceed, cudaFree(data->d_interp_1d));
  CeedCallCuda(ceed, cudaFree(data->d_grad_1d));
  CeedCallCuda(ceed, cudaFree(data->d_collo_grad_1d));
  CeedCallCuda(ceed, cudaFree(data->d_chebyshev_interp_1d));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor basis
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                        const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed                   ceed;
  char                  *basis_kernel_source;
  const char            *basis_kernel_path;
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
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-shared-basis-tensor.h", &basis_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete -----\n");
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
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyTensor_Cuda_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddTensor_Cuda_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPoints_Cuda_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda_shared));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
