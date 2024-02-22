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
int CeedBasisApplyTensor_Cuda_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                     CeedVector v) {
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
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

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
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->InterpTranspose, grid, thread_1d, 1, elems_per_block, shared_mem, interp_args));
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
          CeedCallBackend(
              CeedRunKernelDimShared_Cuda(ceed, data->InterpTranspose, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
        }
      } else if (dim == 3) {
        CeedInt elems_per_block = 1;
        CeedInt grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(
              CeedRunKernelDimShared_Cuda(ceed, data->InterpTranspose, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
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
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->GradTranspose, grid, thread_1d, 1, elems_per_block, shared_mem, grad_args));
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
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->GradTranspose, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        }
      } else if (dim == 3) {
        CeedInt elems_per_block = 1;
        CeedInt grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
        CeedInt shared_mem      = elems_per_block * thread_1d * thread_1d * sizeof(CeedScalar);

        if (t_mode == CEED_TRANSPOSE) {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->GradTranspose, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        } else {
          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
        }
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedInt Q_1d;
      CeedInt block_size = 32;

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

//------------------------------------------------------------------------------
// Destroy basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Cuda_shared(CeedBasis basis) {
  Ceed                   ceed;
  CeedBasis_Cuda_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallCuda(ceed, cuModuleUnload(data->module));
  CeedCallCuda(ceed, cudaFree(data->d_q_weight_1d));
  CeedCallCuda(ceed, cudaFree(data->d_interp_1d));
  CeedCallCuda(ceed, cudaFree(data->d_grad_1d));
  CeedCallCuda(ceed, cudaFree(data->d_collo_grad_1d));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor basis
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                        const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed                   ceed;
  char                  *basis_kernel_path, *basis_kernel_source;
  CeedInt                num_comp;
  const CeedInt          q_bytes      = Q_1d * sizeof(CeedScalar);
  const CeedInt          interp_bytes = q_bytes * P_1d;
  CeedBasis_Cuda_shared *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight_1d, q_bytes));
  CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes, cudaMemcpyHostToDevice));
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
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Grad", &data->Grad));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "GradTranspose", &data->GradTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyTensor_Cuda_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda_shared));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
