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

#include "../cuda/ceed-cuda-compile.h"
#include "ceed-cuda-ref.h"

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
int CeedBasisApply_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  Ceed_Cuda *ceed_Cuda;
  CeedCallBackend(CeedGetData(ceed, &ceed_Cuda));
  CeedBasis_Cuda *data;
  CeedCallBackend(CeedBasisGetData(basis, &data));
  const CeedInt transpose      = t_mode == CEED_TRANSPOSE;
  const int     max_block_size = 32;

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  if (eval_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  }
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (t_mode == CEED_TRANSPOSE) {
    CeedSize length;
    CeedCallBackend(CeedVectorGetLength(v, &length));
    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
  }
  CeedInt Q_1d, dim;
  CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));

  // Basis action
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      void   *interp_args[] = {(void *)&num_elem, (void *)&transpose, &data->d_interp_1d, &d_u, &d_v};
      CeedInt block_size    = CeedIntMin(CeedIntPow(Q_1d, dim), max_block_size);

      CeedCallBackend(CeedRunKernelCuda(ceed, data->Interp, num_elem, block_size, interp_args));
    } break;
    case CEED_EVAL_GRAD: {
      void   *grad_args[] = {(void *)&num_elem, (void *)&transpose, &data->d_interp_1d, &data->d_grad_1d, &d_u, &d_v};
      CeedInt block_size  = max_block_size;

      CeedCallBackend(CeedRunKernelCuda(ceed, data->Grad, num_elem, block_size, grad_args));
    } break;
    case CEED_EVAL_WEIGHT: {
      void     *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight_1d, &d_v};
      const int grid_size     = num_elem;
      CeedCallBackend(CeedRunKernelDimCuda(ceed, data->Weight, grid_size, Q_1d, dim >= 2 ? Q_1d : 1, 1, weight_args));
    } break;
    // LCOV_EXCL_START
    // Evaluate the divergence to/from the quadrature points
    case CEED_EVAL_DIV:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
    // Evaluate the curl to/from the quadrature points
    case CEED_EVAL_CURL:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
    // Take no action, BasisApply should not have been called
    case CEED_EVAL_NONE:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
      // LCOV_EXCL_STOP
  }

  // Restore vectors
  if (eval_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  }
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - non-tensor
//------------------------------------------------------------------------------
int CeedBasisApplyNonTensor_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                 CeedVector v) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  Ceed_Cuda *ceed_Cuda;
  CeedCallBackend(CeedGetData(ceed, &ceed_Cuda));
  CeedBasisNonTensor_Cuda *data;
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedInt num_nodes, num_qpts;
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
  const CeedInt transpose       = t_mode == CEED_TRANSPOSE;
  int           elems_per_block = 1;
  int           grid            = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  if (eval_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  }
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (t_mode == CEED_TRANSPOSE) {
    CeedSize length;
    CeedCallBackend(CeedVectorGetLength(v, &length));
    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
  }

  // Apply basis operation
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      void *interp_args[] = {(void *)&num_elem, (void *)&transpose, &data->d_interp, &d_u, &d_v};
      if (transpose) {
        CeedCallBackend(CeedRunKernelDimCuda(ceed, data->Interp, grid, num_nodes, 1, elems_per_block, interp_args));
      } else {
        CeedCallBackend(CeedRunKernelDimCuda(ceed, data->Interp, grid, num_qpts, 1, elems_per_block, interp_args));
      }
    } break;
    case CEED_EVAL_GRAD: {
      void *grad_args[] = {(void *)&num_elem, (void *)&transpose, &data->d_grad, &d_u, &d_v};
      if (transpose) {
        CeedCallBackend(CeedRunKernelDimCuda(ceed, data->Grad, grid, num_nodes, 1, elems_per_block, grad_args));
      } else {
        CeedCallBackend(CeedRunKernelDimCuda(ceed, data->Grad, grid, num_qpts, 1, elems_per_block, grad_args));
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      void *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight, &d_v};
      CeedCallBackend(CeedRunKernelDimCuda(ceed, data->Weight, grid, num_qpts, 1, elems_per_block, weight_args));
    } break;
    // LCOV_EXCL_START
    // Evaluate the divergence to/from the quadrature points
    case CEED_EVAL_DIV:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
    // Evaluate the curl to/from the quadrature points
    case CEED_EVAL_CURL:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
    // Take no action, BasisApply should not have been called
    case CEED_EVAL_NONE:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
      // LCOV_EXCL_STOP
  }

  // Restore vectors
  if (eval_mode != CEED_EVAL_WEIGHT) {
    CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  }
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Cuda(CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  CeedBasis_Cuda *data;
  CeedCallBackend(CeedBasisGetData(basis, &data));

  CeedCallCuda(ceed, cuModuleUnload(data->module));

  CeedCallCuda(ceed, cudaFree(data->d_q_weight_1d));
  CeedCallCuda(ceed, cudaFree(data->d_interp_1d));
  CeedCallCuda(ceed, cudaFree(data->d_grad_1d));
  CeedCallBackend(CeedFree(&data));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Cuda(CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  CeedBasisNonTensor_Cuda *data;
  CeedCallBackend(CeedBasisGetData(basis, &data));

  CeedCallCuda(ceed, cuModuleUnload(data->module));

  CeedCallCuda(ceed, cudaFree(data->d_q_weight));
  CeedCallCuda(ceed, cudaFree(data->d_interp));
  CeedCallCuda(ceed, cudaFree(data->d_grad));
  CeedCallBackend(CeedFree(&data));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                 const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasis_Cuda *data;
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy data to GPU
  const CeedInt q_bytes = Q_1d * sizeof(CeedScalar);
  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight_1d, q_bytes));
  CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes, cudaMemcpyHostToDevice));

  const CeedInt interp_bytes = q_bytes * P_1d;
  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp_1d, interp_bytes));
  CeedCallCuda(ceed, cudaMemcpy(data->d_interp_1d, interp_1d, interp_bytes, cudaMemcpyHostToDevice));

  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad_1d, interp_bytes));
  CeedCallCuda(ceed, cudaMemcpy(data->d_grad_1d, grad_1d, interp_bytes, cudaMemcpyHostToDevice));

  // Complie basis kernels
  CeedInt num_comp;
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  char *basis_kernel_path, *basis_kernel_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-basis-tensor.h", &basis_kernel_path));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompileCuda(ceed, basis_kernel_source, &data->module, 7, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "BASIS_BUF_LEN",
                                  num_comp * CeedIntPow(Q_1d > P_1d ? Q_1d : P_1d, dim), "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp,
                                  "BASIS_NUM_NODES", CeedIntPow(P_1d, dim), "BASIS_NUM_QPTS", CeedIntPow(Q_1d, dim)));
  CeedCallBackend(CeedGetKernelCuda(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernelCuda(ceed, data->module, "Grad", &data->Grad));
  CeedCallBackend(CeedGetKernelCuda(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                           const CeedScalar *qref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasisNonTensor_Cuda *data;
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  const CeedInt q_bytes = num_qpts * sizeof(CeedScalar);
  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight, q_bytes));
  CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight, q_weight, q_bytes, cudaMemcpyHostToDevice));

  const CeedInt interp_bytes = q_bytes * num_nodes;
  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp, interp_bytes));
  CeedCallCuda(ceed, cudaMemcpy(data->d_interp, interp, interp_bytes, cudaMemcpyHostToDevice));

  const CeedInt grad_bytes = q_bytes * num_nodes * dim;
  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad, grad_bytes));
  CeedCallCuda(ceed, cudaMemcpy(data->d_grad, grad, grad_bytes, cudaMemcpyHostToDevice));

  // Compile basis kernels
  CeedInt num_comp;
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  char *basis_kernel_path, *basis_kernel_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-basis-nontensor.h", &basis_kernel_path));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallCuda(ceed, CeedCompileCuda(ceed, basis_kernel_source, &data->module, 4, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_DIM", dim,
                                     "BASIS_NUM_COMP", num_comp));
  CeedCallCuda(ceed, CeedGetKernelCuda(ceed, data->module, "Interp", &data->Interp));
  CeedCallCuda(ceed, CeedGetKernelCuda(ceed, data->module, "Grad", &data->Grad));
  CeedCallCuda(ceed, CeedGetKernelCuda(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Cuda));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
