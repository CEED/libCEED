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

#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
int CeedBasisApply_Hip(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  Ceed_Hip *ceed_Hip;
  CeedCallBackend(CeedGetData(ceed, &ceed_Hip));
  CeedBasis_Hip *data;
  CeedCallBackend(CeedBasisGetData(basis, &data));
  const CeedInt transpose      = t_mode == CEED_TRANSPOSE;
  const int     max_block_size = 64;

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
    CeedCallHip(ceed, hipMemset(d_v, 0, length * sizeof(CeedScalar)));
  }

  // Basis action
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      void   *interp_args[] = {(void *)&num_elem, (void *)&transpose, &data->d_interp_1d, &d_u, &d_v};
      CeedInt Q_1d, dim;
      CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
      CeedCallBackend(CeedBasisGetDimension(basis, &dim));
      CeedInt block_size = CeedIntMin(CeedIntPow(Q_1d, dim), max_block_size);

      CeedCallBackend(CeedRunKernelHip(ceed, data->Interp, num_elem, block_size, interp_args));
    } break;
    case CEED_EVAL_GRAD: {
      void   *grad_args[] = {(void *)&num_elem, (void *)&transpose, &data->d_interp_1d, &data->d_grad_1d, &d_u, &d_v};
      CeedInt block_size  = max_block_size;

      CeedCallBackend(CeedRunKernelHip(ceed, data->Grad, num_elem, block_size, grad_args));
    } break;
    case CEED_EVAL_WEIGHT: {
      void     *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight_1d, &d_v};
      const int block_size    = 64;
      int       grid_size     = num_elem / block_size;
      if (block_size * grid_size < num_elem) grid_size += 1;

      CeedCallBackend(CeedRunKernelHip(ceed, data->Weight, grid_size, block_size, weight_args));
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
int CeedBasisApplyNonTensor_Hip(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                CeedVector v) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  Ceed_Hip *ceed_Hip;
  CeedCallBackend(CeedGetData(ceed, &ceed_Hip));
  CeedBasisNonTensor_Hip *data;
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedInt num_nodes, num_qpts;
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
  const CeedInt transpose     = t_mode == CEED_TRANSPOSE;
  int           elemsPerBlock = 1;
  int           grid          = num_elem / elemsPerBlock + ((num_elem / elemsPerBlock * elemsPerBlock < num_elem) ? 1 : 0);

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
    CeedCallHip(ceed, hipMemset(d_v, 0, length * sizeof(CeedScalar)));
  }

  // Apply basis operation
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      void     *interp_args[] = {(void *)&num_elem, (void *)&transpose, &data->d_interp, &d_u, &d_v};
      const int block_size_x  = transpose ? num_nodes : num_qpts;
      CeedCallBackend(CeedRunKernelDimHip(ceed, data->Interp, grid, block_size_x, 1, elemsPerBlock, interp_args));
    } break;
    case CEED_EVAL_GRAD: {
      void     *grad_args[]  = {(void *)&num_elem, (void *)&transpose, &data->d_grad, &d_u, &d_v};
      const int block_size_x = transpose ? num_nodes : num_qpts;
      CeedCallBackend(CeedRunKernelDimHip(ceed, data->Grad, grid, block_size_x, 1, elemsPerBlock, grad_args));
    } break;
    case CEED_EVAL_WEIGHT: {
      void *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight, &d_v};
      CeedCallBackend(CeedRunKernelDimHip(ceed, data->Weight, grid, num_qpts, 1, elemsPerBlock, weight_args));
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
static int CeedBasisDestroy_Hip(CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  CeedBasis_Hip *data;
  CeedCallBackend(CeedBasisGetData(basis, &data));

  CeedCallHip(ceed, hipModuleUnload(data->module));

  CeedCallHip(ceed, hipFree(data->d_q_weight_1d));
  CeedCallHip(ceed, hipFree(data->d_interp_1d));
  CeedCallHip(ceed, hipFree(data->d_grad_1d));
  CeedCallBackend(CeedFree(&data));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Hip(CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  CeedBasisNonTensor_Hip *data;
  CeedCallBackend(CeedBasisGetData(basis, &data));

  CeedCallHip(ceed, hipModuleUnload(data->module));

  CeedCallHip(ceed, hipFree(data->d_q_weight));
  CeedCallHip(ceed, hipFree(data->d_interp));
  CeedCallHip(ceed, hipFree(data->d_grad));
  CeedCallBackend(CeedFree(&data));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Hip(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                const CeedScalar *qref1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasis_Hip *data;
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy data to GPU
  const CeedInt q_bytes = Q_1d * sizeof(CeedScalar);
  CeedCallHip(ceed, hipMalloc((void **)&data->d_q_weight_1d, q_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes, hipMemcpyHostToDevice));

  const CeedInt interp_bytes = q_bytes * P_1d;
  CeedCallHip(ceed, hipMalloc((void **)&data->d_interp_1d, interp_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_interp_1d, interp_1d, interp_bytes, hipMemcpyHostToDevice));

  CeedCallHip(ceed, hipMalloc((void **)&data->d_grad_1d, interp_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_grad_1d, grad_1d, interp_bytes, hipMemcpyHostToDevice));

  // Complie basis kernels
  CeedInt ncomp;
  CeedCallBackend(CeedBasisGetNumComponents(basis, &ncomp));
  char *basis_kernel_path, *basis_kernel_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-basis-tensor.h", &basis_kernel_path));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompileHip(ceed, basis_kernel_source, &data->module, 7, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "BASIS_BUF_LEN",
                                 ncomp * CeedIntPow(Q_1d > P_1d ? Q_1d : P_1d, dim), "BASIS_DIM", dim, "BASIS_NUM_COMP", ncomp, "BASIS_NUM_NODES",
                                 CeedIntPow(P_1d, dim), "BASIS_NUM_QPTS", CeedIntPow(Q_1d, dim)));
  CeedCallBackend(CeedGetKernelHip(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernelHip(ceed, data->module, "Grad", &data->Grad));
  CeedCallBackend(CeedGetKernelHip(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Hip(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                          const CeedScalar *qref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasisNonTensor_Hip *data;
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  const CeedInt q_bytes = num_qpts * sizeof(CeedScalar);
  CeedCallHip(ceed, hipMalloc((void **)&data->d_q_weight, q_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_q_weight, q_weight, q_bytes, hipMemcpyHostToDevice));

  const CeedInt interp_bytes = q_bytes * num_nodes;
  CeedCallHip(ceed, hipMalloc((void **)&data->d_interp, interp_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_interp, interp, interp_bytes, hipMemcpyHostToDevice));

  const CeedInt grad_bytes = q_bytes * num_nodes * dim;
  CeedCallHip(ceed, hipMalloc((void **)&data->d_grad, grad_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_grad, grad, grad_bytes, hipMemcpyHostToDevice));

  // Compile basis kernels
  CeedInt ncomp;
  CeedCallBackend(CeedBasisGetNumComponents(basis, &ncomp));
  char *basis_kernel_path, *basis_kernel_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-basis-nontensor.h", &basis_kernel_path));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompileHip(ceed, basis_kernel_source, &data->module, 4, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_DIM", dim,
                                 "BASIS_NUM_COMP", ncomp));
  CeedCallBackend(CeedGetKernelHip(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernelHip(ceed, data->module, "Grad", &data->Grad));
  CeedCallBackend(CeedGetKernelHip(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));
  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Hip));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
