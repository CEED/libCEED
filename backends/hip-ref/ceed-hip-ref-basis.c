// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <hip/hip_runtime.h>

#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
int CeedBasisApply_Hip(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  Ceed              ceed;
  CeedInt           Q_1d, dim;
  const CeedInt     is_transpose   = t_mode == CEED_TRANSPOSE;
  const int         max_block_size = 64;
  const CeedScalar *d_u;
  CeedScalar       *d_v;
  CeedBasis_Hip    *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));

  // Get read/write access to u, v
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(eval_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (is_transpose) {
    CeedSize length;

    CeedCallBackend(CeedVectorGetLength(v, &length));
    CeedCallHip(ceed, hipMemset(d_v, 0, length * sizeof(CeedScalar)));
  }
  CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));

  // Basis action
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      void         *interp_args[] = {(void *)&num_elem, (void *)&is_transpose, &data->d_interp_1d, &d_u, &d_v};
      const CeedInt block_size    = CeedIntMin(CeedIntPow(Q_1d, dim), max_block_size);

      CeedCallBackend(CeedRunKernel_Hip(ceed, data->Interp, num_elem, block_size, interp_args));
    } break;
    case CEED_EVAL_GRAD: {
      void         *grad_args[] = {(void *)&num_elem, (void *)&is_transpose, &data->d_interp_1d, &data->d_grad_1d, &d_u, &d_v};
      const CeedInt block_size  = max_block_size;

      CeedCallBackend(CeedRunKernel_Hip(ceed, data->Grad, num_elem, block_size, grad_args));
    } break;
    case CEED_EVAL_WEIGHT: {
      void     *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight_1d, &d_v};
      const int block_size_x  = Q_1d;
      const int block_size_y  = dim >= 2 ? Q_1d : 1;

      CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->Weight, num_elem, block_size_x, block_size_y, 1, weight_args));
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
// Basis apply - non-tensor
//------------------------------------------------------------------------------
int CeedBasisApplyNonTensor_Hip(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
                                CeedVector v) {
  Ceed                    ceed;
  CeedInt                 num_nodes, num_qpts;
  const CeedInt           is_transpose    = t_mode == CEED_TRANSPOSE;
  const int               elems_per_block = 1;
  const int               grid            = CeedDivUpInt(num_elem, elems_per_block);
  const CeedScalar       *d_u;
  CeedScalar             *d_v;
  CeedBasisNonTensor_Hip *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));

  // Get read/write access to u, v
  if (u != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  else CeedCheck(eval_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));

  // Clear v for transpose operation
  if (is_transpose) {
    CeedSize length;

    CeedCallBackend(CeedVectorGetLength(v, &length));
    CeedCallHip(ceed, hipMemset(d_v, 0, length * sizeof(CeedScalar)));
  }

  // Apply basis operation
  switch (eval_mode) {
    case CEED_EVAL_INTERP: {
      void     *interp_args[] = {(void *)&num_elem, &data->d_interp, &d_u, &d_v};
      const int block_size_x  = is_transpose ? num_nodes : num_qpts;

      if (is_transpose) {
        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->InterpTranspose, grid, block_size_x, 1, elems_per_block, interp_args));
      } else {
        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->Interp, grid, block_size_x, 1, elems_per_block, interp_args));
      }
    } break;
    case CEED_EVAL_GRAD: {
      void     *grad_args[]  = {(void *)&num_elem, &data->d_grad, &d_u, &d_v};
      const int block_size_x = is_transpose ? num_nodes : num_qpts;

      if (is_transpose) {
        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, grad_args));
      } else {
        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, grad_args));
      }
    } break;
    case CEED_EVAL_DIV: {
      void     *div_args[]   = {(void *)&num_elem, &data->d_div, &d_u, &d_v};
      const int block_size_x = is_transpose ? num_nodes : num_qpts;

      if (is_transpose) {
        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, div_args));
      } else {
        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, div_args));
      }
    } break;
    case CEED_EVAL_CURL: {
      void     *curl_args[]  = {(void *)&num_elem, &data->d_curl, &d_u, &d_v};
      const int block_size_x = is_transpose ? num_nodes : num_qpts;

      if (is_transpose) {
        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, curl_args));
      } else {
        CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, curl_args));
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      void *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight, &d_v};

      CeedCallBackend(CeedRunKernelDim_Hip(ceed, data->Weight, grid, num_qpts, 1, elems_per_block, weight_args));
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

//------------------------------------------------------------------------------
// Destroy tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Hip(CeedBasis basis) {
  Ceed           ceed;
  CeedBasis_Hip *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
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
  Ceed                    ceed;
  CeedBasisNonTensor_Hip *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallHip(ceed, hipModuleUnload(data->module));
  CeedCallHip(ceed, hipFree(data->d_q_weight));
  CeedCallHip(ceed, hipFree(data->d_interp));
  CeedCallHip(ceed, hipFree(data->d_grad));
  CeedCallHip(ceed, hipFree(data->d_div));
  CeedCallHip(ceed, hipFree(data->d_curl));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Hip(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed           ceed;
  char          *basis_kernel_source;
  const char    *basis_kernel_path;
  CeedInt        num_comp;
  const CeedInt  q_bytes      = Q_1d * sizeof(CeedScalar);
  const CeedInt  interp_bytes = q_bytes * P_1d;
  CeedBasis_Hip *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy data to GPU
  CeedCallHip(ceed, hipMalloc((void **)&data->d_q_weight_1d, q_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes, hipMemcpyHostToDevice));
  CeedCallHip(ceed, hipMalloc((void **)&data->d_interp_1d, interp_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_interp_1d, interp_1d, interp_bytes, hipMemcpyHostToDevice));
  CeedCallHip(ceed, hipMalloc((void **)&data->d_grad_1d, interp_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_grad_1d, grad_1d, interp_bytes, hipMemcpyHostToDevice));

  // Compile basis kernels
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-basis-tensor.h", &basis_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Hip(ceed, basis_kernel_source, &data->module, 7, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "BASIS_BUF_LEN",
                                  num_comp * CeedIntPow(Q_1d > P_1d ? Q_1d : P_1d, dim), "BASIS_DIM", dim, "BASIS_NUM_COMP", num_comp,
                                  "BASIS_NUM_NODES", CeedIntPow(P_1d, dim), "BASIS_NUM_QPTS", CeedIntPow(Q_1d, dim)));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Grad", &data->Grad));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H^1
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Hip(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                          const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                    ceed;
  char                   *basis_kernel_source;
  const char             *basis_kernel_path;
  CeedInt                 num_comp, q_comp_interp, q_comp_grad;
  const CeedInt           q_bytes = num_qpts * sizeof(CeedScalar);
  CeedBasisNonTensor_Hip *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_GRAD, &q_comp_grad));
  CeedCallHip(ceed, hipMalloc((void **)&data->d_q_weight, q_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_q_weight, q_weight, q_bytes, hipMemcpyHostToDevice));
  if (interp) {
    const CeedInt interp_bytes = q_bytes * num_nodes * q_comp_interp;

    CeedCallHip(ceed, hipMalloc((void **)&data->d_interp, interp_bytes));
    CeedCallHip(ceed, hipMemcpy(data->d_interp, interp, interp_bytes, hipMemcpyHostToDevice));
  }
  if (grad) {
    const CeedInt grad_bytes = q_bytes * num_nodes * q_comp_grad;

    CeedCallHip(ceed, hipMalloc((void **)&data->d_grad, grad_bytes));
    CeedCallHip(ceed, hipMemcpy(data->d_grad, grad, grad_bytes, hipMemcpyHostToDevice));
  }

  // Compile basis kernels
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-basis-nontensor.h", &basis_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Hip(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
                                  q_comp_interp, "BASIS_Q_COMP_DERIV", q_comp_grad, "BASIS_NUM_COMP", num_comp));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Deriv", &data->Deriv));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H(div)
//------------------------------------------------------------------------------
int CeedBasisCreateHdiv_Hip(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *div,
                            const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                    ceed;
  char                   *basis_kernel_source;
  const char             *basis_kernel_path;
  CeedInt                 num_comp, q_comp_interp, q_comp_div;
  const CeedInt           q_bytes = num_qpts * sizeof(CeedScalar);
  CeedBasisNonTensor_Hip *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_DIV, &q_comp_div));
  CeedCallHip(ceed, hipMalloc((void **)&data->d_q_weight, q_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_q_weight, q_weight, q_bytes, hipMemcpyHostToDevice));
  if (interp) {
    const CeedInt interp_bytes = q_bytes * num_nodes * q_comp_interp;

    CeedCallHip(ceed, hipMalloc((void **)&data->d_interp, interp_bytes));
    CeedCallHip(ceed, hipMemcpy(data->d_interp, interp, interp_bytes, hipMemcpyHostToDevice));
  }
  if (div) {
    const CeedInt div_bytes = q_bytes * num_nodes * q_comp_div;

    CeedCallHip(ceed, hipMalloc((void **)&data->d_div, div_bytes));
    CeedCallHip(ceed, hipMemcpy(data->d_div, div, div_bytes, hipMemcpyHostToDevice));
  }

  // Compile basis kernels
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-basis-nontensor.h", &basis_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Hip(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
                                  q_comp_interp, "BASIS_Q_COMP_DERIV", q_comp_div, "BASIS_NUM_COMP", num_comp));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Deriv", &data->Deriv));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H(curl)
//------------------------------------------------------------------------------
int CeedBasisCreateHcurl_Hip(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                             const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                    ceed;
  char                   *basis_kernel_source;
  const char             *basis_kernel_path;
  CeedInt                 num_comp, q_comp_interp, q_comp_curl;
  const CeedInt           q_bytes = num_qpts * sizeof(CeedScalar);
  CeedBasisNonTensor_Hip *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Copy basis data to GPU
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_CURL, &q_comp_curl));
  CeedCallHip(ceed, hipMalloc((void **)&data->d_q_weight, q_bytes));
  CeedCallHip(ceed, hipMemcpy(data->d_q_weight, q_weight, q_bytes, hipMemcpyHostToDevice));
  if (interp) {
    const CeedInt interp_bytes = q_bytes * num_nodes * q_comp_interp;

    CeedCallHip(ceed, hipMalloc((void **)&data->d_interp, interp_bytes));
    CeedCallHip(ceed, hipMemcpy(data->d_interp, interp, interp_bytes, hipMemcpyHostToDevice));
  }
  if (curl) {
    const CeedInt curl_bytes = q_bytes * num_nodes * q_comp_curl;

    CeedCallHip(ceed, hipMalloc((void **)&data->d_curl, curl_bytes));
    CeedCallHip(ceed, hipMemcpy(data->d_curl, curl, curl_bytes, hipMemcpyHostToDevice));
  }

  // Compile basis kernels
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-basis-nontensor.h", &basis_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, basis_kernel_path, &basis_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Basis Kernel Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Hip(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
                                  q_comp_interp, "BASIS_Q_COMP_DERIV", q_comp_curl, "BASIS_NUM_COMP", num_comp));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Deriv", &data->Deriv));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedFree(&basis_kernel_path));
  CeedCallBackend(CeedFree(&basis_kernel_source));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
