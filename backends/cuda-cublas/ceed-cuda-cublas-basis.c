// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
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
#include "ceed-cuda-cublas.h"

static int CeedTensorContractStrided_cublas(CeedBasis basis, CeedInt M, CeedInt N, CeedInt K, CeedInt D, const CeedScalar *restrict t,
                                            CeedTransposeMode t_mode, const bool add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  // Note: Weird ordering/transpose modes because cuBLAS requires column-major ordering
  // These are identical to those used for magma
  CeedBasis_Cuda_cublas  *data;
  CeedScalar              one = 1.0, zero = 0.0;
  const cublasOperation_t transpose_a   = t_mode == CEED_NOTRANSPOSE ? CUBLAS_OP_T : CUBLAS_OP_N;
  const CeedInt           leading_dim_t = t_mode == CEED_NOTRANSPOSE ? K : M;

  CeedCallBackend(CeedBasisGetData(basis, &data));
  for (CeedInt d = 0; d < D; d++) {
    CeedScalar       *d_v  = t_mode == CEED_NOTRANSPOSE ? v + d * N * M : v;
    const CeedScalar *d_u  = t_mode == CEED_NOTRANSPOSE ? u : u + d * N * K;
    const CeedScalar *d_t  = t + d * K * M;
    const CeedScalar *beta = (t_mode == CEED_TRANSPOSE && d > 0) || add ? &one : &zero;

#ifdef CEED_SCALAR_IS_FP32
    CeedCallCublas(CeedBasisReturnCeed(basis),
                   cublasSgemm(data->handle, transpose_a, CUBLAS_OP_N, M, N, K, &one, d_t, leading_dim_t, d_u, K, beta, d_v, M));
#else
    CeedCallCublas(CeedBasisReturnCeed(basis),
                   cublasDgemm(data->handle, transpose_a, CUBLAS_OP_N, M, N, K, &one, d_t, leading_dim_t, d_u, K, beta, d_v, M));
#endif
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorCore_Cuda_cublas(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode,
                                                   CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  Ceed                   ceed;
  Ceed_Cuda             *ceed_Cuda;
  CeedInt                dim, num_comp, q_comp, num_nodes, num_qpts;
  const CeedScalar      *d_u;
  CeedScalar            *d_v;
  CeedBasis_Cuda_cublas *data;
  bool                   add = apply_add && (t_mode != CEED_NOTRANSPOSE);

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_Cuda));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, eval_mode, &q_comp));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));

  // Get read/write access to u, v
  if (u != CEED_VECTOR_NONE) {
    CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  } else {
    CeedCheck(eval_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  }
  if (apply_add) {
    CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  } else {
    CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));
  }

  // Non-tensor basis
  const CeedInt P = num_nodes, Q = num_qpts;
  const CeedInt M = t_mode == CEED_NOTRANSPOSE ? Q : P;
  const CeedInt K = t_mode == CEED_NOTRANSPOSE ? P : Q;
  const CeedInt N = num_elem * num_comp;

  // Apply basis operation
  switch (eval_mode) {
    // Interpolate to/from quadrature points
    case CEED_EVAL_INTERP: {
      CeedCheck(data->d_interp, ceed, CEED_ERROR_BACKEND, "%s not supported; interp not set", CeedEvalModes[eval_mode]);
      CeedCallBackend(CeedTensorContractStrided_cublas(basis, M, N, K, q_comp, data->d_interp, t_mode, add, d_u, d_v));
    } break;
    // Evaluate the gradient to/from quadrature points
    case CEED_EVAL_GRAD: {
      CeedCheck(data->d_grad, ceed, CEED_ERROR_BACKEND, "%s not supported; grad not set", CeedEvalModes[eval_mode]);
      CeedCallBackend(CeedTensorContractStrided_cublas(basis, M, N, K, q_comp, data->d_grad, t_mode, add, d_u, d_v));
    } break;
    // Evaluate the divergence to/from the quadrature points
    case CEED_EVAL_DIV: {
      CeedCheck(data->d_div, ceed, CEED_ERROR_BACKEND, "%s not supported; div not set", CeedEvalModes[eval_mode]);
      CeedCallBackend(CeedTensorContractStrided_cublas(basis, M, N, K, q_comp, data->d_div, t_mode, add, d_u, d_v));
    } break;
    // Evaluate the curl to/from the quadrature points
    case CEED_EVAL_CURL: {
      CeedCheck(data->d_curl, ceed, CEED_ERROR_BACKEND, "%s not supported; curl not set", CeedEvalModes[eval_mode]);
      CeedCallBackend(CeedTensorContractStrided_cublas(basis, M, N, K, q_comp, data->d_curl, t_mode, add, d_u, d_v));
    } break;
    // Retrieve interpolation weights
    case CEED_EVAL_WEIGHT: {
      const int elems_per_block = 1;
      const int grid            = CeedDivUpInt(num_elem, elems_per_block);

      CeedCheck(data->d_q_weight, ceed, CEED_ERROR_BACKEND, "%s not supported; q_weights not set", CeedEvalModes[eval_mode]);
      CeedCheck(t_mode == CEED_NOTRANSPOSE, ceed, CEED_ERROR_BACKEND, "Weights not supported for transpose evaluation");
      void *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight, &d_v};

      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid, num_qpts, 1, elems_per_block, weight_args));
    } break;
    // LCOV_EXCL_START
    case CEED_EVAL_NONE:
      return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
      // LCOV_EXCL_STOP
  }
  cudaStream_t stream;

  CeedCallCublas(ceed, cublasGetStream(data->handle, &stream));
  CeedCallCuda(ceed, cudaStreamSynchronize(stream));

  // Restore vectors, cover CEED_EVAL_NONE
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  if (eval_mode == CEED_EVAL_NONE) CeedCallBackend(CeedVectorSetArray(v, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)d_u));
  if (eval_mode != CEED_EVAL_WEIGHT) CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyNonTensor_Cuda_cublas(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                                               CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Cuda_cublas(basis, false, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAddNonTensor_Cuda_cublas(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                                                  CeedVector u, CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Cuda_cublas(basis, true, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Cuda_cublas(CeedBasis basis) {
  Ceed                   ceed;
  CeedBasis_Cuda_cublas *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallCuda(ceed, cuModuleUnload(data->module));
  if (data->d_q_weight) CeedCallCuda(ceed, cudaFree(data->d_q_weight));
  if (data->d_interp) CeedCallCuda(ceed, cudaFree(data->d_interp));
  if (data->d_grad) CeedCallCuda(ceed, cudaFree(data->d_grad));
  if (data->d_curl) CeedCallCuda(ceed, cudaFree(data->d_curl));
  if (data->d_div) CeedCallCuda(ceed, cudaFree(data->d_div));
  CeedCallCublas(ceed, cublasDestroy(data->handle));
  CeedCallBackend(CeedFree(&data));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisCreateH1_Cuda_cublas_core(bool is_deterministic, CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts,
                                              const CeedScalar *interp, const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight,
                                              CeedBasis basis) {
  Ceed                   ceed;
  CeedInt                num_comp, q_comp_interp, q_comp_grad;
  const CeedInt          q_bytes = num_qpts * sizeof(CeedScalar);
  CeedBasis_Cuda_cublas *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Create cublas handle
  CeedCallCublas(ceed, cublasCreate(&data->handle));
  CeedCallCublas(ceed, cublasSetPointerMode(data->handle, CUBLAS_POINTER_MODE_HOST));
  CeedCallCublas(ceed, cublasSetAtomicsMode(data->handle, is_deterministic ? CUBLAS_ATOMICS_NOT_ALLOWED : CUBLAS_ATOMICS_ALLOWED));

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
  const char basis_kernel_source[] = "// Nontensor basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-nontensor.h>\n";

  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, "basis_h1_nontensor", &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes,
                                   "BASIS_Q_COMP_INTERP", q_comp_interp, "BASIS_Q_COMP_DERIV", q_comp_grad, "BASIS_NUM_COMP", num_comp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda_cublas));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda_cublas));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda_cublas));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisCreateH1_Cuda_cublas(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                  const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreateH1_Cuda_cublas_core(false, topo, dim, num_nodes, num_qpts, interp, grad, q_ref, q_weight, basis));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisCreateH1_Cuda_cublas_det(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                      const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreateH1_Cuda_cublas_core(true, topo, dim, num_nodes, num_qpts, interp, grad, q_ref, q_weight, basis));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H(div)
//------------------------------------------------------------------------------
static int CeedBasisCreateHdiv_Cuda_cublas_core(bool is_deterministic, CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts,
                                                const CeedScalar *interp, const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight,
                                                CeedBasis basis) {
  Ceed                   ceed;
  CeedInt                num_comp, q_comp_interp, q_comp_div;
  const CeedInt          q_bytes = num_qpts * sizeof(CeedScalar);
  CeedBasis_Cuda_cublas *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Create cublas handle
  CeedCallCublas(ceed, cublasCreate(&data->handle));
  CeedCallCublas(ceed, cublasSetPointerMode(data->handle, CUBLAS_POINTER_MODE_HOST));
  CeedCallCublas(ceed, cublasSetAtomicsMode(data->handle, is_deterministic ? CUBLAS_ATOMICS_NOT_ALLOWED : CUBLAS_ATOMICS_ALLOWED));

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
  const char basis_kernel_source[] = "// Nontensor basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-nontensor.h>\n";

  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, "basis_h_div", &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes,
                                   "BASIS_Q_COMP_INTERP", q_comp_interp, "BASIS_Q_COMP_DERIV", q_comp_div, "BASIS_NUM_COMP", num_comp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda_cublas));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda_cublas));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda_cublas));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisCreateHdiv_Cuda_cublas(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                    const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreateHdiv_Cuda_cublas_core(false, topo, dim, num_nodes, num_qpts, interp, div, q_ref, q_weight, basis));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisCreateHdiv_Cuda_cublas_det(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                        const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreateHdiv_Cuda_cublas_core(true, topo, dim, num_nodes, num_qpts, interp, div, q_ref, q_weight, basis));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor H(curl)
//------------------------------------------------------------------------------
static int CeedBasisCreateHcurl_Cuda_cublas_core(bool is_deterministic, CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts,
                                                 const CeedScalar *interp, const CeedScalar *curl, const CeedScalar *q_ref,
                                                 const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                   ceed;
  CeedInt                num_comp, q_comp_interp, q_comp_curl;
  const CeedInt          q_bytes = num_qpts * sizeof(CeedScalar);
  CeedBasis_Cuda_cublas *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

  // Create cublas handle
  CeedCallCublas(ceed, cublasCreate(&data->handle));
  CeedCallCublas(ceed, cublasSetPointerMode(data->handle, CUBLAS_POINTER_MODE_HOST));
  CeedCallCublas(ceed, cublasSetAtomicsMode(data->handle, is_deterministic ? CUBLAS_ATOMICS_NOT_ALLOWED : CUBLAS_ATOMICS_ALLOWED));

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
  const char basis_kernel_source[] = "// Nontensor basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-nontensor.h>\n";

  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, "basis_h_curl", &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes,
                                   "BASIS_Q_COMP_INTERP", q_comp_interp, "BASIS_Q_COMP_DERIV", q_comp_curl, "BASIS_NUM_COMP", num_comp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));

  CeedCallBackend(CeedBasisSetData(basis, data));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda_cublas));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda_cublas));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda_cublas));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisCreateHcurl_Cuda_cublas(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                     const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreateHcurl_Cuda_cublas_core(false, topo, dim, num_nodes, num_qpts, interp, curl, q_ref, q_weight, basis));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisCreateHcurl_Cuda_cublas_det(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                         const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreateHcurl_Cuda_cublas_core(true, topo, dim, num_nodes, num_qpts, interp, curl, q_ref, q_weight, basis));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
