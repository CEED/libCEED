// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <cuda_runtime.h>
#include <stdbool.h>

#include "../cuda/ceed-cuda-common.h"
#include "../cuda/ceed-cuda-compile.h"
#include "ceed-cutlass.h"

//------------------------------------------------------------------------------
// Copy an array to the GPU
//------------------------------------------------------------------------------
static int CeedBasisCopyToDevice_Cutlass(Ceed ceed, const CeedScalar *h_array, size_t length, CeedScalar **d_array) {
  if (h_array == NULL) return CEED_ERROR_SUCCESS;
  size_t size = length * sizeof(CeedScalar);

  CeedCallCuda(ceed, cudaMalloc((void **)d_array, size));
  CeedCallCuda(ceed, cudaMemcpy(*d_array, h_array, size, cudaMemcpyHostToDevice));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// GEMM tile for an M x K times K x N product: TM x TN threads (about 256), each holding (M / TM) x RN outputs
//   Narrow blocks (RN = 1 once TN >= 32) keep enough blocks in flight for small num_elem
//------------------------------------------------------------------------------
static void CeedBasisGetTile_Cutlass(CeedInt m, CeedInt *tm, CeedInt *tn, CeedInt *rn) {
  CeedInt rm = CeedIntMin(8, CeedIntMax(m / 2, 1));

  while (m % rm) rm--;
  *tm = m / rm;
  *tn = CeedIntMax(1, 256 / *tm);
  *rn = CeedIntMax(1, CeedIntMin(32 / rm, 32 / *tn));
}

//------------------------------------------------------------------------------
// Destroy basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Cutlass(CeedBasis basis) {
  Ceed                        ceed;
  CeedBasisNonTensor_Cutlass *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));

  CeedCallCuda(ceed, cuModuleUnload(data->module));
  CeedCallCuda(ceed, cudaFree(data->d_curl));
  CeedCallCuda(ceed, cudaFree(data->d_div));
  CeedCallCuda(ceed, cudaFree(data->d_grad));
  CeedCallCuda(ceed, cudaFree(data->d_interp));
  CeedCallCuda(ceed, cudaFree(data->d_q_weight));

  CeedCallBackend(CeedFree(&data));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply basis
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorCore_Cutlass(CeedBasis basis, bool apply_add, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode,
                                               CeedVector u, CeedVector v) {
  Ceed                        ceed;
  CeedBasisNonTensor_Cutlass *data;
  CeedInt                     num_qpts, num_comp, grid;
  const CeedScalar           *d_u;
  CeedScalar                 *d_v;
  const CeedInt               is_transpose = t_mode == CEED_TRANSPOSE, add = apply_add;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  grid = CeedDivUpInt(num_elem * num_comp, data->num_cols_per_block[is_transpose]);

  // Get read/write access to u, v
  if (u != CEED_VECTOR_NONE) {
    CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  } else {
    CeedCheck(e_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  }

  // Transpose kernels overwrite v unless add is set, so v is not cleared first
  if (apply_add) {
    CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  } else {
    CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));
  }

  // Apply basis operation
  switch (e_mode) {
    case CEED_EVAL_INTERP: {
      void *interp_args[] = {(void *)&num_elem, (void *)&add, &data->d_interp, &d_u, &d_v};

      CeedCallBackend(CeedRunKernel_Cuda(ceed, is_transpose ? data->InterpTranspose : data->Interp, grid, data->num_threads[is_transpose],
                                         interp_args));
    } break;
    case CEED_EVAL_GRAD:
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL: {
      CeedScalar *d_deriv      = e_mode == CEED_EVAL_GRAD ? data->d_grad : (e_mode == CEED_EVAL_DIV ? data->d_div : data->d_curl);
      void       *deriv_args[] = {(void *)&num_elem, (void *)&add, &d_deriv, &d_u, &d_v};

      CeedCallBackend(CeedRunKernel_Cuda(ceed, is_transpose ? data->DerivTranspose : data->Deriv, grid, data->num_threads[is_transpose], deriv_args));
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedCheck(data->d_q_weight, ceed, CEED_ERROR_BACKEND, "%s not supported; q_weights not set", CeedEvalModes[e_mode]);
      CeedCheck(!is_transpose, ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
      void *weight_args[] = {(void *)&num_elem, (void *)&data->d_q_weight, &d_v};

      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, num_elem, num_qpts, 1, 1, weight_args));
    } break;
    case CEED_EVAL_NONE: {
    } break;
  }

  // Restore vectors, cover CEED_EVAL_NONE
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  if (e_mode == CEED_EVAL_NONE) CeedCallBackend(CeedVectorSetArray(v, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)d_u));
  if (e_mode != CEED_EVAL_WEIGHT) CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply basis (overwrite v)
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensor_Cutlass(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector u,
                                           CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Cutlass(basis, false, num_elem, t_mode, e_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply basis (add into v)
//------------------------------------------------------------------------------
static int CeedBasisApplyAddNonTensor_Cutlass(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector u,
                                              CeedVector v) {
  CeedCallBackend(CeedBasisApplyNonTensorCore_Cutlass(basis, true, num_elem, t_mode, e_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor basis (shared by H1, H(div), H(curl))
//------------------------------------------------------------------------------
static int CeedBasisCreateNonTensor_Cutlass(CeedInt num_nodes, CeedInt num_qpts, CeedEvalMode deriv_mode, const CeedScalar *interp,
                                            const CeedScalar *deriv, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                        ceed;
  CeedInt                     num_comp, q_comp_interp, q_comp_deriv, tile[2][3];
  CeedBasisNonTensor_Cutlass *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  // Tile [0] is NOTRANSPOSE (M = Q) and [1] is TRANSPOSE (M = P), checked before anything is allocated
  CeedBasisGetTile_Cutlass(num_qpts, &tile[0][0], &tile[0][1], &tile[0][2]);
  CeedBasisGetTile_Cutlass(num_nodes, &tile[1][0], &tile[1][1], &tile[1][2]);
  for (CeedInt t = 0; t < 2; t++) {
    CeedCheck(tile[t][0] * tile[t][1] <= 1024, ceed, CEED_ERROR_BACKEND, "Basis size not supported by the CUTLASS backend");
  }
  CeedCallBackend(CeedCalloc(1, &data));
  for (CeedInt t = 0; t < 2; t++) {
    data->num_threads[t]        = tile[t][0] * tile[t][1];
    data->num_cols_per_block[t] = tile[t][1] * tile[t][2];
  }

  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, deriv_mode, &q_comp_deriv));

  CeedCallBackend(CeedBasisCopyToDevice_Cutlass(ceed, q_weight, num_qpts, &data->d_q_weight));

  CeedCallBackend(CeedBasisCopyToDevice_Cutlass(ceed, interp, num_qpts * num_nodes * q_comp_interp, &data->d_interp));

  if (deriv_mode == CEED_EVAL_GRAD) {
    CeedCallBackend(CeedBasisCopyToDevice_Cutlass(ceed, deriv, num_qpts * num_nodes * q_comp_deriv, &data->d_grad));
  } else if (deriv_mode == CEED_EVAL_DIV) {
    CeedCallBackend(CeedBasisCopyToDevice_Cutlass(ceed, deriv, num_qpts * num_nodes * q_comp_deriv, &data->d_div));
  } else if (deriv_mode == CEED_EVAL_CURL) {
    CeedCallBackend(CeedBasisCopyToDevice_Cutlass(ceed, deriv, num_qpts * num_nodes * q_comp_deriv, &data->d_curl));
  }

  // Compile basis kernels
  const char basis_kernel_source[] = "// Nontensor basis source\n#include <ceed/jit-source/cuda/cuda-cutlass-basis-nontensor.h>\n";

  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, "basis_cutlass_nontensor", &data->module, 11, "BASIS_Q", num_qpts, "BASIS_P", num_nodes,
                                   "BASIS_Q_COMP_INTERP", q_comp_interp, "BASIS_Q_COMP_DERIV", q_comp_deriv, "BASIS_NUM_COMP", num_comp,
                                   "BASIS_NOTRANSPOSE_TM", tile[0][0], "BASIS_NOTRANSPOSE_TN", tile[0][1], "BASIS_NOTRANSPOSE_RN", tile[0][2],
                                   "BASIS_TRANSPOSE_TM", tile[1][0], "BASIS_TRANSPOSE_TN", tile[1][1], "BASIS_TRANSPOSE_RN", tile[1][2]));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Deriv", &data->Deriv));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
  CeedCallBackend(CeedBasisSetData(basis, data));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cutlass));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cutlass));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Cutlass));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create H1 basis
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Cutlass(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                              const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreateNonTensor_Cutlass(num_nodes, num_qpts, CEED_EVAL_GRAD, interp, grad, q_weight, basis));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create H(div) basis
//------------------------------------------------------------------------------
int CeedBasisCreateHdiv_Cutlass(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreateNonTensor_Cutlass(num_nodes, num_qpts, CEED_EVAL_DIV, interp, div, q_weight, basis));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create H(curl) basis
//------------------------------------------------------------------------------
int CeedBasisCreateHcurl_Cutlass(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                 const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreateNonTensor_Cutlass(num_nodes, num_qpts, CEED_EVAL_CURL, interp, curl, q_weight, basis));
  return CEED_ERROR_SUCCESS;
}
