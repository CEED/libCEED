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
// Destroy basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Cutlass(CeedBasis basis) {
  Ceed                        ceed;
  CeedBasisNonTensor_Cutlass *notebook;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &notebook));

  CeedCallCuda(ceed, cudaFree(notebook->d_curl));
  CeedCallCuda(ceed, cudaFree(notebook->d_div));
  CeedCallCuda(ceed, cudaFree(notebook->d_grad));
  CeedCallCuda(ceed, cudaFree(notebook->d_interp));
  CeedCallCuda(ceed, cudaFree(notebook->d_q_weight));

  CeedCallBackend(CeedFree(&notebook));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply one table (INTERP, GRAD, DIV or CURL) with a GEMM per piece
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorGemm_Cutlass(Ceed ceed, CeedBasis basis, CeedEvalMode e_mode, bool apply_add, bool is_transpose, CeedInt num_nodes,
                                               CeedInt num_qpts, CeedInt N, const CeedScalar *mat, const CeedScalar *d_u, CeedScalar *d_v) {
  CeedInt q_comp;

  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, e_mode, &q_comp));
  for (int d = 0; d < q_comp; d++) {
    int ierr;

    if (is_transpose) {
      const CeedScalar beta = (apply_add || (d > 0)) ? 1.0 : 0.0;

      ierr = CeedCutlassGemm_Cuda(false, num_nodes, N, num_qpts, 1.0, mat + d * num_nodes * num_qpts, num_nodes, d_u + d * N * num_qpts, num_qpts,
                                  beta, d_v, num_nodes);
    } else {
      ierr = CeedCutlassGemm_Cuda(true, num_qpts, N, num_nodes, 1.0, mat + d * num_nodes * num_qpts, num_nodes, d_u, num_nodes, 0.0,
                                  d_v + d * N * num_qpts, num_qpts);
    }
    CeedCheck(!ierr, ceed, CEED_ERROR_BACKEND, "CUTLASS GEMM failed");
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply basis
//------------------------------------------------------------------------------
static int CeedBasisApplyNonTensorCore_Cutlass(CeedBasis basis, bool apply_add, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode,
                                               CeedVector u, CeedVector v) {
  Ceed                        ceed;
  CeedBasisNonTensor_Cutlass *data;
  CeedInt                     num_nodes, num_qpts, num_comp, N;
  const CeedScalar           *d_u;
  CeedScalar                 *d_v;
  const CeedInt               is_transpose = t_mode == CEED_TRANSPOSE;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedBasisGetData(basis, &data));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  N = num_elem * num_comp;

  // Get read/write access to u, v
  if (u != CEED_VECTOR_NONE) {
    CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  } else {
    CeedCheck(e_mode == CEED_EVAL_WEIGHT, ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  }

  if (apply_add) {
    CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  } else {
    CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));
  }

  // Apply basis operation
  switch (e_mode) {
    case CEED_EVAL_INTERP:
      CeedCallBackend(CeedBasisApplyNonTensorGemm_Cutlass(ceed, basis, CEED_EVAL_INTERP, apply_add, is_transpose, num_nodes, num_qpts, N,
                                                          data->d_interp, d_u, d_v));
      break;
    case CEED_EVAL_GRAD: {
      CeedCallBackend(CeedBasisApplyNonTensorGemm_Cutlass(ceed, basis, CEED_EVAL_GRAD, apply_add, is_transpose, num_nodes, num_qpts, N, data->d_grad,
                                                          d_u, d_v));

    } break;
    case CEED_EVAL_DIV: {
      CeedCallBackend(CeedBasisApplyNonTensorGemm_Cutlass(ceed, basis, CEED_EVAL_DIV, apply_add, is_transpose, num_nodes, num_qpts, N, data->d_div,
                                                          d_u, d_v));
    } break;
    case CEED_EVAL_CURL: {
      CeedCallBackend(CeedBasisApplyNonTensorGemm_Cutlass(ceed, basis, CEED_EVAL_CURL, apply_add, is_transpose, num_nodes, num_qpts, N, data->d_curl,
                                                          d_u, d_v));
    } break;
    case CEED_EVAL_WEIGHT: {
      CeedCheck(data->d_q_weight, ceed, CEED_ERROR_BACKEND, "%s not supported; q_weights not set", CeedEvalModes[e_mode]);
      CeedCheck(!is_transpose, ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
      CeedCutlassWeight_Cuda(num_elem, num_qpts, data->d_q_weight, d_v);

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
  return CeedBasisApplyNonTensorCore_Cutlass(basis, false, num_elem, t_mode, e_mode, u, v);
}

//------------------------------------------------------------------------------
// Apply basis (add into v)
//------------------------------------------------------------------------------
static int CeedBasisApplyAddNonTensor_Cutlass(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode e_mode, CeedVector u,
                                              CeedVector v) {
  return CeedBasisApplyNonTensorCore_Cutlass(basis, true, num_elem, t_mode, e_mode, u, v);
}

//------------------------------------------------------------------------------
// Create non-tensor basis (shared by H1, H(div), H(curl))
//------------------------------------------------------------------------------
static int CeedBasisCreateNonTensor_Cutlass(CeedInt num_nodes, CeedInt num_qpts, CeedEvalMode deriv_mode, const CeedScalar *interp,
                                            const CeedScalar *deriv, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed                        ceed;
  CeedInt                     q_comp_interp, q_comp_deriv;
  CeedBasisNonTensor_Cutlass *data;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));

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
  return CeedBasisCreateNonTensor_Cutlass(num_nodes, num_qpts, CEED_EVAL_GRAD, interp, grad, q_weight, basis);
}

//------------------------------------------------------------------------------
// Create H(div) basis
//------------------------------------------------------------------------------
int CeedBasisCreateHdiv_Cutlass(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  return CeedBasisCreateNonTensor_Cutlass(num_nodes, num_qpts, CEED_EVAL_DIV, interp, div, q_weight, basis);
}

//------------------------------------------------------------------------------
// Create H(curl) basis
//------------------------------------------------------------------------------
int CeedBasisCreateHcurl_Cutlass(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                 const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  return CeedBasisCreateNonTensor_Cutlass(num_nodes, num_qpts, CEED_EVAL_CURL, interp, curl, q_weight, basis);
}
