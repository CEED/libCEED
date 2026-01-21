// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Basis Apply H1
//------------------------------------------------------------------------------
static int CeedBasisApplyCore_Ref(CeedBasis basis, bool apply_add, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector U,
                                  CeedVector V) {
  bool               is_tensor_basis, add = apply_add || (t_mode == CEED_TRANSPOSE);
  CeedInt            dim, num_comp, q_comp, num_nodes, num_qpts;
  const CeedScalar  *u;
  CeedScalar        *v;
  CeedTensorContract contract;
  CeedBasis_Ref     *impl;

  CeedCallBackend(CeedBasisGetData(basis, &impl));
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, eval_mode, &q_comp));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCallBackend(CeedBasisGetTensorContract(basis, &contract));
  if (U != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u));
  else CeedCheck(eval_mode == CEED_EVAL_WEIGHT, CeedBasisReturnCeed(basis), CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  // Clear v if operating in transpose
  if (apply_add) CeedCallBackend(CeedVectorGetArray(V, CEED_MEM_HOST, &v));
  else CeedCallBackend(CeedVectorGetArrayWrite(V, CEED_MEM_HOST, &v));

  if (t_mode == CEED_TRANSPOSE && !apply_add) {
    CeedSize len;

    CeedCallBackend(CeedVectorGetLength(V, &len));
    for (CeedInt i = 0; i < len; i++) v[i] = 0.0;
  }

  CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor_basis));
  if (is_tensor_basis) {
    // Tensor basis
    CeedInt P_1d, Q_1d;

    CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
    CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
    switch (eval_mode) {
      // Interpolate to/from quadrature points
      case CEED_EVAL_INTERP: {
        if (impl->is_collocated) {
          memcpy(v, u, num_elem * num_comp * num_nodes * sizeof(u[0]));
        } else {
          CeedInt P = P_1d, Q = Q_1d;

          if (t_mode == CEED_TRANSPOSE) {
            P = Q_1d;
            Q = P_1d;
          }
          CeedInt           pre = num_comp * CeedIntPow(P, dim - 1), post = num_elem;
          CeedScalar        tmp[2][num_elem * num_comp * Q * CeedIntPow(P > Q ? P : Q, dim - 1)];
          const CeedScalar *interp_1d;

          CeedCallBackend(CeedBasisGetInterp1D(basis, &interp_1d));
          for (CeedInt d = 0; d < dim; d++) {
            CeedCallBackend(CeedTensorContractApply(contract, pre, P, post, Q, interp_1d, t_mode, add && (d == dim - 1), d == 0 ? u : tmp[d % 2],
                                                    d == dim - 1 ? v : tmp[(d + 1) % 2]));
            pre /= P;
            post *= Q;
          }
        }
      } break;
      // Evaluate the gradient to/from quadrature points
      case CEED_EVAL_GRAD: {
        // In CEED_NOTRANSPOSE mode:
        // u has shape [dim, num_comp, P^dim, num_elem], row-major layout
        // v has shape [dim, num_comp, Q^dim, num_elem], row-major layout
        // In CEED_TRANSPOSE mode, the sizes of u and v are switched.
        CeedInt P = P_1d, Q = Q_1d;

        if (t_mode == CEED_TRANSPOSE) {
          P = Q_1d;
          Q = Q_1d;
        }
        CeedInt           pre = num_comp * CeedIntPow(P, dim - 1), post = num_elem;
        const CeedScalar *interp_1d;

        CeedCallBackend(CeedBasisGetInterp1D(basis, &interp_1d));
        if (impl->collo_grad_1d) {
          CeedScalar tmp[2][num_elem * num_comp * Q * CeedIntPow(P > Q ? P : Q, dim - 1)];
          CeedScalar interp[num_elem * num_comp * Q * CeedIntPow(P > Q ? P : Q, dim - 1)];

          // Interpolate to quadrature points (NoTranspose)
          //  or Grad to quadrature points (Transpose)
          for (CeedInt d = 0; d < dim; d++) {
            CeedCallBackend(CeedTensorContractApply(contract, pre, P, post, Q, (t_mode == CEED_NOTRANSPOSE ? interp_1d : impl->collo_grad_1d), t_mode,
                                                    (t_mode == CEED_TRANSPOSE) && (d > 0),
                                                    (t_mode == CEED_NOTRANSPOSE ? (d == 0 ? u : tmp[d % 2]) : &u[d * num_qpts * num_comp * num_elem]),
                                                    (t_mode == CEED_NOTRANSPOSE ? (d == dim - 1 ? interp : tmp[(d + 1) % 2]) : interp)));
            pre /= P;
            post *= Q;
          }
          // Grad to quadrature points (NoTranspose)
          //  or Interpolate to nodes (Transpose)
          P = Q_1d, Q = Q_1d;
          if (t_mode == CEED_TRANSPOSE) {
            P = Q_1d;
            Q = P_1d;
          }
          pre = num_comp * CeedIntPow(P, dim - 1), post = num_elem;
          for (CeedInt d = 0; d < dim; d++) {
            CeedCallBackend(CeedTensorContractApply(contract, pre, P, post, Q, (t_mode == CEED_NOTRANSPOSE ? impl->collo_grad_1d : interp_1d), t_mode,
                                                    (t_mode == CEED_NOTRANSPOSE && apply_add) || (t_mode == CEED_TRANSPOSE && (d == dim - 1)),
                                                    (t_mode == CEED_NOTRANSPOSE ? interp : (d == 0 ? interp : tmp[d % 2])),
                                                    (t_mode == CEED_NOTRANSPOSE ? &v[d * num_qpts * num_comp * num_elem]
                                                                                : (d == dim - 1 ? v : tmp[(d + 1) % 2]))));
            pre /= P;
            post *= Q;
          }
        } else if (impl->is_collocated) {  // Qpts collocated with nodes
          const CeedScalar *grad_1d;

          CeedCallBackend(CeedBasisGetGrad1D(basis, &grad_1d));

          // Dim contractions, identity in other directions
          CeedInt pre = num_comp * CeedIntPow(P, dim - 1), post = num_elem;

          for (CeedInt d = 0; d < dim; d++) {
            CeedCallBackend(CeedTensorContractApply(contract, pre, P, post, Q, grad_1d, t_mode, add && (d > 0),
                                                    t_mode == CEED_NOTRANSPOSE ? u : &u[d * num_comp * num_qpts * num_elem],
                                                    t_mode == CEED_TRANSPOSE ? v : &v[d * num_comp * num_qpts * num_elem]));
            pre /= P;
            post *= Q;
          }
        } else {  // Underintegration, P > Q
          const CeedScalar *grad_1d;

          CeedCallBackend(CeedBasisGetGrad1D(basis, &grad_1d));

          if (t_mode == CEED_TRANSPOSE) {
            P = Q_1d;
            Q = P_1d;
          }
          CeedScalar tmp[2][num_elem * num_comp * Q * CeedIntPow(P > Q ? P : Q, dim - 1)];

          // Dim**2 contractions, apply grad when pass == dim
          for (CeedInt p = 0; p < dim; p++) {
            CeedInt pre = num_comp * CeedIntPow(P, dim - 1), post = num_elem;

            for (CeedInt d = 0; d < dim; d++) {
              CeedCallBackend(CeedTensorContractApply(
                  contract, pre, P, post, Q, (p == d) ? grad_1d : interp_1d, t_mode, add && (d == dim - 1),
                  (d == 0 ? (t_mode == CEED_NOTRANSPOSE ? u : &u[p * num_comp * num_qpts * num_elem]) : tmp[d % 2]),
                  (d == dim - 1 ? (t_mode == CEED_TRANSPOSE ? v : &v[p * num_comp * num_qpts * num_elem]) : tmp[(d + 1) % 2])));
              pre /= P;
              post *= Q;
            }
          }
        }
      } break;
      // Retrieve interpolation weights
      case CEED_EVAL_WEIGHT: {
        CeedInt           Q = Q_1d;
        const CeedScalar *q_weight_1d;

        CeedCheck(t_mode == CEED_NOTRANSPOSE, CeedBasisReturnCeed(basis), CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
        CeedCallBackend(CeedBasisGetQWeights(basis, &q_weight_1d));
        for (CeedInt d = 0; d < dim; d++) {
          CeedInt pre = CeedIntPow(Q, dim - d - 1), post = CeedIntPow(Q, d);

          for (CeedInt i = 0; i < pre; i++) {
            for (CeedInt j = 0; j < Q; j++) {
              for (CeedInt k = 0; k < post; k++) {
                const CeedScalar w = q_weight_1d[j] * (d == 0 ? 1 : v[((i * Q + j) * post + k) * num_elem]);

                for (CeedInt e = 0; e < num_elem; e++) v[((i * Q + j) * post + k) * num_elem + e] = w;
              }
            }
          }
        }
      } break;
      // LCOV_EXCL_START
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        return CeedError(CeedBasisReturnCeed(basis), CEED_ERROR_BACKEND, "%s not supported", CeedEvalModes[eval_mode]);
      case CEED_EVAL_NONE:
        return CeedError(CeedBasisReturnCeed(basis), CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
        // LCOV_EXCL_STOP
    }
  } else {
    // Non-tensor basis
    CeedInt P = num_nodes, Q = num_qpts;

    switch (eval_mode) {
      // Interpolate to/from quadrature points
      case CEED_EVAL_INTERP: {
        const CeedScalar *interp;

        CeedCallBackend(CeedBasisGetInterp(basis, &interp));
        CeedCallBackend(CeedTensorContractStridedApply(contract, num_comp, P, num_elem, q_comp, Q, interp, t_mode, add, u, v));
      } break;
      // Evaluate the gradient to/from quadrature points
      case CEED_EVAL_GRAD: {
        const CeedScalar *grad;

        CeedCallBackend(CeedBasisGetGrad(basis, &grad));
        CeedCallBackend(CeedTensorContractStridedApply(contract, num_comp, P, num_elem, q_comp, Q, grad, t_mode, add, u, v));
      } break;
      // Evaluate the divergence to/from the quadrature points
      case CEED_EVAL_DIV: {
        const CeedScalar *div;

        CeedCallBackend(CeedBasisGetDiv(basis, &div));
        CeedCallBackend(CeedTensorContractStridedApply(contract, num_comp, P, num_elem, q_comp, Q, div, t_mode, add, u, v));
      } break;
      // Evaluate the curl to/from the quadrature points
      case CEED_EVAL_CURL: {
        const CeedScalar *curl;

        CeedCallBackend(CeedBasisGetCurl(basis, &curl));
        CeedCallBackend(CeedTensorContractStridedApply(contract, num_comp, P, num_elem, q_comp, Q, curl, t_mode, add, u, v));
      } break;
      // Retrieve interpolation weights
      case CEED_EVAL_WEIGHT: {
        const CeedScalar *q_weight;

        CeedCheck(t_mode == CEED_NOTRANSPOSE, CeedBasisReturnCeed(basis), CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
        CeedCallBackend(CeedBasisGetQWeights(basis, &q_weight));
        for (CeedInt i = 0; i < num_qpts; i++) {
          for (CeedInt e = 0; e < num_elem; e++) v[i * num_elem + e] = q_weight[i];
        }
      } break;
      // LCOV_EXCL_START
      case CEED_EVAL_NONE:
        return CeedError(CeedBasisReturnCeed(basis), CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
        // LCOV_EXCL_STOP
    }
  }
  if (U != CEED_VECTOR_NONE) {
    CeedCallBackend(CeedVectorRestoreArrayRead(U, &u));
  }
  CeedCallBackend(CeedVectorRestoreArray(V, &v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApply_Ref(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector U, CeedVector V) {
  CeedCallBackend(CeedBasisApplyCore_Ref(basis, false, num_elem, t_mode, eval_mode, U, V));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAdd_Ref(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector U, CeedVector V) {
  CeedCallBackend(CeedBasisApplyCore_Ref(basis, true, num_elem, t_mode, eval_mode, U, V));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Destroy Tensor
//------------------------------------------------------------------------------
static int CeedBasisDestroyTensor_Ref(CeedBasis basis) {
  CeedBasis_Ref *impl;

  CeedCallBackend(CeedBasisGetData(basis, &impl));
  CeedCallBackend(CeedFree(&impl->collo_grad_1d));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Create Tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Ref(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis) {
  Ceed               ceed, ceed_parent;
  CeedBasis_Ref     *impl;
  CeedTensorContract contract;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetParent(ceed, &ceed_parent));

  CeedCallBackend(CeedCalloc(1, &impl));
  // Calculate collocated grad
  CeedCallBackend(CeedBasisIsCollocated(basis, &impl->is_collocated));
  if (Q_1d >= P_1d && !impl->is_collocated) {
    CeedCallBackend(CeedMalloc(Q_1d * Q_1d, &impl->collo_grad_1d));
    CeedCallBackend(CeedBasisGetCollocatedGrad(basis, impl->collo_grad_1d));
  }
  CeedCallBackend(CeedBasisSetData(basis, impl));

  CeedCallBackend(CeedTensorContractCreate(ceed_parent, &contract));
  CeedCallBackend(CeedBasisSetTensorContract(basis, contract));
  CeedCallBackend(CeedTensorContractDestroy(&contract));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAdd_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyTensor_Ref));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedDestroy(&ceed_parent));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Create Non-Tensor H^1
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Ref(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                          const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed               ceed, ceed_parent;
  CeedTensorContract contract;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetParent(ceed, &ceed_parent));

  CeedCallBackend(CeedTensorContractCreate(ceed_parent, &contract));
  CeedCallBackend(CeedBasisSetTensorContract(basis, contract));
  CeedCallBackend(CeedTensorContractDestroy(&contract));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAdd_Ref));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedDestroy(&ceed_parent));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Create Non-Tensor H(div)
//------------------------------------------------------------------------------
int CeedBasisCreateHdiv_Ref(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *div,
                            const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed               ceed, ceed_parent;
  CeedTensorContract contract;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetParent(ceed, &ceed_parent));

  CeedCallBackend(CeedTensorContractCreate(ceed_parent, &contract));
  CeedCallBackend(CeedBasisSetTensorContract(basis, contract));
  CeedCallBackend(CeedTensorContractDestroy(&contract));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAdd_Ref));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedDestroy(&ceed_parent));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Create Non-Tensor H(curl)
//------------------------------------------------------------------------------
int CeedBasisCreateHcurl_Ref(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                             const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed               ceed, ceed_parent;
  CeedTensorContract contract;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetParent(ceed, &ceed_parent));

  CeedCallBackend(CeedTensorContractCreate(ceed_parent, &contract));
  CeedCallBackend(CeedBasisSetTensorContract(basis, contract));
  CeedCallBackend(CeedTensorContractDestroy(&contract));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAdd_Ref));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedDestroy(&ceed_parent));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
