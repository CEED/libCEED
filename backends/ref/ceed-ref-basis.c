// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Basis Apply
//------------------------------------------------------------------------------
static int CeedBasisApply_Ref(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector U, CeedVector V) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedInt dim, num_comp, num_nodes, num_qpts, Q_comp;
  CeedCallBackend(CeedBasisGetDimension(basis, &dim));
  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, &Q_comp));
  CeedTensorContract contract;
  CeedCallBackend(CeedBasisGetTensorContract(basis, &contract));
  const CeedInt     add = (t_mode == CEED_TRANSPOSE);
  const CeedScalar *u;
  CeedScalar       *v;
  if (U != CEED_VECTOR_NONE) {
    CeedCallBackend(CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u));
  } else if (eval_mode != CEED_EVAL_WEIGHT) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
    // LCOV_EXCL_STOP
  }
  CeedCallBackend(CeedVectorGetArrayWrite(V, CEED_MEM_HOST, &v));

  // Clear v if operating in transpose
  if (t_mode == CEED_TRANSPOSE) {
    const CeedInt v_size = num_elem * num_comp * num_nodes;
    for (CeedInt i = 0; i < v_size; i++) v[i] = (CeedScalar)0.0;
  }
  bool tensor_basis;
  CeedCallBackend(CeedBasisIsTensor(basis, &tensor_basis));
  // Tensor basis
  if (tensor_basis) {
    CeedInt P_1d, Q_1d;
    CeedCallBackend(CeedBasisGetNumNodes1D(basis, &P_1d));
    CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
    switch (eval_mode) {
      // Interpolate to/from quadrature points
      case CEED_EVAL_INTERP: {
        CeedBasis_Ref *impl;
        CeedCallBackend(CeedBasisGetData(basis, &impl));
        if (impl->has_collo_interp) {
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
          P = Q_1d, Q = Q_1d;
        }
        CeedBasis_Ref *impl;
        CeedCallBackend(CeedBasisGetData(basis, &impl));
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
                                                    add && (d > 0),
                                                    (t_mode == CEED_NOTRANSPOSE ? (d == 0 ? u : tmp[d % 2]) : u + d * num_qpts * num_comp * num_elem),
                                                    (t_mode == CEED_NOTRANSPOSE ? (d == dim - 1 ? interp : tmp[(d + 1) % 2]) : interp)));
            pre /= P;
            post *= Q;
          }
          // Grad to quadrature points (NoTranspose)
          //  or Interpolate to nodes (Transpose)
          P = Q_1d, Q = Q_1d;
          if (t_mode == CEED_TRANSPOSE) {
            P = Q_1d, Q = P_1d;
          }
          pre = num_comp * CeedIntPow(P, dim - 1), post = num_elem;
          for (CeedInt d = 0; d < dim; d++) {
            CeedCallBackend(CeedTensorContractApply(
                contract, pre, P, post, Q, (t_mode == CEED_NOTRANSPOSE ? impl->collo_grad_1d : interp_1d), t_mode, add && (d == dim - 1),
                (t_mode == CEED_NOTRANSPOSE ? interp : (d == 0 ? interp : tmp[d % 2])),
                (t_mode == CEED_NOTRANSPOSE ? v + d * num_qpts * num_comp * num_elem : (d == dim - 1 ? v : tmp[(d + 1) % 2]))));
            pre /= P;
            post *= Q;
          }
        } else if (impl->has_collo_interp) {  // Qpts collocated with nodes
          const CeedScalar *grad_1d;
          CeedCallBackend(CeedBasisGetGrad1D(basis, &grad_1d));

          // Dim contractions, identity in other directions
          CeedInt pre = num_comp * CeedIntPow(P, dim - 1), post = num_elem;
          for (CeedInt d = 0; d < dim; d++) {
            CeedCallBackend(CeedTensorContractApply(contract, pre, P, post, Q, grad_1d, t_mode, add && (d > 0),
                                                    t_mode == CEED_NOTRANSPOSE ? u : u + d * num_comp * num_qpts * num_elem,
                                                    t_mode == CEED_TRANSPOSE ? v : v + d * num_comp * num_qpts * num_elem));
            pre /= P;
            post *= Q;
          }
        } else {  // Underintegration, P > Q
          const CeedScalar *grad_1d;
          CeedCallBackend(CeedBasisGetGrad1D(basis, &grad_1d));

          if (t_mode == CEED_TRANSPOSE) {
            P = Q_1d, Q = P_1d;
          }
          CeedScalar tmp[2][num_elem * num_comp * Q * CeedIntPow(P > Q ? P : Q, dim - 1)];

          // Dim**2 contractions, apply grad when pass == dim
          for (CeedInt p = 0; p < dim; p++) {
            CeedInt pre = num_comp * CeedIntPow(P, dim - 1), post = num_elem;
            for (CeedInt d = 0; d < dim; d++) {
              CeedCallBackend(CeedTensorContractApply(
                  contract, pre, P, post, Q, (p == d) ? grad_1d : interp_1d, t_mode, add && (d == dim - 1),
                  (d == 0 ? (t_mode == CEED_NOTRANSPOSE ? u : u + p * num_comp * num_qpts * num_elem) : tmp[d % 2]),
                  (d == dim - 1 ? (t_mode == CEED_TRANSPOSE ? v : v + p * num_comp * num_qpts * num_elem) : tmp[(d + 1) % 2])));
              pre /= P;
              post *= Q;
            }
          }
        }
      } break;
      // Retrieve interpolation weights
      case CEED_EVAL_WEIGHT: {
        if (t_mode == CEED_TRANSPOSE) {
          // LCOV_EXCL_START
          return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
          // LCOV_EXCL_STOP
        }
        CeedInt           Q = Q_1d;
        const CeedScalar *q_weight_1d;
        CeedCallBackend(CeedBasisGetQWeights(basis, &q_weight_1d));
        for (CeedInt d = 0; d < dim; d++) {
          CeedInt pre = CeedIntPow(Q, dim - d - 1), post = CeedIntPow(Q, d);
          for (CeedInt i = 0; i < pre; i++) {
            for (CeedInt j = 0; j < Q; j++) {
              for (CeedInt k = 0; k < post; k++) {
                CeedScalar w = q_weight_1d[j] * (d == 0 ? 1 : v[((i * Q + j) * post + k) * num_elem]);
                for (CeedInt e = 0; e < num_elem; e++) v[((i * Q + j) * post + k) * num_elem + e] = w;
              }
            }
          }
        }
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
  } else {
    // Non-tensor basis
    switch (eval_mode) {
      // Interpolate to/from quadrature points
      case CEED_EVAL_INTERP: {
        CeedInt           P = num_nodes, Q = Q_comp * num_qpts;
        const CeedScalar *interp;
        CeedCallBackend(CeedBasisGetInterp(basis, &interp));
        if (t_mode == CEED_TRANSPOSE) {
          P = Q_comp * num_qpts;
          Q = num_nodes;
        }
        CeedCallBackend(CeedTensorContractApply(contract, num_comp, P, num_elem, Q, interp, t_mode, add, u, v));
      } break;
      // Evaluate the gradient to/from quadrature points
      case CEED_EVAL_GRAD: {
        CeedInt           P = num_nodes, Q = num_qpts;
        CeedInt           dim_stride  = num_qpts * num_comp * num_elem;
        CeedInt           grad_stride = num_qpts * num_nodes;
        const CeedScalar *grad;
        CeedCallBackend(CeedBasisGetGrad(basis, &grad));
        if (t_mode == CEED_TRANSPOSE) {
          P = num_qpts;
          Q = num_nodes;
          for (CeedInt d = 0; d < dim; d++) {
            CeedCallBackend(CeedTensorContractApply(contract, num_comp, P, num_elem, Q, grad + d * grad_stride, t_mode, add, u + d * dim_stride, v));
          }
        } else {
          for (CeedInt d = 0; d < dim; d++) {
            CeedCallBackend(CeedTensorContractApply(contract, num_comp, P, num_elem, Q, grad + d * grad_stride, t_mode, add, u, v + d * dim_stride));
          }
        }
      } break;
      // Retrieve interpolation weights
      case CEED_EVAL_WEIGHT: {
        if (t_mode == CEED_TRANSPOSE) {
          // LCOV_EXCL_START
          return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
          // LCOV_EXCL_STOP
        }
        const CeedScalar *q_weight;
        CeedCallBackend(CeedBasisGetQWeights(basis, &q_weight));
        for (CeedInt i = 0; i < num_qpts; i++) {
          for (CeedInt e = 0; e < num_elem; e++) v[i * num_elem + e] = q_weight[i];
        }
      } break;
      // Evaluate the divergence to/from the quadrature points
      case CEED_EVAL_DIV: {
        CeedInt           P = num_nodes, Q = num_qpts;
        const CeedScalar *div;
        CeedCallBackend(CeedBasisGetDiv(basis, &div));
        if (t_mode == CEED_TRANSPOSE) {
          P = num_qpts;
          Q = num_nodes;
        }
        CeedCallBackend(CeedTensorContractApply(contract, num_comp, P, num_elem, Q, div, t_mode, add, u, v));
      } break;
      // LCOV_EXCL_START
      // Evaluate the curl to/from the quadrature points
      case CEED_EVAL_CURL:
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
      // Take no action, BasisApply should not have been called
      case CEED_EVAL_NONE:
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
        // LCOV_EXCL_STOP
    }
  }
  if (U != CEED_VECTOR_NONE) {
    CeedCallBackend(CeedVectorRestoreArrayRead(U, &u));
  }
  CeedCallBackend(CeedVectorRestoreArray(V, &v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Create Non-Tensor H^1
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Ref(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                          const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  Ceed parent;
  CeedCallBackend(CeedGetParent(ceed, &parent));
  CeedTensorContract contract;
  CeedCallBackend(CeedTensorContractCreate(parent, basis, &contract));
  CeedCallBackend(CeedBasisSetTensorContract(basis, contract));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Ref));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Create Non-Tensor H(div)
//------------------------------------------------------------------------------
int CeedBasisCreateHdiv_Ref(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *div,
                            const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  Ceed parent;
  CeedCallBackend(CeedGetParent(ceed, &parent));
  CeedTensorContract contract;
  CeedCallBackend(CeedTensorContractCreate(parent, basis, &contract));
  CeedCallBackend(CeedBasisSetTensorContract(basis, contract));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Ref));

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
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedBasis_Ref *impl;
  CeedCallBackend(CeedCalloc(1, &impl));
  // Check for collocated interp
  if (Q_1d == P_1d) {
    bool collocated = 1;
    for (CeedInt i = 0; i < P_1d; i++) {
      collocated = collocated && (fabs(interp_1d[i + P_1d * i] - 1.0) < 1e-14);
      for (CeedInt j = 0; j < P_1d; j++) {
        if (j != i) collocated = collocated && (fabs(interp_1d[j + P_1d * i]) < 1e-14);
      }
    }
    impl->has_collo_interp = collocated;
  }
  // Calculate collocated grad
  if (Q_1d >= P_1d && !impl->has_collo_interp) {
    CeedCallBackend(CeedMalloc(Q_1d * Q_1d, &impl->collo_grad_1d));
    CeedCallBackend(CeedBasisGetCollocatedGrad(basis, impl->collo_grad_1d));
  }
  CeedCallBackend(CeedBasisSetData(basis, impl));

  Ceed parent;
  CeedCallBackend(CeedGetParent(ceed, &parent));
  CeedTensorContract contract;
  CeedCallBackend(CeedTensorContractCreate(parent, basis, &contract));
  CeedCallBackend(CeedBasisSetTensorContract(basis, contract));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyTensor_Ref));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
