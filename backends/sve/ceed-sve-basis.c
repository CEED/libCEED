// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <arm_sve.h>
#include <limits.h>
#include <stdbool.h>

#include "ceed-sve.h"

#ifdef CEED_SCALAR_IS_FP64
#define basis_vlength() ((CeedInt)svcntd())
#else
#define basis_vlength() ((CeedInt)svcntw())
#endif

//------------------------------------------------------------------------------
// Basis Apply
//------------------------------------------------------------------------------
static int CeedBasisApplyCore_Sve(CeedBasis basis, bool apply_add, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector U,
                                  CeedVector V) {
  CeedInt            num_comp, q_comp, P, Q;
  const CeedScalar  *u = NULL, *t = NULL;
  CeedScalar        *v;
  CeedTensorContract contract;

  CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, eval_mode, &q_comp));
  CeedCallBackend(CeedBasisGetNumNodes(basis, &P));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &Q));
  switch (eval_mode) {
    case CEED_EVAL_INTERP:
      CeedCallBackend(CeedBasisGetInterp(basis, &t));
      break;
    case CEED_EVAL_GRAD:
      CeedCallBackend(CeedBasisGetGrad(basis, &t));
      break;
    case CEED_EVAL_DIV:
      CeedCallBackend(CeedBasisGetDiv(basis, &t));
      break;
    case CEED_EVAL_CURL:
      CeedCallBackend(CeedBasisGetCurl(basis, &t));
      break;
    case CEED_EVAL_WEIGHT:
      CeedCallBackend(CeedBasisGetQWeights(basis, &t));
      break;
    case CEED_EVAL_NONE:
      return CeedError(CeedBasisReturnCeed(basis), CEED_ERROR_BACKEND, "CEED_EVAL_NONE does not make sense in this context");
  }
  if (U != CEED_VECTOR_NONE) {
    CeedCallBackend(CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u));
  } else {
    CeedCheck(eval_mode == CEED_EVAL_WEIGHT, CeedBasisReturnCeed(basis), CEED_ERROR_BACKEND, "An input vector is required for this CeedEvalMode");
  }
  if (apply_add)
    CeedCallBackend(CeedVectorGetArray(V, CEED_MEM_HOST, &v));
  else
    CeedCallBackend(CeedVectorGetArrayWrite(V, CEED_MEM_HOST, &v));

  if (eval_mode == CEED_EVAL_WEIGHT) {
    CeedCheck(t_mode == CEED_NOTRANSPOSE, CeedBasisReturnCeed(basis), CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    for (CeedInt q = 0; q < Q; q++) {
      for (CeedInt e = 0; e < num_elem; e++) v[(CeedSize)q * num_elem + e] = t[q];
    }
  } else {
    // The 256-entry transpose limit is empirical; long forward maps need a full four-vector tile.
    const CeedSize q_size                  = (CeedSize)q_comp * Q;
    const CeedSize max_flattened_reduction = 256;
    const CeedInt  vector_length           = basis_vlength();
    const bool     flatten =
        num_comp == 1 && (q_size <= max_flattened_reduction || (q_size <= INT_MAX && t_mode == CEED_NOTRANSPOSE && num_elem / 4 >= vector_length));

    CeedCallBackend(CeedBasisGetTensorContract(basis, &contract));
    if (flatten) {
      if (t_mode == CEED_TRANSPOSE) {
        CeedCallBackend(CeedTensorContractApply(contract, 1, (CeedInt)q_size, num_elem, P, t, CEED_TRANSPOSE, apply_add, u, v));
      } else {
        CeedCallBackend(CeedTensorContractApply(contract, 1, P, num_elem, (CeedInt)q_size, t, CEED_NOTRANSPOSE, apply_add, u, v));
      }
    } else {
      if (t_mode == CEED_TRANSPOSE && !apply_add) {
        CeedSize length;

        CeedCallBackend(CeedVectorGetLength(V, &length));
        for (CeedSize i = 0; i < length; i++) v[i] = 0.0;
      }
      CeedCallBackend(CeedTensorContractStridedApply(contract, num_comp, P, num_elem, q_comp, Q, t, t_mode, apply_add || t_mode == CEED_TRANSPOSE, u,
                                                     v));
    }
  }
  if (U != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorRestoreArrayRead(U, &u));
  CeedCallBackend(CeedVectorRestoreArray(V, &v));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApply_Sve(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector U, CeedVector V) {
  CeedCallBackend(CeedBasisApplyCore_Sve(basis, false, num_elem, t_mode, eval_mode, U, V));
  return CEED_ERROR_SUCCESS;
}

static int CeedBasisApplyAdd_Sve(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector U, CeedVector V) {
  CeedCallBackend(CeedBasisApplyCore_Sve(basis, true, num_elem, t_mode, eval_mode, U, V));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis Create
//------------------------------------------------------------------------------
static int CeedBasisCreate_Sve(CeedBasis basis) {
  Ceed               ceed, ceed_parent;
  CeedTensorContract contract;

  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
  CeedCallBackend(CeedGetParent(ceed, &ceed_parent));
  CeedCallBackend(CeedTensorContractCreate(ceed_parent, &contract));
  CeedCallBackend(CeedBasisSetTensorContract(basis, contract));
  CeedCallBackend(CeedTensorContractDestroy(&contract));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Sve));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAdd_Sve));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedDestroy(&ceed_parent));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisCreateH1_Sve(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
                          const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreate_Sve(basis));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisCreateHdiv_Sve(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *div,
                            const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreate_Sve(basis));
  return CEED_ERROR_SUCCESS;
}

int CeedBasisCreateHcurl_Sve(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                             const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis) {
  CeedCallBackend(CeedBasisCreate_Sve(basis));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
