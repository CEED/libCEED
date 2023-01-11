// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Ref(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                       CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;
  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  if (!add) {
    for (CeedInt q = 0; q < A * J * C; q++) v[q] = (CeedScalar)0.0;
  }

  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt b = 0; b < B; b++) {
      for (CeedInt j = 0; j < J; j++) {
        CeedScalar tq = t[j * t_stride_0 + b * t_stride_1];
        for (CeedInt c = 0; c < C; c++) v[(a * J + j) * C + c] += tq * u[(a * B + b) * C + c];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Destroy
//------------------------------------------------------------------------------
static int CeedTensorContractDestroy_Ref(CeedTensorContract contract) { return CEED_ERROR_SUCCESS; }

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Ref(CeedBasis basis, CeedTensorContract contract) {
  Ceed ceed;
  CeedCallBackend(CeedTensorContractGetCeed(contract, &ceed));

  CeedCallBackend(CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply", CeedTensorContractApply_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy", CeedTensorContractDestroy_Ref));

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
