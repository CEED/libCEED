// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <libxsmm.h>

#include "ceed-xsmm.h"

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Xsmm(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                        CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  Ceed ceed;

  CeedCallBackend(CeedTensorContractGetCeed(contract, &ceed));

  if (C == 1) {
    // Build or query the required kernel
    const int                  flags_t    = LIBXSMM_GEMM_FLAGS(!t_mode ? 'T' : 'N', 'N');
    const int                  flags_ab   = (!add) ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_BASIC_GEMM_FLAG_NONE;
    const int                  flags      = (flags_t | flags_ab);
    const libxsmm_gemm_shape   gemm_shape = (CEED_SCALAR_TYPE == CEED_SCALAR_FP64)
                                                ? libxsmm_create_gemm_shape(J, A, B, !t_mode ? B : J, B, J, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
                                                                            LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64)
                                                : libxsmm_create_gemm_shape(J, A, B, !t_mode ? B : J, B, J, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                                            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    const libxsmm_gemmfunction kernel = libxsmm_dispatch_gemm_v2(gemm_shape, (libxsmm_bitfield)(flags), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
    libxsmm_gemm_param         gemm_param;

    CeedCheck(kernel, ceed, CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");

    // Run kernel
    gemm_param.a.primary = (CeedScalar *)&t[0];
    gemm_param.b.primary = (CeedScalar *)&u[0];
    gemm_param.c.primary = (CeedScalar *)&v[0];
    kernel(&gemm_param);
  } else {
    // Build or query the required kernel
    const int                  flags_t    = LIBXSMM_GEMM_FLAGS('N', t_mode ? 'T' : 'N');
    const int                  flags_ab   = (!add) ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_BASIC_GEMM_FLAG_NONE;
    const int                  flags      = (flags_t | flags_ab);
    const libxsmm_gemm_shape   gemm_shape = (CEED_SCALAR_TYPE == CEED_SCALAR_FP64)
                                                ? libxsmm_create_gemm_shape(C, J, B, C, !t_mode ? B : J, C, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
                                                                            LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64)
                                                : libxsmm_create_gemm_shape(C, J, B, C, !t_mode ? B : J, C, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                                            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    const libxsmm_gemmfunction kernel = libxsmm_dispatch_gemm_v2(gemm_shape, (libxsmm_bitfield)(flags), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
    libxsmm_gemm_param         gemm_param;

    CeedCheck(kernel, ceed, CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");

    // Run kernel
    gemm_param.b.primary = (CeedScalar *)&t[0];
    for (CeedInt a = 0; a < A; a++) {
      gemm_param.a.primary = (CeedScalar *)&u[a * B * C];
      gemm_param.c.primary = (CeedScalar *)&v[a * J * C];
      kernel(&gemm_param);
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Xsmm(CeedTensorContract contract) {
  CeedCallBackend(CeedSetBackendFunction(CeedTensorContractReturnCeed(contract), "TensorContract", contract, "Apply", CeedTensorContractApply_Xsmm));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
