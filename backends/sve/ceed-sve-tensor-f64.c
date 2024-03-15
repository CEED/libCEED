// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#include <stdbool.h>

#include "ceed-sve.h"

//------------------------------------------------------------------------------
// Blocked Tensor Contract
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Sve_Blocked(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                                 CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v,
                                                 const CeedInt JJ) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;

  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt b = 0; b < B; b++) {
      // Blocks of JJ rows
      for (CeedInt j = 0; j < (J / JJ) * JJ; j += JJ) {
        for (CeedInt jj = 0; jj < JJ; jj++) {  // unroll
          // C vectorization by compiler
          for (int32_t c = 0; c < C; c += svcntd()) {
            svbool_t pg = svwhilelt_b64(c, C);
            // Load u, v into vectors
            svfloat64_t u_vec = svld1(pg, &u[(a * B + b) * C + c]);
            svfloat64_t v_vec = svld1(pg, &v[(a * J + j + jj) * C + c]);
            // Basis matrix value
            double tq = t[(j + jj) * t_stride_0 + b * t_stride_1];

            // fmadd
            svst1(pg, &v[(a * J + j + jj) * C + c], svmla_x(pg, v_vec, u_vec, tq));
          }
        }
      }
    }
  }

  // Remainder of rows
  const CeedInt j = (J / JJ) * JJ;

  if (j < J) {
    for (CeedInt a = 0; a < A; a++) {
      for (CeedInt b = 0; b < B; b++) {
        // Blocks of JJ rows
        for (CeedInt jj = 0; jj < J - j; jj++) {  // not unrolled
          // C vectorization by compiler
          for (int32_t c = 0; c < C; c += svcntd()) {
            svbool_t pg = svwhilelt_b64(c, C);
            // Load u, v into vectors
            svfloat64_t u_vec = svld1(pg, &u[(a * B + b) * C + c]);
            svfloat64_t v_vec = svld1(pg, &v[(a * J + j + jj) * C + c]);
            // Basis matrix value
            double tq = t[(j + jj) * t_stride_0 + b * t_stride_1];

            // fmadd
            svst1(pg, &v[(a * J + j + jj) * C + c], svmla_x(pg, v_vec, u_vec, tq));
          }
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Blocked Tensor Contract
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Sve_Serial(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                                CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v,
                                                const CeedInt JJ) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;

  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt b = 0; b < B; b++) {
      for (CeedInt j = 0; j < (J / JJ) * JJ; j += JJ) {
        for (CeedInt jj = 0; jj < JJ; jj++) {  // unroll
          v[a * J + (j + jj)] += t[(j + jj) * t_stride_0 + b * t_stride_1] * u[a * B + b];
        }
      }
    }
  }

  const CeedInt j = (J / JJ) * JJ;

  if (j < J) {
    for (CeedInt a = 0; a < A; a++) {
      for (CeedInt b = 0; b < B; b++) {
        for (CeedInt jj = 0; jj < J - j; jj++) {  // not unrolled
          v[a * J + (j + jj)] += t[(j + jj) * t_stride_0 + b * t_stride_1] * u[a * B + b];
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract - Common Sizes
//------------------------------------------------------------------------------
static int CeedTensorContract_Sve_Blocked_8(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                            CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v) {
  return CeedTensorContract_Sve_Blocked(contract, A, B, C, J, t, t_mode, add, u, v, 8);
}
static int CeedTensorContract_Sve_Serial_8(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                           CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v) {
  return CeedTensorContract_Sve_Serial(contract, A, B, C, J, t, t_mode, add, u, v, 8);
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Sve(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                       CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v) {
  if (!add) {
    for (CeedInt q = 0; q < A * J * C; q++) v[q] = (double)0.0;
  }

  if (C == 1) CeedTensorContract_Sve_Serial_8(contract, A, B, C, J, t, t_mode, true, u, v);
  else CeedTensorContract_Sve_Blocked_8(contract, A, B, C, J, t, t_mode, true, u, v);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_f64_Sve(CeedBasis basis, CeedTensorContract contract) {
  Ceed ceed;

  CeedCallBackend(CeedTensorContractGetCeed(contract, &ceed));
  CeedCallBackend(CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply", CeedTensorContractApply_Sve));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
