// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/hash.h>
#include <ceed/khash.h>
#include <libxsmm.h>
#include <stddef.h>

#include "ceed-xsmm.h"

//------------------------------------------------------------------------------
// Tensor Contract C=1
//------------------------------------------------------------------------------
static int CeedTensorContract_Xsmm_C1(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const float *restrict t,
                                      CeedTransposeMode t_mode, const CeedInt add, const float *restrict u, float *restrict v) {
  float alpha = 1.0, beta = 1.0;
  char  trans_u = 'N', trans_t = 'N';
  if ((t_mode == CEED_TRANSPOSE && C != 1) || (t_mode == CEED_NOTRANSPOSE && C == 1)) trans_t = 'T';
  if (!add) beta = 0.0;

  // libXSMM GEMM
  libxsmm_sgemm(&trans_t, &trans_u, &J, &A, &B, &alpha, &t[0], NULL, &u[0], NULL, &beta, &v[0], NULL);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Xsmm(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const float *restrict t,
                                        CeedTransposeMode t_mode, const CeedInt add, const float *restrict u, float *restrict v) {
  CeedTensorContract_Xsmm *impl;
  CeedCallBackend(CeedTensorContractGetData(contract, &impl));

  // Get kernel
  libxsmm_gemmfunction kernel;
  CeedHashIJKLMKey     key = {B, C, J, t_mode, add};
  khint_t              k   = kh_get(f32, impl->lookup_f32, key);
  CeedHashGetValue(impl->lookup_f32, k, kernel);

  // Run kernel or fallback to default implementation
  if (C != 1) {
    libxsmm_gemm_param gemm_param;
    gemm_param.b.primary = (float *)&t[0];
    for (CeedInt a = 0; a < A; a++) {
      gemm_param.a.primary = (float *)&u[a * B * C];
      gemm_param.c.primary = (float *)&v[a * J * C];
      kernel(&gemm_param);
    }
  } else {
    CeedTensorContract_Xsmm_C1(contract, A, B, C, J, t, t_mode, add, u, v);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Destroy
//------------------------------------------------------------------------------
static int CeedTensorContractDestroy_Xsmm(CeedTensorContract contract) {
  CeedTensorContract_Xsmm *impl;
  CeedCallBackend(CeedTensorContractGetData(contract, &impl));

  // Release the hash table (no need to free kernels)
  kh_destroy(f32, impl->lookup_f32);
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_f32_Xsmm(CeedBasis basis, CeedTensorContract contract) {
  Ceed ceed;
  CeedCallBackend(CeedTensorContractGetCeed(contract, &ceed));
  CeedTensorContract_Xsmm *impl;
  CeedCallBackend(CeedCalloc(1, &impl));

  // Setup kernels hash table
  impl->lookup_f32 = kh_init(f32);

  // Set up pointers to kernels
  CeedCallBackend(CeedBasisIsTensor(basis, &impl->is_tensor));
  if (impl->is_tensor) {
    CeedCallBackend(CeedBasisGetNumNodes1D(basis, &impl->P));
    CeedCallBackend(CeedBasisGetNumQuadraturePoints1D(basis, &impl->Q));
    CeedCallBackend(CeedBasisGetDimension(basis, &impl->dim));
    // Build all required kernels
    for (CeedInt num_elem = 1; num_elem <= 8; num_elem += 7) {
      for (CeedInt add = 0; add <= 1; add++) {
        for (CeedInt t_mode = 0; t_mode <= 1; t_mode++) {
          for (CeedInt grad = 0; grad <= 1; grad++) {
            for (CeedInt dim = 0; dim < impl->dim; dim++) {
              const int flags_t  = LIBXSMM_GEMM_FLAGS('N', t_mode ? 'T' : 'N');
              const int flags_ab = (!add) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
              const int flags    = (flags_t | flags_ab);
              CeedInt   B = grad ? impl->Q : (t_mode ? impl->Q : impl->P), J = grad ? impl->Q : (t_mode ? impl->P : impl->Q),
                      C = num_elem * CeedIntPow(J, dim);
              // Add key, kernel pair to hash table
              CeedHashIJKLMKey key = {B, C, J, t_mode, add};
              int              new_item;
              khint_t          k = kh_put(f32, impl->lookup_f32, key, &new_item);
              if (new_item) {
                // Build kernel
                const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(C, J, B, C, !t_mode ? B : J, C, LIBXSMM_DATATYPE_F32,
                                                                                LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
                libxsmm_gemmfunction     kernel =
                    libxsmm_dispatch_gemm_v2(gemm_shape, (libxsmm_bitfield)(flags), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
                CeedCheck(kernel, ceed, CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");
                // Add kernel to hash table
                kh_value(impl->lookup_f32, k) = kernel;
              }
            }
          }
        }
      }
    }
  } else {
    CeedCallBackend(CeedBasisGetNumNodes(basis, &impl->P));
    CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis, &impl->Q));
    CeedCallBackend(CeedBasisGetDimension(basis, &impl->dim));
    // Build all required kernels
    for (CeedInt num_elem = 1; num_elem <= 8; num_elem += 7) {
      for (CeedInt add = 0; add <= 1; add++) {
        for (CeedInt t_mode = 0; t_mode <= 1; t_mode++) {
          const CeedInt q_comp   = 1;
          const int     flags_t  = LIBXSMM_GEMM_FLAGS('N', t_mode ? 'T' : 'N');
          const int     flags_ab = (!add) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;
          const int     flags    = (flags_t | flags_ab);
          CeedInt       B = t_mode ? q_comp * impl->Q : impl->P, J = t_mode ? impl->P : q_comp * impl->Q, C = num_elem;
          // Add key, kernel pair to hash table
          CeedHashIJKLMKey key = {B, C, J, t_mode, add};
          int              new_item;
          khint_t          k = kh_put(f32, impl->lookup_f32, key, &new_item);
          if (new_item) {
            // Build kernel
            const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(C, J, B, C, !t_mode ? B : J, C, LIBXSMM_DATATYPE_F32,
                                                                            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
            libxsmm_gemmfunction     kernel =
                libxsmm_dispatch_gemm_v2(gemm_shape, (libxsmm_bitfield)(flags), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
            CeedCheck(kernel, ceed, CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");
            // Add kernel to hash table
            kh_value(impl->lookup_f32, k) = kernel;
          }
        }
      }
    }
  }
  CeedCallBackend(CeedTensorContractSetData(contract, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply", CeedTensorContractApply_Xsmm));
  CeedCallBackend(CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy", CeedTensorContractDestroy_Xsmm));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
