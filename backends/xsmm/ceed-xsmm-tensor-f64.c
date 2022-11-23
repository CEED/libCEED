// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/hash.h>
#include <ceed/khash.h>
#include <libxsmm.h>
#include <stddef.h>

#include "ceed-xsmm.h"

//------------------------------------------------------------------------------
// Tensor Contract C=1
//------------------------------------------------------------------------------
static int CeedTensorContract_Xsmm_C1(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                      CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v) {
  double alpha = 1.0, beta = 1.0;
  char   trans_u = 'N', trans_t = 'N';
  if ((t_mode == CEED_TRANSPOSE && C != 1) || (t_mode == CEED_NOTRANSPOSE && C == 1)) trans_t = 'T';

  if (!add) beta = 0.0;

  // libXSMM GEMM
  libxsmm_dgemm(&trans_t, &trans_u, &J, &A, &B, &alpha, &t[0], NULL, &u[0], NULL, &beta, &v[0], NULL);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Xsmm(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                        CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v) {
  CeedTensorContract_Xsmm *impl;
  CeedCallBackend(CeedTensorContractGetData(contract, &impl));

  // Get kernel
  libxsmm_dmmfunction kernel;
  CeedHashIJKLMKey    key = {B, C, J, t_mode, add};
  khint_t             k   = kh_get(f64, impl->lookup_f64, key);
  CeedHashGetValue(impl->lookup_f64, k, kernel);

  // Run kernel or fallback to default implementation
  if (C != 1) {
    for (CeedInt a = 0; a < A; a++) LIBXSMM_MMFUNCTION_KERNEL(&u[a * B * C], &t[0], &v[a * J * C]);
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
  libxsmm_dmmfunction      kernel;

  CeedCallBackend(CeedTensorContractGetData(contract, &impl));
  // Free kernels
  kh_foreach_value(impl->lookup_f64, kernel, libxsmm_release_kernel(&kernel));
  kh_destroy(f64, impl->lookup_f64);
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_f64_Xsmm(CeedBasis basis, CeedTensorContract contract) {
  Ceed ceed;
  CeedCallBackend(CeedTensorContractGetCeed(contract, &ceed));
  CeedTensorContract_Xsmm *impl;
  CeedCallBackend(CeedCalloc(1, &impl));

  // Setup kernels hash table
  impl->lookup_f64 = kh_init(f64);

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
              const int flags = LIBXSMM_GEMM_FLAGS('N', t_mode ? 'T' : 'N');
              CeedInt   B = grad ? impl->Q : (t_mode ? impl->Q : impl->P), J = grad ? impl->Q : (t_mode ? impl->P : impl->Q),
                      C = num_elem * CeedIntPow(J, dim);
              // Add key, kernel pair to hash table
              CeedHashIJKLMKey key = {B, C, J, t_mode, add};
              int              new_item;
              khint_t          k = kh_put(f64, impl->lookup_f64, key, &new_item);
              if (new_item) {
                // Build kernel
                double alpha = 1.0, beta = 1.0;
                if (!add) beta = 0.0;
                libxsmm_dmmfunction kernel = libxsmm_dmmdispatch(C, J, B, NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
                if (!kernel)
                  // LCOV_EXCL_START
                  return CeedError(ceed, CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");
                // LCOV_EXCL_STOP
                // Add kernel to hash table
                kh_value(impl->lookup_f64, k) = kernel;
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
          CeedInt gradstride = CeedIntMax(impl->dim - 1, 1);
          for (CeedInt grad = 1; grad <= impl->dim; grad += gradstride) {
            const int flags = LIBXSMM_GEMM_FLAGS('N', t_mode ? 'T' : 'N');
            CeedInt   B = t_mode ? grad * impl->Q : impl->P, J = t_mode ? impl->P : grad * impl->Q, C = num_elem;
            // Add key, kernel pair to hash table
            CeedHashIJKLMKey key = {B, C, J, t_mode, add};
            int              new_item;
            khint_t          k = kh_put(f64, impl->lookup_f64, key, &new_item);
            if (new_item) {
              // Build kernel
              double alpha = 1.0, beta = 1.0;
              if (!add) beta = 0.0;
              libxsmm_dmmfunction kernel = libxsmm_dmmdispatch(C, J, B, NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
              if (!kernel)
                // LCOV_EXCL_START
                return CeedError(ceed, CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");
              // LCOV_EXCL_STOP
              // Add kernel to hash table
              kh_value(impl->lookup_f64, k) = kernel;
            }
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
