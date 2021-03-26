// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/hash.h>
#include <ceed/khash.h>
#include <libxsmm.h>
#include <stddef.h>
#include "ceed-xsmm.h"

//------------------------------------------------------------------------------
// Tensor Contract C=1
//------------------------------------------------------------------------------
static int CeedTensorContract_Xsmm_C1(CeedTensorContract contract,
                                      CeedInt A, CeedInt B, CeedInt C,
                                      CeedInt J, const CeedScalar *restrict t,
                                      CeedTransposeMode tmode,
                                      const CeedInt add,
                                      const CeedScalar *restrict u,
                                      CeedScalar *restrict v) {
  CeedScalar alpha = 1.0, beta = 1.0;
  char transu = 'N', transt = 'N';
  if ((tmode == CEED_TRANSPOSE && C != 1)
      || (tmode == CEED_NOTRANSPOSE && C == 1))
    transt = 'T';

  if (!add)
    beta = 0.0;

  // libXSMM GEMM
  libxsmm_dgemm(&transt, &transu, &J, &A, &B,
                &alpha, &t[0], NULL, &u[0], NULL,
                &beta, &v[0], NULL);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Xsmm(CeedTensorContract contract, CeedInt A,
                                        CeedInt B, CeedInt C, CeedInt J,
                                        const CeedScalar *restrict t,
                                        CeedTransposeMode tmode,
                                        const CeedInt add,
                                        const CeedScalar *restrict u,
                                        CeedScalar *restrict v) {
  int ierr;
  CeedTensorContract_Xsmm *impl;
  ierr = CeedTensorContractGetData(contract, &impl); CeedChkBackend(ierr);

  // Get kernel
  libxsmm_dmmfunction kernel;
  CeedHashIJKLMKey key = {B, C, J, tmode, add};
  khint_t k = kh_get(m32, impl->lookup, key);
  CeedHashGetValue(impl->lookup, k, kernel);

  // Run kernel or fallback to default implementation
  if (C != 1)
    for (CeedInt a=0; a<A; a++)
      kernel(&u[a*B*C], &t[0], &v[a*J*C], NULL, NULL, NULL);
  else
    CeedTensorContract_Xsmm_C1(contract, A, B, C, J, t, tmode, add, u, v);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Destroy
//------------------------------------------------------------------------------
static int CeedTensorContractDestroy_Xsmm(CeedTensorContract contract) {
  int ierr;
  CeedTensorContract_Xsmm *impl;
  libxsmm_dmmfunction kernel;

  ierr = CeedTensorContractGetData(contract, &impl); CeedChkBackend(ierr);
  // Free kernels
  kh_foreach_value(impl->lookup, kernel, libxsmm_release_kernel(&kernel));
  kh_destroy(m32, impl->lookup);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Xsmm(CeedBasis basis,
                                  CeedTensorContract contract) {
  int ierr;
  Ceed ceed;
  ierr = CeedTensorContractGetCeed(contract, &ceed); CeedChkBackend(ierr);
  CeedTensorContract_Xsmm *impl;
  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);

  // Setup kernels hash table
  impl->lookup = kh_init(m32);

  // Set up pointers to kernels
  ierr = CeedBasisIsTensor(basis, &impl->isTensor); CeedChkBackend(ierr);
  if (impl->isTensor) {
    ierr = CeedBasisGetNumNodes1D(basis, &impl->P); CeedChkBackend(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &impl->Q); CeedChkBackend(ierr);
    ierr = CeedBasisGetDimension(basis, &impl->dim); CeedChkBackend(ierr);
    // Build all required kernels
    for (CeedInt nelem = 1; nelem <= 8; nelem+=7)
      for (CeedInt add = 0; add <= 1; add++)
        for (CeedInt tmode = 0; tmode <= 1; tmode++)
          for (CeedInt grad = 0; grad <=1; grad++)
            for (CeedInt dim = 0; dim < impl->dim; dim++) {
              const int flags = LIBXSMM_GEMM_FLAGS('N', tmode ? 'T' : 'N');
              CeedInt B = grad ? impl->Q : (tmode ? impl->Q : impl->P),
                      J = grad ? impl->Q : (tmode ? impl->P : impl->Q),
                      C = nelem*CeedIntPow(J, dim);
              // Add key, kernel pair to hash table
              CeedHashIJKLMKey key = {B, C, J, tmode, add};
              int new_item;
              khint_t k = kh_put(m32, impl->lookup, key, &new_item);
              if (new_item) {
                // Build kernel
                CeedScalar alpha = 1.0, beta = 1.0;
                if (!add) beta = 0.0;
                libxsmm_dmmfunction kernel = libxsmm_dmmdispatch(
                                               C, J, B, NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
                if (!kernel)
                  // LCOV_EXCL_START
                  return CeedError(ceed, CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");
                // LCOV_EXCL_STOP
                // Add kernel to hash table
                kh_value(impl->lookup, k) = kernel;
              }
            }
  } else {
    ierr = CeedBasisGetNumNodes(basis, &impl->P); CeedChkBackend(ierr);
    ierr = CeedBasisGetNumQuadraturePoints(basis, &impl->Q); CeedChkBackend(ierr);
    ierr = CeedBasisGetDimension(basis, &impl->dim); CeedChkBackend(ierr);
    // Build all required kernels
    for (CeedInt nelem = 1; nelem <= 8; nelem+=7)
      for (CeedInt add = 0; add <= 1; add++)
        for (CeedInt tmode = 0; tmode <= 1; tmode++) {
          CeedInt gradstride = CeedIntMax(impl->dim-1, 1);
          for (CeedInt grad = 1; grad <= impl->dim; grad+=gradstride) {
            const int flags = LIBXSMM_GEMM_FLAGS('N', tmode ? 'T' : 'N');
            CeedInt B = tmode ? grad*impl->Q : impl->P,
                    J = tmode ? impl->P : grad*impl->Q,
                    C = nelem;
            // Add key, kernel pair to hash table
            CeedHashIJKLMKey key = {B, C, J, tmode, add};
            int new_item;
            khint_t k = kh_put(m32, impl->lookup, key, &new_item);
            if (new_item) {
              // Build kernel
              CeedScalar alpha = 1.0, beta = 1.0;
              if (!add) beta = 0.0;
              libxsmm_dmmfunction kernel = libxsmm_dmmdispatch(
                                             C, J, B, NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
              if (!kernel)
                // LCOV_EXCL_START
                return CeedError(ceed, CEED_ERROR_BACKEND, "LIBXSMM kernel failed to build.");
              // LCOV_EXCL_STOP
              // Add kernel to hash table
              kh_value(impl->lookup, k) = kernel;
            }
          }
        }
  }
  ierr = CeedTensorContractSetData(contract, impl); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply",
                                CeedTensorContractApply_Xsmm); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy",
                                CeedTensorContractDestroy_Xsmm); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
