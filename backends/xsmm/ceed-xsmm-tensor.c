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

#include "ceed-xsmm.h"

//------------------------------------------------------------------------------
// Get Kernel Index
//------------------------------------------------------------------------------
int CeedGetXsmmInd_NonTensor(CeedInt add, CeedInt P, CeedInt Q, CeedInt B,
                             CeedInt C, CeedInt J) {
  return (C == 8 ? 1:0)*4*2 + (add ? 1:0)*4 +
         (B == P ? (J == Q ? 0:1) : (B == Q ? 2:3));
}

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

  return 0;
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
  CeedInt ind;
  CeedTensorContract_Xsmm *impl;
  ierr = CeedTensorContractGetData(contract, (void *)&impl); CeedChk(ierr);

  // Get kernel index
  if (impl->tensorbasis) {
    CeedInt key = ((B*impl->indScale + C)*impl->indScale + J)*10 +
                  2*(tmode == CEED_TRANSPOSE) + add;
    khint_t k = kh_get(m32, impl->lookup, key);
    ind = kh_value(impl->lookup, k);
  } else {
    ind = CeedGetXsmmInd_NonTensor(add, impl->P, impl->Q, B, C, J);
  }

  // Run kernel or fallback to default implementation
  if (C != 1)
    for (CeedInt a=0; a<A; a++)
      impl->kernels[ind](&u[a*B*C], &t[0], &v[a*J*C], NULL, NULL, NULL);
  else
    CeedTensorContract_Xsmm_C1(contract, A, B, C, J, t, tmode, add, u, v);

  return 0;
}

//------------------------------------------------------------------------------
// Tensor Contract Destroy
//------------------------------------------------------------------------------
static int CeedTensorContractDestroy_Xsmm(CeedTensorContract contract) {
  int ierr;
  CeedTensorContract_Xsmm *impl;
  ierr = CeedTensorContractGetData(contract, (void *)&impl); CeedChk(ierr);
  // Free kernels
  // Note: invalid reads in LIBXSMM when freeing from 0 -> numkernels - 1
  for (CeedInt i = impl->numkernels - 1; i >= 0; i--)
    if (impl->kernels[i])
      libxsmm_release_kernel(&impl->kernels[i]);
  kh_destroy(m32, impl->lookup);
  ierr = CeedFree(&impl->kernels); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Xsmm(CeedBasis basis,
                                  CeedTensorContract contract) {
  int ierr;
  Ceed ceed;
  ierr = CeedTensorContractGetCeed(contract, &ceed); CeedChk(ierr);
  CeedTensorContract_Xsmm *impl;
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);

  // Set up pointers to kernels
  ierr = CeedBasisGetTensorStatus(basis, &impl->tensorbasis); CeedChk(ierr);
  if (impl->tensorbasis) {
    ierr = CeedBasisGetNumNodes1D(basis, &impl->P); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &impl->Q); CeedChk(ierr);
    ierr = CeedBasisGetDimension(basis, &impl->dim); CeedChk(ierr);
    // Set up kernel pointer array
    impl->numkernels = 2*2*4*impl->dim;
    ierr = CeedCalloc(impl->numkernels, &impl->kernels); CeedChk(ierr);
    // Setup indices hash table
    CeedInt ind = 0;
    impl->indScale = 10;
    while ((impl->P > impl->Q ? impl->P : impl->Q)/impl->indScale > 0)
      impl->indScale *= 10;
    impl->lookup = kh_init(m32);
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
              // Add key, ind pair to hash table
              CeedInt key = ((B*impl->indScale + C)*impl->indScale + J)*10 +
                            2*(tmode) + add;
              khint_t k = kh_get(m32, impl->lookup, key);
              CeedInt keyMissing = (k == kh_end(impl->lookup));
              // Skip if duplicate key
              if (keyMissing) {
                CeedInt absent;
                k = kh_put(m32, impl->lookup, key, &absent);
                kh_value(impl->lookup, k) = ind;
                // Build kernel
                CeedScalar alpha = 1.0, beta = 1.0;
                if (!add) beta = 0.0;
                impl->kernels[ind] = libxsmm_dmmdispatch(C, J, B,
                                     NULL, NULL, NULL, &alpha,
                                     &beta, &flags, NULL);
                if (!impl->kernels[ind])
                  // LCOV_EXCL_START
                  return CeedError(ceed, 1, "LIBXSMM kernel failed to build.");
                // LCOV_EXCL_STOP
              }
              ind++;
            }
  } else {
    ierr = CeedBasisGetNumNodes(basis, &impl->P); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints(basis, &impl->Q); CeedChk(ierr);
    ierr = CeedBasisGetDimension(basis, &impl->dim); CeedChk(ierr);
    // Set up kernel pointer array
    impl->numkernels = 4*2*2;
    ierr = CeedCalloc(impl->numkernels, &impl->kernels); CeedChk(ierr);
    for (CeedInt nelem = 1; nelem <= 8; nelem+=7)
      for (CeedInt add = 0; add <= 1; add++)
        for (CeedInt tmode = 0; tmode <= 1; tmode++)
          for (CeedInt grad = 1; grad <= impl->dim; grad+=impl->dim-1) {
            const int flags = LIBXSMM_GEMM_FLAGS('N', tmode ? 'T' : 'N');
            CeedInt B = tmode ? grad*impl->Q : impl->P,
                    J = tmode ? impl->P : grad*impl->Q;
            int ind = CeedGetXsmmInd_NonTensor(add, impl->P, impl->Q, B, nelem,
                                               J);
            CeedScalar alpha = 1.0, beta = 1.0;
            if (!add) beta = 0.0;
            impl->kernels[ind] = libxsmm_dmmdispatch(nelem, J, B,
                                 NULL, NULL, NULL, &alpha,
                                 &beta, &flags, NULL);
            if (!impl->kernels[ind])
              // LCOV_EXCL_START
              return CeedError(ceed, 1, "LIBXSMM kernel failed to build.");
            // LCOV_EXCL_STOP
          }
  }
  ierr = CeedTensorContractSetData(contract, (void *)&impl); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply",
                                CeedTensorContractApply_Xsmm); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy",
                                CeedTensorContractDestroy_Xsmm); CeedChk(ierr);

  return 0;
}
//------------------------------------------------------------------------------
