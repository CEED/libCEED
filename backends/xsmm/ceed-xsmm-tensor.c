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

#include <string.h>
#include <libxsmm.h>
#include "ceed-xsmm.h"

// Blocked Tensor Contract
static int CeedTensorContract_Xsmm_Blocked(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
    CeedTransposeMode tmode, const CeedInt Add, const CeedScalar *restrict u,
    CeedScalar *restrict v) {
  CeedScalar alpha = 1.0, beta = 1.0;
  char transu = 'N', transt = 'N';
  if (tmode == CEED_TRANSPOSE)
    transt = 'T';

  if (!Add)
    beta = 0.0;

  for (CeedInt a=0; a<A; a++)
    // libXSMM GEMM
    libxsmm_dgemm(&transu, &transt, &C, &J, &B,
                  &alpha, &u[a*B*C], NULL, &t[0], NULL,
                  &beta, &v[a*J*C], NULL);
  return 0;
}

// Serial Tensor Contact
static int CeedTensorContract_Xsmm_Serial(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
    CeedTransposeMode tmode, const CeedInt Add, const CeedScalar *restrict u,
    CeedScalar *restrict v) {
  CeedScalar alpha = 1.0, beta = 1.0;
  char transu = 'N', transt = 'N';
  if ((tmode == CEED_TRANSPOSE && C != 1)
      || (tmode == CEED_NOTRANSPOSE && C == 1))
    transt = 'T';

  if (!Add)
    beta = 0.0;

  if (C != 1)
    for (CeedInt a=0; a<A; a++)
      // libXSMM GEMM
      libxsmm_dgemm(&transu, &transt, &C, &J, &B,
                    &alpha, &u[a*B*C], NULL, &t[0], NULL,
                    &beta, &v[a*J*C], NULL);
  else
    // libXSMM GEMM
    libxsmm_dgemm(&transt, &transu, &J, &A, &B,
                  &alpha, &t[0], NULL, &u[0], NULL,
                  &beta, &v[0], NULL);

  return 0;
}

// Switch for Tensor Contract
static int CeedTensorContractApply_Xsmm(CeedTensorContract contract, CeedInt A,
                                        CeedInt B, CeedInt C, CeedInt J,
                                        const CeedScalar *restrict t,
                                        CeedTransposeMode tmode,
                                        const CeedInt Add,
                                        const CeedScalar *restrict u,
                                        CeedScalar *restrict v) {
  CeedInt blksize = 8;

  if (C % blksize)
    CeedTensorContract_Xsmm_Serial(contract, A, B, C, J, t, tmode, Add, u, v);
  else
    CeedTensorContract_Xsmm_Blocked(contract, A, B, C, J, t, tmode, Add, u, v);

  return 0;
}

static int CeedTensorContractDestroy_Xsmm(CeedTensorContract contract) {
  return 0;
}

int CeedTensorContractCreate_Xsmm(CeedTensorContract contract) {
  int ierr;
  Ceed ceed;
  ierr = CeedTensorContractGetCeed(contract, &ceed); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply",
                                CeedTensorContractApply_Xsmm); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy",
                                CeedTensorContractDestroy_Xsmm); CeedChk(ierr);

  return 0;
}
