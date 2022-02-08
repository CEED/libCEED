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
#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Ref(CeedTensorContract contract, CeedInt A,
                                       CeedInt B, CeedInt C, CeedInt J,
                                       const CeedScalar *restrict t,
                                       CeedTransposeMode t_mode, const CeedInt add,
                                       const CeedScalar *restrict u,
                                       CeedScalar *restrict v) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;
  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1; t_stride_1 = J;
  }

  if (!add)
    for (CeedInt q=0; q<A*J*C; q++)
      v[q] = (CeedScalar) 0.0;

  for (CeedInt a=0; a<A; a++)
    for (CeedInt b=0; b<B; b++)
      for (CeedInt j=0; j<J; j++) {
        CeedScalar tq = t[j*t_stride_0 + b*t_stride_1];
        for (CeedInt c=0; c<C; c++)
          v[(a*J+j)*C+c] += tq * u[(a*B+b)*C+c];
      }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Destroy
//------------------------------------------------------------------------------
static int CeedTensorContractDestroy_Ref(CeedTensorContract contract) {
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Ref(CeedBasis basis, CeedTensorContract contract) {
  int ierr;
  Ceed ceed;
  ierr = CeedTensorContractGetCeed(contract, &ceed); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply",
                                CeedTensorContractApply_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy",
                                CeedTensorContractDestroy_Ref); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
