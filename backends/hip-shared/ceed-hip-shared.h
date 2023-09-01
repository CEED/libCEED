// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_HIP_SHARED_H
#define CEED_HIP_SHARED_H

#include <ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>

typedef struct {
  hipModule_t   module;
  hipFunction_t Interp;
  hipFunction_t InterpTranspose;
  hipFunction_t Grad;
  hipFunction_t GradTranspose;
  hipFunction_t Weight;
  CeedInt       block_sizes[3];  // interp, grad, weight thread block sizes
  CeedScalar   *d_interp_1d;
  CeedScalar   *d_grad_1d;
  CeedScalar   *d_collo_grad_1d;
  CeedScalar   *d_q_weight_1d;
} CeedBasis_Hip_shared;

CEED_INTERN int CeedBasisCreateTensorH1_Hip_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                                   const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);

#endif  // CEED_HIP_SHARED_H
