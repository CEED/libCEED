// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_cuda_shared_h
#define _ceed_cuda_shared_h

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <cuda.h>

#include "../cuda/ceed-cuda-common.h"

typedef struct {
  CUmodule    module;
  CUfunction  Interp;
  CUfunction  InterpTranspose;
  CUfunction  Grad;
  CUfunction  GradTranspose;
  CUfunction  Weight;
  CeedScalar *d_interp_1d;
  CeedScalar *d_grad_1d;
  CeedScalar *d_collo_grad_1d;
  CeedScalar *d_q_weight_1d;
  CeedScalar *c_B;
  CeedScalar *c_G;
} CeedBasis_Cuda_shared;

CEED_INTERN int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                                    const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);

#endif  // _ceed_cuda_shared_h
