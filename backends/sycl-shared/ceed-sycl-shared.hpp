// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_sycl_shared_hpp
#define _ceed_sycl_shared_hpp

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-common.hpp"

typedef struct {
  // CUmodule    module;
  // CUfunction  Interp;
  // CUfunction  InterpTranspose;
  // CUfunction  Grad;
  // CUfunction  GradTranspose;
  // CUfunction  Weight;
  CeedScalar *d_interp_1d;
  CeedScalar *d_grad_1d;
  CeedScalar *d_collo_grad_1d;
  CeedScalar *d_q_weight_1d;
  CeedScalar *c_B;
  CeedScalar *c_G;
} CeedBasis_Sycl_shared;

CEED_INTERN int CeedBasisCreateTensorH1_Sycl_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                                    const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);

#endif  // _ceed_sycl_shared_h
