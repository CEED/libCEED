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
#include "../sycl/ceed-sycl-compile.hpp"

typedef struct {
  CeedInt interp_local_range[3];
  CeedInt grad_local_range[3];
  CeedInt weight_local_range[3];
  SyclModule_t *sycl_module;
  sycl::kernel *interp_kernel;
  sycl::kernel *interp_transpose_kernel;
  sycl::kernel *grad_kernel;
  sycl::kernel *grad_transpose_kernel;
  sycl::kernel *weight_kernel;
  CeedScalar *d_interp_1d;
  CeedScalar *d_grad_1d;
  CeedScalar *d_collo_grad_1d; //eliminate 
  CeedScalar *d_q_weight_1d;
} CeedBasis_Sycl_shared;

CEED_INTERN int CeedBasisCreateTensorH1_Sycl_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                                    const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);

#endif  // _ceed_sycl_shared_h
