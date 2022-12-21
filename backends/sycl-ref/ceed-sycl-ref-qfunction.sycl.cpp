// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <string>

#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref-qfunction-load.hpp"
#include "ceed-sycl-ref.hpp"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Sycl(CeedQFunction qf, CeedInt Q, CeedVector *U,
                                   CeedVector *V) {
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Sycl(CeedQFunction qf) {
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Set User QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionSetSYCLUserFunction_Sycl(CeedQFunction qf/*, CUfunction f*/) {
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Sycl(CeedQFunction qf) {
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}
//------------------------------------------------------------------------------
