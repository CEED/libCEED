// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-ref.hpp"

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <string>

#include <sycl/sycl.hpp>

//------------------------------------------------------------------------------
// SYCL preferred MemType
//------------------------------------------------------------------------------
static int CeedGetPreferredMemType_Sycl(CeedMemType *mem_type) {
  return CeedError(NULL, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Get CUBLAS handle
//------------------------------------------------------------------------------
// int CeedSyclGetCublasHandle(Ceed ceed, cublasHandle_t *handle) {
//   return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
// }

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Sycl(const char *resource, Ceed ceed) {
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
  // return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Sycl(void) {
  return CeedRegister("/gpu/sycl/ref", CeedInit_Sycl, 40);
}
//------------------------------------------------------------------------------
