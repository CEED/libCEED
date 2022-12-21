// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-common.hpp"

#include <string>

//------------------------------------------------------------------------------
// Get root resource without device spec
//------------------------------------------------------------------------------
int CeedSyclGetResourceRoot(Ceed ceed, const char *resource, char **resource_root) {
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedSyclInit(Ceed ceed, const char *resource) {
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
// Backend destroy
//------------------------------------------------------------------------------
int CeedDestroy_Sycl(Ceed ceed) {
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}

//------------------------------------------------------------------------------
