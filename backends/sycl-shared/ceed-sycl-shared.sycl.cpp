// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-shared.hpp"

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <string>

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Sycl_shared(const char *resource, Ceed ceed) {
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
  // return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Sycl_Shared(void) { return CeedRegister("/gpu/sycl/shared", CeedInit_Sycl_shared, 25); }
//------------------------------------------------------------------------------
