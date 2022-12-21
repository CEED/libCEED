// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>
#include <string>

#include <iostream>
#include <sstream>

#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref.hpp"

//------------------------------------------------------------------------------
// Build QFunction kernel
//------------------------------------------------------------------------------
extern "C" int CeedSyclBuildQFunction(CeedQFunction qf) {
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed SYCL function not implemented");
}
//------------------------------------------------------------------------------
