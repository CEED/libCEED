// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-backend-weak.h"

#define CEED_BACKEND(name, num_prefixes, ...) \
  CEED_INTERN int __attribute__((weak)) name(void) { return CeedRegister_Weak(__func__, num_prefixes, __VA_ARGS__); }
// LCOV_EXCL_START
CEED_BACKEND(CeedRegister_Sycl, 1, "/gpu/sycl/ref")
CEED_BACKEND(CeedRegister_Sycl_Shared, 1, "/gpu/sycl/shared")
CEED_BACKEND(CeedRegister_Sycl_Gen, 1, "/gpu/sycl/gen")
// LCOV_EXCL_STOP
#undef CEED_BACKEND
