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
CEED_BACKEND(CeedRegister_Cuda, 1, "/gpu/cuda/ref")
CEED_BACKEND(CeedRegister_Cuda_Gen, 1, "/gpu/cuda/gen")
CEED_BACKEND(CeedRegister_Cuda_Shared, 1, "/gpu/cuda/shared")
// LCOV_EXCL_STOP
#undef CEED_BACKEND
