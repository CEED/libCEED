// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// This header does not have guards because it may be included multiple times.

// List each backend registration function in the corresponding `ceed-backend-list-*.h` file, grouped by install requirement.
// Include each of those files here.
// This will be expanded inside CeedRegisterAll() to call each registration function in the order listed.

// Always compiled
#include "ceed-backend-list-ref.h"
// Requires AVX support
#include "ceed-backend-list-avx.h"
// Requires Valgrind
#include "ceed-backend-list-memcheck.h"
// Requires LIBXSMM
#include "ceed-backend-list-xsmm.h"
// Requires CUDA
#include "ceed-backend-list-cuda.h"
// Requires ROCm
#include "ceed-backend-list-hip.h"
// Requires SYCL
#include "ceed-backend-list-sycl.h"
// Requires MAGMA + (CUDA or ROCm)
#include "ceed-backend-list-magma.h"
