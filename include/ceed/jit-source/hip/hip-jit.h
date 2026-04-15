// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for HIP backend macro and type definitions for JiT source

#define CEED_QFUNCTION(name) inline __device__ int name
#define CEED_QFUNCTION_HELPER inline __device__
#define CeedPragmaSIMD
#define CEED_Q_VLA 1

// If we are using Chipstar, then we have to ensure all threads have the same workloads
//   and hit __syncthreads() at the same places/number of times
#ifdef __HIP_PLATFORM_SPIRV__
#define CEED_HIP_USE_CHIPSTAR true
#else
#define CEED_HIP_USE_CHIPSTAR false
#endif

#include "hip-types.h"
