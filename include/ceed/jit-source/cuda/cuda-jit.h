// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA backend macro and type definitions for JiT source

#define CEED_QFUNCTION(name) inline __device__ int name
#define CEED_QFUNCTION_HELPER inline __device__
#define CeedPragmaSIMD
#define CEED_Q_VLA 1

#ifndef DBL_EPSILON
#define DBL_EPSILON 2.22044604925031308084726333618164062e-16
#endif
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209289550781250000000000000000000e-7F
#endif

#include "cuda-types.h"
