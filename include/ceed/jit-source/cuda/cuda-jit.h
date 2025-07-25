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

#define CEED_QFUNCTION_RUST(name)                                                                                       \
  extern "C" __device__ int name##_rs(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out); \
  static __device__ int name(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return name##_rs(ctx, Q, in, out); }

#include "cuda-types.h"
