// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA backend macro and type definitions for JiT source
#ifndef _ceed_cuda_jit_defs_h
#define _ceed_cuda_jit_defs_h

#define CEED_QFUNCTION(name) inline __device__ int name
#define CEED_QFUNCTION_HELPER inline __device__
#define CeedPragmaSIMD
#define CEED_Q_VLA 1

#include <ceed/types.h>

typedef struct {
  const CeedScalar* inputs[16];
  CeedScalar*       outputs[16];
} Fields_Cuda;

#endif
