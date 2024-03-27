// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA backend common definitions
#ifndef CEED_MAGMA_COMMON_DEFS_H
#define CEED_MAGMA_COMMON_DEFS_H

#define MAGMA_DEVICE_SHARED(type, name) extern __shared__ type name[];

#define MAGMA_MAXTHREADS_1D 128
#define MAGMA_MAXTHREADS_2D 128
#define MAGMA_MAXTHREADS_3D 64

// Define macro for determining number of threads in y-direction for basis kernels
#define MAGMA_BASIS_NTCOL(x, maxt) (((maxt) < (x)) ? 1 : ((maxt) / (x)))

// Define macro for computing the total threads in a block for use with __launch_bounds__()
#define MAGMA_BASIS_BOUNDS(x, maxt) (x * MAGMA_BASIS_NTCOL(x, maxt))

#endif  // CEED_MAGMA_COMMON_DEFS_H
