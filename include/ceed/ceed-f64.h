/// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
/// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
///
/// SPDX-License-Identifier: BSD-2-Clause
///
/// This file is part of CEED:  http://github.com/ceed

/// @file
/// Public header for definitions related to using FP64 floating point (double precision) for CeedScalar.
/// This is the default header included in ceed.h.
#pragma once

#define CEED_SCALAR_IS_FP64

/// Set base scalar type to FP64. (See CeedScalarType enum in ceed.h for all options.)
#define CEED_SCALAR_TYPE CEED_SCALAR_FP64
#if defined(CEED_RUNNING_JIT_PASS) && defined(CEED_JIT_MIXED_PRECISION)
typedef float  CeedScalar;
typedef double CeedScalarCPU;

/// Machine epsilon
#define CEED_EPSILON 0x1p-23
#else
typedef double     CeedScalar;
typedef CeedScalar CeedScalarCPU;

/// Machine epsilon
#define CEED_EPSILON 0x1p-52
#endif  // CEED_RUNNING_JIT_PASS && CEED_JIT_MIXED_PRECISION
