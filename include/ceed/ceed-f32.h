/// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
/// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
///
/// SPDX-License-Identifier: BSD-2-Clause
///
/// This file is part of CEED:  http://github.com/ceed

/// @file
/// Public header for definitions related to using FP32 floating point (single precision) for CeedScalar.
/// Include this header in ceed.h to use float instead of double.
#pragma once

#ifndef CEED_RUNNING_JIT_PASS
#include <float.h>
#endif

#define CEED_SCALAR_IS_FP32

/// Set base scalar type to FP32. (See CeedScalarType enum in ceed.h for all options.)
#define CEED_SCALAR_TYPE CEED_SCALAR_FP32
#if defined(CEED_RUNNING_JIT_PASS) && defined(CEED_JIT_PRECISION) && (CEED_JIT_PRECISION != CEED_SCALAR_TYPE)
#ifdef CEED_JIT_PRECISION == CEED_SCALAR_FP64
typedef double CeedScalar;
typedef float  CeedScalarBase;

/// Machine epsilon
static const CeedScalar CEED_EPSILON = DBL_EPSILON;
#endif  // CEED_JIT_PRECISION
#else
typedef float      CeedScalar;
typedef CeedScalar CeedScalarBase;

/// Machine epsilon
static const CeedScalar CEED_EPSILON = FLT_EPSILON;
#endif
