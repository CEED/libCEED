// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef utils_h
#define utils_h

#include <ceed.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CEED_QFUNCTION_HELPER CeedScalar Max(CeedScalar a, CeedScalar b) { return a < b ? b : a; }
CEED_QFUNCTION_HELPER CeedScalar Min(CeedScalar a, CeedScalar b) { return a < b ? a : b; }

CEED_QFUNCTION_HELPER CeedScalar Square(CeedScalar x) { return x * x; }
CEED_QFUNCTION_HELPER CeedScalar Cube(CeedScalar x) { return x * x * x; }

// @brief Dot product of 3 element vectors
CEED_QFUNCTION_HELPER CeedScalar Dot3(const CeedScalar u[3], const CeedScalar v[3]) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

// @brief Unpack Kelvin-Mandel notation symmetric tensor into full tensor
CEED_QFUNCTION_HELPER void KMUnpack(const CeedScalar v[6], CeedScalar A[3][3]) {
  const CeedScalar weight = 1 / sqrt(2.);
  A[0][0]                 = v[0];
  A[1][1]                 = v[1];
  A[2][2]                 = v[2];
  A[2][1] = A[1][2] = weight * v[3];
  A[2][0] = A[0][2] = weight * v[4];
  A[1][0] = A[0][1] = weight * v[5];
}

#endif  // utils_h
