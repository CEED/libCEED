// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/types.h>

CEED_QFUNCTION(ones_1)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  CeedScalar *v = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    v[i] = 1.0;
  }
  return 0;
}

CEED_QFUNCTION(ones_2)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  CeedScalar *v = out[0];
  for (CeedInt i = 0; i < 2 * Q; i++) {
    v[i] = 1.0;
  }
  return 0;
}
