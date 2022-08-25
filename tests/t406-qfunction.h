// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// Note: intentionally testing strange spacing in '#include's
#include <ceed.h>
#include <math.h>

#include "./t406-qfunction-scales.h"
#include "t406-qfunction-helper.h"

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *w      = in[0];
  CeedScalar       *q_data = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    q_data[i] = w[i];
  }
  return 0;
}

CEED_QFUNCTION(mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *q_data = in[0], *u = in[1];
  CeedScalar       *v = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    v[i] = q_data[i] * (times_two(u[i]) + times_three(u[i])) * sqrt(1.0 * SCALE_TWO);
  }
  return 0;
}
