// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

CEED_QFUNCTION(scale)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  CeedScalar       *scale = (CeedScalar *)ctx;
  const CeedScalar *u     = in[0];
  CeedScalar       *v     = out[0];

  for (CeedInt i = 0; i < Q; i++) {
    v[i] = scale[1] * u[i];
  }
  scale[0] = 42;

  return 0;
}
