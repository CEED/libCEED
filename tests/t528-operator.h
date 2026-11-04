// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
// SPDX-License-Identifier: BSD-2-Clause
#include <ceed/types.h>

CEED_QFUNCTION(SliceCoefficients)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedInt components = *(const CeedInt *)ctx;

  for (CeedInt i = 0; i < Q; i++) {
    for (CeedInt direction = 0; direction < 3; direction++) {
      for (CeedInt component = 0; component < components; component++) {
        const CeedInt offset = (direction * components + component) * Q + i;
        out[0][offset]       = in[0][offset] * (in[1][i] + in[2][direction * Q + i] + in[3][i]);
      }
    }
    out[1][i] = in[0][i] * in[1][i];
  }
  return 0;
}
