// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#include <ceed/types.h>

typedef struct {
  CeedInt    size, calls, observe, use_extra;
  CeedScalar scale;
} PipelineContext;

CEED_QFUNCTION(ScaleComponents)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  PipelineContext *context = (PipelineContext *)ctx;
  const CeedInt    size    = context->size;
  if (context->observe) {
#if defined(__CUDA_ARCH__)
    atomicAdd(&context->calls, Q);
#else
    context->calls += Q;
#endif
  }
  for (CeedInt j = 0; j < size; j++) {
    for (CeedInt i = 0; i < Q; i++) out[0][j * Q + i] = context->scale * (1 + 0.01 * j) * in[0][j * Q + i] + (context->use_extra ? in[1][i] : 0);
  }
  return 0;
}
