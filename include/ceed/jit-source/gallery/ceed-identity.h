// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/**
  @brief  Identity QFunction that copies inputs directly into outputs
**/

#ifndef CEED_IDENTITY_H
#define CEED_IDENTITY_H

#include <ceed.h>

typedef struct {
  CeedInt size;
} IdentityCtx;

CEED_QFUNCTION(Identity)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Ctx holds field size
  IdentityCtx   identity_ctx = *(IdentityCtx *)ctx;
  const CeedInt size         = identity_ctx.size;

  // in[0] is input, size (Q*size)
  const CeedScalar *input = in[0];
  // out[0] is output, size (Q*size)
  CeedScalar *output = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q * size; i++) { output[i] = input[i]; }  // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif  // CEED_IDENTITY_H
