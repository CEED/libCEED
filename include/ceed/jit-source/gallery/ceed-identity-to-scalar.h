// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/**
  @brief  Identity QFunction that copies first input component directly into output
**/
#include <ceed/types.h>

CEED_QFUNCTION(IdentityScalar)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is input, size (Q*size)
  const CeedScalar *input = in[0];
  // out[0] is output, size (Q)
  CeedScalar *output = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { output[i] = input[i]; }  // End of Quadrature Point Loop
  return CEED_ERROR_SUCCESS;
}
