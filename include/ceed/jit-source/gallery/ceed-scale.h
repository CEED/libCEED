// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/**
  @brief  Scaling QFunction that scales inputs
**/

#ifndef scale_h
#define scale_h

#include <ceed.h>

CEED_QFUNCTION(Scale)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Ctx holds field size
  const CeedInt size = *(CeedInt *)ctx;

  // in[0] is input, size (Q*size)
  // in[1] is scaling factor, size (Q*size)
  const CeedScalar *input = in[0];
  const CeedScalar *scale = in[1];
  // out[0] is output, size (Q*size)
  CeedScalar *output = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q * size; i++) { output[i] = input[i] * scale[i]; }  // End of Quadrature Point Loop
  return 0;
}

#endif  // scale_h
