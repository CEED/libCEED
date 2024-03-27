// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#include <ceed.h>

// @brief Calculate the inverse of the multiplicity, reducing to a single component
CEED_QFUNCTION(InverseMultiplicity)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*multiplicity)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  CeedScalar(*inv_multiplicity)               = (CeedScalar(*))out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) inv_multiplicity[i] = 1.0 / multiplicity[0][i];
  return 0;
}
