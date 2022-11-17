// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef dirichlet_boundary_h
#define dirichlet_boundary_h

#include <ceed.h>

CEED_QFUNCTION(SetupDirichletBC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  // *INDENT-OFF*
  const CeedScalar(*coords)[CEED_Q_VLA]       = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*multiplicity)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // *INDENT-ON*

  // Outputs
  CeedScalar(*coords_stored)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*scale_stored)              = (CeedScalar(*))out[1];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    for (int j = 0; j < 3; j++) coords_stored[j][i] = coords[j][i];
    scale_stored[i] = 1.0 / multiplicity[0][i];
  }
  return 0;
}

#endif  // dirichlet_boundary_h
