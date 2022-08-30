// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef dirichlet_boundary_h
#define dirichlet_boundary_h

#include <ceed.h>

CEED_QFUNCTION(SetupDirichletBC)(void *ctx, CeedInt Q,
                                 const CeedScalar *const *in,
                                 CeedScalar *const *out) {
  // Inputs
  // *INDENT-OFF*
  typedef CeedScalar vec_t[CEED_Q_VLA];
  const vec_t* coords = (const vec_t*) in[0];
  const vec_t* multiplicity = (const vec_t*) in[1];
  // *INDENT-ON*

  // Outputs
  vec_t* coords_stored = (vec_t*) out[0];
  CeedScalar * const scale_stored = out[1];

  CeedPragmaSIMD
  for(CeedInt i=0; i<Q; i++) {
    for (int j=0; j<3; j++) coords_stored[j][i] = coords[j][i];
    scale_stored[i] = 1.0 / multiplicity[0][i];
  }
  return 0;
}

#endif // dirichlet_boundary_h
