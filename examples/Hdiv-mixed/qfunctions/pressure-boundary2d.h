// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Pressure boundary conditions 2D

#ifndef pressure_bc_2d_h
#define pressure_bc_2d_h

// -----------------------------------------------------------------------------
// Strong form:
//  u       = -\grad(p)      on \Omega
//  \div(u) = f              on \Omega
//  p = p0                   on \Gamma_D
//  u.n = g                  on \Gamma_N
// Weak form: Find (u,p) \in VxQ (V=H(div), Q=L^2) on \Omega
//  (v, u) - (\div(v), p) = -<v, p0 n>_{\Gamma_D}
// -(q, \div(u))          = -(q, f)
// This QFunction sets up the pressure boundary conditions : -<v, p0 n>_{\Gamma_D}
// Inputs:
//   w     : weight of quadrature
//   p0    : pressure value on the boundary
//
// Output:
//   v     : p0 * N * w
// Note that the Piola map of the H(div) basis and physical normal "n" got canceled
// and we need to multiply by the reference normal "N" on each face
// -----------------------------------------------------------------------------
CEED_QFUNCTION(BCPressure2D)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // User context
  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar p0 = 10.;
    for (CeedInt k = 0; k < 2; k++) {
      v[k][i] += p0 * w[i];
    }
  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************

#endif // pressure_bc_2d_h
