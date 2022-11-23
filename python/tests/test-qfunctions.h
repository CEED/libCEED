// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//------------------------------------------------------------------------------
// Setup 1D mass matrix
//------------------------------------------------------------------------------
CEED_QFUNCTION(setup_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is quadrature weights, size (Q)
  // in[1] is Jacobians, size (Q)
  const CeedScalar *w = in[0], *J = in[1];

  // out[0] is quadrature data, size (Q)
  CeedScalar *qdata = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { qdata[i] = J[i] * w[i]; }

  return 0;
}

//------------------------------------------------------------------------------
// Setup 2D mass matrix
//------------------------------------------------------------------------------
CEED_QFUNCTION(setup_mass_2d)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is quadrature weights, size (Q)
  // in[1] is Jacobians with shape [2, nc=2, Q]
  const CeedScalar *w = in[0], *J = in[1];

  // out[0] is quadrature data, size (Q)
  CeedScalar *qdata = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { qdata[i] = w[i] * (J[i + Q * 0] * J[i + Q * 3] - J[i + Q * 1] * J[i + Q * 2]); }

  return 0;
}

//------------------------------------------------------------------------------
// Apply mass matrix
//------------------------------------------------------------------------------
CEED_QFUNCTION(apply_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Get scaling factor, if set
  const CeedScalar *scale_array = ctx ? (CeedScalar *)ctx : NULL;
  const CeedScalar  scale       = ctx ? scale_array[4] : 1.;

  // in[0] is quadrature data, size (Q)
  // in[1] is u, size (Q)
  const CeedScalar *qdata = in[0], *u = in[1];

  // out[0] is v, size (Q)
  CeedScalar *v = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { v[i] = scale * qdata[i] * u[i]; }

  return 0;
}

//------------------------------------------------------------------------------
// Apply mass matrix to two components
//------------------------------------------------------------------------------
CEED_QFUNCTION(apply_mass_two)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is quadrature data, size (Q)
  // in[1] is u, size (2*Q)
  const CeedScalar *qdata = in[0], *u = in[1];

  // out[0] is v, size (2*Q)
  CeedScalar *v = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    v[i]     = qdata[i] * u[i];
    v[Q + i] = qdata[i] * u[Q + i];
  }

  return 0;
}

//------------------------------------------------------------------------------
