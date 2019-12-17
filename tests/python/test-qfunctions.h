// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

//------------------------------------------------------------------------------
// Setup 1D mass matrix
//------------------------------------------------------------------------------
CEED_QFUNCTION(setup_mass)(void *ctx, const CeedInt Q,
                           const CeedScalar *const *in,
                           CeedScalar *const *out) {
  // in[0] is quadrature weights, size (Q)
  // in[1] is Jacobians, size (Q)
  const CeedScalar *w = in[0], *J = in[1];

  // out[0] is quadrature data, size (Q)
  CeedScalar *qdata = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    qdata[i] = J[i] * w[i];
  }

  return 0;
}

//------------------------------------------------------------------------------
// Setup 2D mass matrix
//------------------------------------------------------------------------------
CEED_QFUNCTION(setup_mass_2d)(void *ctx, const CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // in[0] is quadrature weights, size (Q)
  // in[1] is Jacobians with shape [2, nc=2, Q]
  const CeedScalar *w = in[0], *J = in[1];

  // out[0] is quadrature data, size (Q)
  CeedScalar *qdata = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    qdata[i] = w[i] * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]);
  }

  return 0;
}

//------------------------------------------------------------------------------
// Apply mass matrix
//------------------------------------------------------------------------------
CEED_QFUNCTION(apply_mass)(void *ctx, const CeedInt Q,
                           const CeedScalar *const *in,
                           CeedScalar *const *out) {
  // Get scaling factor, if set
  const CeedScalar *scale_array = ctx ? (CeedScalar *)ctx : NULL;
  const CeedScalar scale = ctx ? scale_array[4] : 1.;

  // in[0] is quadrature data, size (Q)
  // in[1] is u, size (Q)
  const CeedScalar *qdata = in[0], *u = in[1];

  // out[0] is v, size (Q)
  CeedScalar *v = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    v[i] = scale * qdata[i] * u[i];
  }

  return 0;
}

//------------------------------------------------------------------------------
// Apply mass matrix to two components
//------------------------------------------------------------------------------
CEED_QFUNCTION(apply_mass_two)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // in[0] is quadrature data, size (Q)
  // in[1] is u, size (2*Q)
  const CeedScalar *qdata = in[0], *u = in[1];

  // out[0] is v, size (2*Q)
  CeedScalar *v = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    v[i]   = qdata[i] * u[i];
    v[Q+i] = qdata[i] * u[Q+i];
  }

  return 0;
}

//------------------------------------------------------------------------------
