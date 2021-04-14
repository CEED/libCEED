// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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

/// @file
/// libCEED QFunctions for mass operator example using PETSc

#ifndef bp1_h
#define bp1_h

#ifndef __CUDACC__
#  include <math.h>
#endif

// -----------------------------------------------------------------------------
// This QFunction sets up the geometric factors required to apply the
//   mass operator
//
// The quadrature data is stored in the array q_data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( u v )
//
// Qdata: det_J * w
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassGeo)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  const CeedScalar *J = in[1], *w = in[2]; // Note: *X = in[0]
  CeedScalar *q_data = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar det_J = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                             J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                             J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6]));
    q_data[i] = det_J * w[i];
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassRhs)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  const CeedScalar *x = in[0], *w = in[1];
  CeedScalar *true_soln = out[0], *rhs = out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    true_soln[i] = sqrt(x[i]*x[i] + x[i+Q]*x[i+Q] + x[i+2*Q]*x[i+2*Q]);
    rhs[i] = w[i] * true_soln[i];
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction applies the mass operator for a scalar field.
//
// Inputs:
//   u     - Input vector at quadrature points
//   q_data - Geometric factors
//
// Output:
//   v     - Output vector (test functions) at quadrature points
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(Mass)(void *ctx, const CeedInt Q,
                     const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *q_data = in[1];
  CeedScalar *v = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++)
    v[i] = q_data[i] * u[i];

  return 0;
}
// -----------------------------------------------------------------------------

#endif // bp1_h
