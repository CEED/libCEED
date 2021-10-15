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
/// Linear elasticity manufactured solution forcing term for solid mechanics example using PETSc

#ifndef MANUFACTURED_H
#define MANUFACTURED_H

#if !CEED_QFUNCTION_JIT
#  include <math.h>
#endif

#ifndef PHYSICS_STRUCT
#define PHYSICS_STRUCT
typedef struct Physics_private *Physics;
struct Physics_private {
  CeedScalar   nu;      // Poisson's ratio
  CeedScalar   E;       // Young's Modulus
};
#endif

// -----------------------------------------------------------------------------
// Forcing term for linear elasticity manufactured solution
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMMSForce)(void *ctx, const CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // Inputs
  const CeedScalar *coords = in[0], *q_data = in[1];

  // Outputs
  CeedScalar *force = out[0];

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q], z = coords[i+2*Q];
    CeedScalar wdetJ = q_data[i];

    // Forcing function
    // -- Component 1
    force[i+0*Q] = (-(E*(cos(x*2.0)*cos(y*3.0)*exp(z*4.0)*4.0 -
                         cos(z*4.0)*sin( y*3.0)*exp(x*2.0)*8.0)*(nu-0.5))/
                    ((nu*2.0-1.0)*(nu+1.0)) +
                    (E*(cos(z*4.0)*sin(y*3.0)*exp(x*2.0)*(4.5) +
                        sin(x*2.0)*sin(z*4.0)*exp( y*3.0)*3.0)*(nu-0.5))/
                    ((nu*2.0-1.0)*(nu+1.0)) +
                    (E*nu*cos(x*2.0)*cos(y*3.0)*exp(z*4.0)*8.0)/
                    ((nu*2.0-1.0)*(nu+1.0)) -
                    (E*nu*sin(x*2.0)*sin(z*4.0)*exp(y*3.0)*6.0)/
                    ((nu*2.0-1.0)*(nu+1.0)) -
                    (E*cos(z*4.0)*sin(y*3.0)*exp(x*2.0)*(nu-1.0)*4.0)/
                    ((nu*2.0-1.0)*(nu+1.0))) * wdetJ / 1e8;

    // -- Component 2
    force[i+1*Q] = (-(E*(cos(y*3.0)*cos(z*4.0)*exp(x*2.0)*3.0 -
                         cos(x*2.0)*sin( z*4.0)*exp(y*3.0)*2.0)*(nu-0.5))/
                    ((nu*2.0-1.0)*(nu+1.0)) +
                    (E*(cos(x*2.0)*sin(z*4.0)*exp(y*3.0)*8.0 +
                        sin(x*2.0)*sin(y*3.0)*exp(z*4.0)*6.0)*(nu-0.5))/
                    ((nu*2.0-1.0)*(nu+1.0)) +
                    (E*nu*cos(y*3.0)*cos(z*4.0)*exp(x*2.0)*6.0)/
                    ((nu*2.0-1.0)*(nu+1.0)) -
                    (E*nu*sin( x*2.0)*sin(y*3.0)*exp(z*4.0)*12.0)/
                    ((nu*2.0-1.0)*(nu+1.0)) -
                    (E*cos(x*2.0)*sin(z*4.0)*exp(y*3.0)*(nu-1.0)*9.0)/
                    ((nu*2.0-1.0)*(nu+1.0))) * wdetJ / 1e8;

    // -- Component 3
    force[i+2*Q] = (-(E*(cos(x*2.0)*cos(z*4.0)*exp(y*3.0)*6.0 -
                         cos(y*3.0)*sin( x*2.0)*exp(z*4.0)*(4.5))*(nu-0.5))/
                    ((nu*2.0-1.0)*(nu+1.0)) +
                    (E*(cos(y*3.0)*sin(x*2.0)*exp(z*4.0)*2.0 +
                        sin(y*3.0)*sin(z*4.0)*exp(x*2.0)*4.0)*(nu-0.5))/
                    ((nu*2.0-1.0)*(nu+1.0)) +
                    (E*nu*cos(x*2.0)*cos(z*4.0)*exp(y*3.0)*12.0)/
                    ((nu*2.0-1.0)*(nu+1.0)) -
                    (E*nu*sin( y*3.0)*sin(z*4.0)*exp(x*2.0)*8.0)/
                    ((nu*2.0-1.0)*(nu+1.0)) -
                    (E*cos(y*3.0)*sin(x*2.0)*exp(z*4.0)*(nu-1.0)*16.0)/
                    ((nu*2.0-1.0)*(nu+1.0))) * wdetJ / 1e8;

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End MANUFACTURED_H
