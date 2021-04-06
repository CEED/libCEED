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
/// libCEED QFunctions for mass operator example for a vector field on the sphere using PETSc

#ifndef bp4sphere_h
#define bp4sphere_h

#ifndef __CUDACC__
#  include <math.h>
#endif

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffRhs3)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  // Inputs
  const CeedScalar *X = in[0], *qdata = in[1];
  // Outputs
  CeedScalar *true_soln = out[0], *rhs = out[1];

  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar R        = context[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read global Cartesian coordinates
    CeedScalar x = X[i+Q*0], y = X[i+Q*1], z = X[i+Q*2];
    // Normalize quadrature point coordinates to sphere
    CeedScalar rad = sqrt(x*x + y*y + z*z);
    x *= R / rad;
    y *= R / rad;
    z *= R / rad;
    // Compute latitude and longitude
    const CeedScalar theta  = asin(z / R); // latitude
    const CeedScalar lambda = atan2(y, x); // longitude

    // Use absolute value of latitute for true solution
    // Component 1
    true_soln[i+0*Q] = sin(lambda) * cos(theta);
    // Component 2
    true_soln[i+1*Q] = 2 * true_soln[i+0*Q];
    // Component 3
    true_soln[i+2*Q] = 3 * true_soln[i+0*Q];

    // Component 1
    rhs[i+0*Q] = qdata[i+Q*0] * 2 * sin(lambda)*cos(theta) / (R*R);
    // Component 2
    rhs[i+1*Q] = 2 * rhs[i+0*Q];
    // Component 3
    rhs[i+2*Q] = 3 * rhs[i+0*Q];
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction applies the diffusion operator for a vector field of 3 components.
//
// Inputs:
//   ug     - Input vector Jacobian at quadrature points
//   qdata  - Geometric factors
//
// Output:
//   vJ     - Output vector (test functions) Jacobian at quadrature points
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(Diff3)(void *ctx, const CeedInt Q,
                      const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *ug = in[0], *qdata = in[1];
  CeedScalar *vJ = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar uJ[3][2]        = {{ug[i+(0+0*3)*Q],
                                         ug[i+(0+1*3)*Q]},
                                        {ug[i+(1+0*3)*Q],
                                         ug[i+(1+1*3)*Q]},
                                        {ug[i+(2+0*3)*Q],
                                         ug[i+(2+1*3)*Q]}
                                       };
    // Read qdata
    const CeedScalar wdetJ           =   qdata[i+Q*0];
    // -- Grad-to-Grad qdata
    // ---- dXdx_j,k * dXdx_k,j
    const CeedScalar dXdxdXdxT[2][2] = {{qdata[i+Q*1],
                                         qdata[i+Q*3]},
                                        {qdata[i+Q*3],
                                         qdata[i+Q*2]}
                                       };

    for (int k=0; k<3; k++) // k = component
      for (int j=0; j<2; j++) // j = direction of vg
        vJ[i+(k+j*3)*Q] = wdetJ * (uJ[k][0] * dXdxdXdxT[0][j] +
                                   uJ[k][1] * dXdxdXdxT[1][j]);

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // bp4sphere_h
