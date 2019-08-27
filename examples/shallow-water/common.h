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
/// Geometric factors and mass operator for Navier-Stokes example using PETSc

#ifndef common_h
#define common_h

#ifndef CeedPragmaOMP
#  ifdef _OPENMP
#    define CeedPragmaOMP_(a) _Pragma(#a)
#    define CeedPragmaOMP(a) CeedPragmaOMP_(omp a)
#  else
#    define CeedPragmaOMP(a)
#  endif
#endif

#include <math.h>

// *****************************************************************************
// This QFunction sets up the geometric factors required for integration and
//   coordinate transformations
//
// All data is stored in 8 field vector of quadrature data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( v u )
//
// Determinant of Jacobian in 2D:
//   detJ = J11*J22 - J12*J21
//     Jij = Jacobian entry ij
//
// Stored: w detJ
//   qd: 0
//
// We require the transpose of the inverse of the Jacobian to properly compute
//   integrals of the form: int( gradv u )
//
// Inverse of Jacobian:
//   Bij = Aij / detJ
//
// Stored: w B detJ = w A
//   qd: 1 2
//       3 4
//
// We require the product of the inverse of the Jacobian and its transpose to
//   properly compute integrals of the form: int( gradv gradu)
//
// Product of Inverse and Transpose:
//   BBij = sum( Bik Bkj )
//
// Stored: w B^T B detJ = w A^T A / detJ
//   Note: This matrix is symmetric
//     qd: 5 6
//         6 7
//
// *****************************************************************************
static int Setup(void *ctx, CeedInt Q,
                 const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *X = in[0], *J = in[1], *w = in[2];
  // Outputs
  CeedScalar *qdata = out[0];

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    const CeedScalar J11  =   J[i+Q*0];
    const CeedScalar J21  =   J[i+Q*1];
    const CeedScalar J12  =   J[i+Q*2];
    const CeedScalar J22  =   J[i+Q*4];
    const CeedScalar A11  =   J22;
    const CeedScalar A12  = - J12;
    const CeedScalar A21  = - J21;
    const CeedScalar A22  =   J11;
    const CeedScalar detJ =   J11*J22 - J12*J21;
    const CeedScalar qw   =   w[i] / detJ;

    // Qdata
    // Geometric Factors
    // -- Interp-to-Interp qdata
    qdata[i+ 0*Q] = w[i] * detJ;
    // -- Interp-to-Grad qdata
    qdata[i+ 1*Q] = w[i] * A11;
    qdata[i+ 2*Q] = w[i] * A12;
    qdata[i+ 3*Q] = w[i] * A21;
    qdata[i+ 4*Q] = w[i] * A22;
    // -- Grad-to-Grad qdata
    qdata[i+5*Q] = qw * (A11*A11 + A12*A12);
    qdata[i+6*Q] = qw * (A11*A21 + A12*A22);
    qdata[i+7*Q] = qw * (A21*A21 + A22*A22);

    // -- Coordinates
    const CeedScalar x    = X[i+0*Q];
    const CeedScalar y    = X[i+1*Q];
    // h_s terrain function
    qdata[i+8*Q] = sin(x) + cos(y); // put 0 for constant flat topography
    // H_0 reference height function
    qdata[i+9*Q] = 0; // flat
  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction applies the mass matrix to five interlaced fields.
//
// Inputs:
//   u - Input vector at quadrature points
//   w - Quadrature weights
//
// Output:
//   v - Output vector at quadrature points
//
// *****************************************************************************
static int Mass(void *ctx, CeedInt Q,
                const CeedScalar *const *in, CeedScalar *const *out) {
  (void)ctx;
  const CeedScalar *u = in[0], *w = in[1];
  CeedScalar *v = out[0];

  CeedPragmaOMP(simd)
  for (CeedInt i=0; i<Q; i++) {
    v[i+0*Q] = w[i+0*Q] * u[i+0*Q];
    v[i+1*Q] = w[i+0*Q] * u[i+1*Q];
    v[i+2*Q] = w[i+0*Q] * u[i+2*Q];
  }
  return 0;
}

// *****************************************************************************
#endif
