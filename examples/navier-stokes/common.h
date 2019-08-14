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
// All data is stored in 16 field vector of quadrature data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( v u )
//
// Determinant of Jacobian:
//   detJ = J11*A11 + J21*A12 + J31*A13
//     Jij = Jacobian entry ij
//     Aij = Adjoint ij
//
// Stored: w detJ
//     qd: 0
//
// We require the transpose of the inverse of the Jacobian to properly compute
//   integrals of the form: int( gradv u )
//
// Inverse of Jacobian:
//   Bij = Aij / detJ
//
// Stored: w B detJ = w A
//     qd: 1 2 3
//         4 5 6
//         7 8 9
//
// We require the product of the inverse of the Jacobian and its transpose to
//   properly compute integrals of the form: int( gradv gradu )
//
// Product of Inverse and Transpose:
//   BBij = sum( Bik Bkj )
//
// Stored: w B^T B detJ = w A^T A / detJ
//   Note: This matrix is symmetric
//     qd: 10 11 12
//         11 13 14
//         12 14 15
//
// *****************************************************************************
static int Setup(void *ctx, CeedInt Q,
                 const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *J = in[0], *w = in[1];
  // Outputs
  CeedScalar *qdata = out[0];

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    const CeedScalar J11 = J[i+Q*0];
    const CeedScalar J21 = J[i+Q*1];
    const CeedScalar J31 = J[i+Q*2];
    const CeedScalar J12 = J[i+Q*3];
    const CeedScalar J22 = J[i+Q*4];
    const CeedScalar J32 = J[i+Q*5];
    const CeedScalar J13 = J[i+Q*6];
    const CeedScalar J23 = J[i+Q*7];
    const CeedScalar J33 = J[i+Q*8];
    const CeedScalar A11 = J22*J33 - J23*J32;
    const CeedScalar A12 = J13*J32 - J12*J33;
    const CeedScalar A13 = J12*J23 - J13*J22;
    const CeedScalar A21 = J23*J31 - J21*J33;
    const CeedScalar A22 = J11*J33 - J13*J31;
    const CeedScalar A23 = J13*J21 - J11*J23;
    const CeedScalar A31 = J21*J32 - J22*J31;
    const CeedScalar A32 = J12*J31 - J11*J32;
    const CeedScalar A33 = J11*J22 - J12*J21;
    const CeedScalar detJ = J11*A11 + J21*A12 + J31*A13;
    const CeedScalar qw = w[i] / detJ;

    // Qdata
    // -- Interp-to-Interp qdata
    qdata[i+ 0*Q] = w[i] * detJ;
    // -- Interp-to-Grad qdata
    qdata[i+ 1*Q] = w[i] * A11;
    qdata[i+ 2*Q] = w[i] * A12;
    qdata[i+ 3*Q] = w[i] * A13;
    qdata[i+ 4*Q] = w[i] * A21;
    qdata[i+ 5*Q] = w[i] * A22;
    qdata[i+ 6*Q] = w[i] * A23;
    qdata[i+ 7*Q] = w[i] * A31;
    qdata[i+ 8*Q] = w[i] * A32;
    qdata[i+ 9*Q] = w[i] * A33;
    // -- Grad-to-Grad qdata
    qdata[i+10*Q] = qw * (A11*A11 + A12*A12 + A13*A13);
    qdata[i+11*Q] = qw * (A11*A21 + A12*A22 + A13*A23);
    qdata[i+12*Q] = qw * (A11*A31 + A12*A32 + A13*A33);
    qdata[i+13*Q] = qw * (A21*A21 + A22*A22 + A23*A23);
    qdata[i+14*Q] = qw * (A21*A31 + A22*A32 + A23*A33);
    qdata[i+15*Q] = qw * (A31*A31 + A32*A32 + A33*A33);

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
    v[i+3*Q] = w[i+0*Q] * u[i+3*Q];
    v[i+4*Q] = w[i+0*Q] * u[i+4*Q];
  }
  return 0;
}

// *****************************************************************************
#endif
