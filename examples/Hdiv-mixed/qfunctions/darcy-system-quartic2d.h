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
/// Darcy problem 2D (quad element) using PETSc

#ifndef DARCY_SYSTEM_QUARTIC2D_H
#define DARCY_SYSTEM_QUARTIC2D_H

#include <math.h>
#include <ceed.h>
#include "utils.h"

// -----------------------------------------------------------------------------
// See Matthew Farthing, Christopher Kees, Cass Miller (2003)
// https://www.sciencedirect.com/science/article/pii/S0309170802001872
// -----------------------------------------------------------------------------
// Strong form:
//  u        = -grad(\psi)   in \Omega
//  -\div(u) = -f                                   in \Omega
//  p        = p_b                                  on \Gamma_D
//  u.n      = u_b                                  on \Gamma_N
//
// Weak form: Find (u, \psi) \in VxQ (V=H(div), Q=L^2) on \Omega
//  (v, u) -(\div(v), \psi) = -<v, p_b*n>_{\Gamma_D}
// -(q, \div(u))  + (q, f)  = 0
//
// This QFunction setup the mixed form of the above equation
// Inputs:
//   w     : weight of quadrature
//   J     : dx/dX. x physical coordinate, X reference coordinate [-1,1]^dim
//   u     : basis_u at quadrature points
// div(u)  : divergence of basis_u at quadrature points
//   p     : basis_p at quadrature points
//   f     : force vector created in true qfunction
//
// Output:
//   v     : (v, K^{-1}/rho*k_r u) = \int (v^T * K^{-1}/rho*k_r*u detJ*w)dX ==> \int (v^T J^T * K^{-1}/rho*k_r *J*u*w/detJ)dX
//           -(v, rho*g_u)     = \int (v^T * rho*g_u detJ*w)dX ==> \int (v^T J^T * rho*g_u*w) dX
// div(v)  : -(\div(v), \psi) = -\int (div(v)^T * \psi *w) dX
//   q     : -(q, \div(u)) = -\int (q^T * div(u) * w) dX
//            (q, f)       = \int( q^T * f * w*detJ )dX
// -----------------------------------------------------------------------------
#ifndef DARCY_CTX
#define DARCY_CTX
typedef struct DARCYContext_ *DARCYContext;
struct DARCYContext_ {
  CeedScalar kappa;
  CeedScalar g;
  CeedScalar rho_a0;
  CeedScalar alpha_a, b_a;
  CeedScalar lx, ly;
};
#endif
// -----------------------------------------------------------------------------
// Residual evaluation for Darcy problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(DarcySystemQuartic2D)(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*div_u) = (const CeedScalar(*))in[3],
                   (*p) = (const CeedScalar(*))in[4],
                   (*f) = in[5];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*div_v) = (CeedScalar(*))out[1],
             (*q) = (CeedScalar(*))out[2];
  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar det_J = MatDet2x2(J);

    // *INDENT-ON*
    // (v, u): v = J^T*J*u*w/detJ
    // 1) J^T*J
    CeedScalar JT_J[2][2];
    AlphaMatTransposeMatMult2x2(1, J, J, JT_J);

    // 2) Compute v1 = (J^T*J)*u*w/detJ
    CeedScalar u1[2] = {u[0][i], u[1][i]}, v1[2];
    AlphaMatVecMult2x2(w[i]/det_J, JT_J, u1, v1);

    // Output at quadrature points: (v, u)
    for (CeedInt k = 0; k < 2; k++) {
      v[k][i] = v1[k];
    }
    // Output at quadrature points: -(\div(v), \psi)
    div_v[i] = -p[i] * w[i];
    // Output at quadrature points:-(q, \div(u))  + (q, f)
    q[i] = -div_u[i] * w[i] + f[i+0*Q]*w[i]*det_J;
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for Darcy problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(JacobianDarcySystemQuartic2D)(void *ctx, CeedInt Q,
    const CeedScalar *const *in,
    CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*du)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*div_du) = (const CeedScalar(*))in[3],
                   (*dp) = (const CeedScalar(*))in[4];

  // Outputs
  CeedScalar (*dv)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*div_dv) = (CeedScalar(*))out[1],
             (*dq) = (CeedScalar(*))out[2];
  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar det_J = MatDet2x2(J);

    // *INDENT-ON*
    // 1) Compute J^T *J
    CeedScalar JT_J[2][2];
    AlphaMatTransposeMatMult2x2(1, J, J, JT_J);

    // 2) Compute dv1 = J^T *J*du*w/detJ
    CeedScalar du1[2] = {du[0][i], du[1][i]}, dv1[2];
    AlphaMatVecMult2x2(w[i]/det_J, JT_J, du1, dv1);

    // Output at quadrature points: (dv, K^{-1}/rho*k_r u) -(dv, rho*g_u)
    for (CeedInt k = 0; k < 2; k++) {
      dv[k][i] = dv1[k];
    }
    // Output at quadrature points: -(\div(dv), d\psi)
    div_dv[i] = -dp[i] * w[i];
    // Output at quadrature points:-(dq, \div(du))
    dq[i] = -div_du[i] * w[i];
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------

#endif //End of DARCY_SYSTEM2D_H
