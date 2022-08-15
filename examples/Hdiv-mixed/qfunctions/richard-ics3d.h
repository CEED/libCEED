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
/// Force of Richard problem 3D (quad element) using PETSc

#ifndef RICHARD_ICS3D_H
#define RICHARD_ICS3D_H

#include <math.h>
#include <ceed.h>
#include "ceed/ceed-f64.h"
#include "utils.h"

#ifndef RICHARD_CTX
#define RICHARD_CTX
typedef struct RICHARDContext_ *RICHARDContext;
struct RICHARDContext_ {
  CeedScalar kappa;
  CeedScalar g;
  CeedScalar rho_a0;
  CeedScalar alpha_a, b_a;
  CeedScalar beta, p0;
  CeedScalar t, t_final, dt;
  CeedScalar gamma;
  CeedScalar lx, ly, lz;
};
#endif

// -----------------------------------------------------------------------------
// We solve (v, u) = (v, ue) at t=0, to project ue to Hdiv space
// This QFunction create rhs_u0 = (v, ue)
// -----------------------------------------------------------------------------
CEED_QFUNCTION(RichardRhsU03D)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*coords) = in[1],
                   (*dxdX)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*rhs_u0) = out[0];
  // Context
  RICHARDContext  context = (RICHARDContext)ctx;
  const CeedScalar kappa    = context->kappa;
  const CeedScalar alpha_a  = context->alpha_a;
  const CeedScalar b_a      = context->b_a;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q], z = coords[i+2*Q];
    const CeedScalar J[3][3] = {{dxdX[0][0][i], dxdX[1][0][i], dxdX[2][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i], dxdX[2][1][i]},
                                {dxdX[0][2][i], dxdX[1][2][i], dxdX[2][2][i]}};
    // psi = exp(-gamma*t)*sin(pi*x)*sin(pi*y)
    CeedScalar psi1_x = PI_DOUBLE*cos(PI_DOUBLE*x)*sin(PI_DOUBLE*y)*sin(PI_DOUBLE*z);
    CeedScalar psi1_y = PI_DOUBLE*sin(PI_DOUBLE*x)*cos(PI_DOUBLE*y)*sin(PI_DOUBLE*z);
    CeedScalar psi1_z = PI_DOUBLE*sin(PI_DOUBLE*x)*sin(PI_DOUBLE*y)*cos(PI_DOUBLE*z);

    // k_r = b_a + alpha_a * (1 - x*y)
    CeedScalar k_r = b_a + alpha_a*(1-x*y*z);
    // rho = rho_a/rho_a0
    CeedScalar rho = 1.;
    // ue = -rho*k_r*K *[grad(\psi)]
    CeedScalar ue[3] = {-rho*k_r*kappa*psi1_x,
                        -rho*k_r*kappa*psi1_y,
                        -rho*k_r*kappa*psi1_z};
    CeedScalar rhs1[3];
    // rhs = (v, ue) = J^T*ue*w
    AlphaMatTransposeVecMult3x3(w[i], J, ue, rhs1);
    // 
    rhs_u0[i+0*Q] = rhs1[0];
    rhs_u0[i+1*Q] = rhs1[1];
    rhs_u0[i+2*Q] = rhs1[2];
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// We solve (v, u) = (v, ue) at t=0, to project ue to Hdiv space
// This QFunction create mass matrix (v, u), then we solve using ksp to have 
// projected ue in Hdiv space and use it for initial conditions 
// -----------------------------------------------------------------------------
CEED_QFUNCTION(RichardICsU3D)(void *ctx, const CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    const CeedScalar J[3][3] = {{dxdX[0][0][i], dxdX[1][0][i], dxdX[2][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i], dxdX[2][1][i]},
                                {dxdX[0][2][i], dxdX[1][2][i], dxdX[2][2][i]}};
    const CeedScalar det_J = MatDet3x3(J);

    // *INDENT-ON*
    // (v, u): v1 = J^T*J*u*w/detJ
    // 1) Compute J^T *J
    CeedScalar JT_J[3][3];
    AlphaMatTransposeMatMult3x3(1, J, J, JT_J);

    // 4) Compute v1 = J^T*J*u*w/detJ
    CeedScalar u1[3] = {u[0][i], u[1][i], u[2][i]}, v1[3];
    AlphaMatVecMult3x3(w[i]/det_J, JT_J, u1, v1);

    // Output at quadrature points: (v, K^{-1}/rho*k_r u)
    for (CeedInt k = 0; k < 3; k++) {
      v[k][i] = v1[k];
    }
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// We solve (q, p) = (q, pe) at t=0, to project pe to L2 space
// This QFunction create rhs_p0 = (q, pe)
// -----------------------------------------------------------------------------
CEED_QFUNCTION(RichardRhsP03D)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*coords) = in[1],
                   (*dxdX)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*rhs_p0) = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q], z = coords[i+2*Q];
    const CeedScalar J[3][3] = {{dxdX[0][0][i], dxdX[1][0][i], dxdX[2][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i], dxdX[2][1][i]},
                                {dxdX[0][2][i], dxdX[1][2][i], dxdX[2][2][i]}};
    const CeedScalar det_J = MatDet3x3(J);
    // psi = exp(-gamma*t)*sin(pi*x)*sin(pi*y)
    CeedScalar psi1 = sin(PI_DOUBLE*x)*sin(PI_DOUBLE*y)*sin(PI_DOUBLE*z);

    // rhs = (q, pe) = pe*w*det_J
    rhs_p0[i] = psi1*w[i]*det_J;
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// We solve (q, p) = (q, pe) at t=0, to project pe to L2 space
// This QFunction create mass matrix (q, p), then we solve using ksp to have 
// projected pe in L2 space and use it for initial conditions 
// -----------------------------------------------------------------------------
CEED_QFUNCTION(RichardICsP3D)(void *ctx, const CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*p) = (const CeedScalar(*))in[2];

  // Outputs
  CeedScalar (*q) = (CeedScalar(*))out[0];
  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    const CeedScalar J[3][3] = {{dxdX[0][0][i], dxdX[1][0][i], dxdX[2][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i], dxdX[2][1][i]},
                                {dxdX[0][2][i], dxdX[1][2][i], dxdX[2][2][i]}};
    const CeedScalar det_J = MatDet3x3(J);

    // Output at quadrature points: (q, p)
    q[i] = p[i]*w[i]*det_J;
  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif //End of RICHARD_ICS3D_H
