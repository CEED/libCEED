// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Helper functions for computing stabilization terms of a newtonian simulation

#ifndef stabilization_h
#define stabilization_h

#include <ceed.h>

#include "newtonian_state.h"

// *****************************************************************************
// Helper function for computing the variation in primitive variables,
//   given Tau_d
// *****************************************************************************
CEED_QFUNCTION_HELPER void dYFromTau(CeedScalar Y[5], CeedScalar Tau_d[3], CeedScalar dY[5]) {
  dY[0] = Tau_d[0] * Y[0];
  dY[1] = Tau_d[1] * Y[1];
  dY[2] = Tau_d[1] * Y[2];
  dY[3] = Tau_d[1] * Y[3];
  dY[4] = Tau_d[2] * Y[4];
}

// *****************************************************************************
// Helper functions for computing the stabilization terms
// *****************************************************************************
CEED_QFUNCTION_HELPER void StabilizationMatrix(NewtonianIdealGasContext gas, State s, CeedScalar Tau_d[3], CeedScalar R[5], const CeedScalar x[3],
                                               CeedScalar stab[5][3]) {
  CeedScalar        dY[5];
  const CeedScalar  dx_i[3] = {0};
  StateConservative dF[3];
  // Zero stab so all future terms can safely sum into it
  for (CeedInt i = 0; i < 5; i++) {
    for (CeedInt j = 0; j < 3; j++) stab[i][j] = 0;
  }
  dYFromTau(R, Tau_d, dY);
  State ds = StateFromY_fwd(gas, s, dY, x, dx_i);
  FluxInviscid_fwd(gas, s, ds, dF);
  for (CeedInt i = 0; i < 3; i++) {
    CeedScalar dF_i[5];
    UnpackState_U(dF[i], dF_i);
    for (CeedInt j = 0; j < 5; j++) stab[j][i] += dF_i[j];
  }
}

CEED_QFUNCTION_HELPER void Stabilization(NewtonianIdealGasContext gas, State s, CeedScalar Tau_d[3], State ds[3], CeedScalar U_dot[5],
                                         const CeedScalar body_force[5], const CeedScalar x[3], CeedScalar stab[5][3]) {
  // -- Stabilization method: none (Galerkin), SU, or SUPG
  CeedScalar R[5] = {0};
  switch (gas->stabilization) {
    case STAB_NONE:
      break;
    case STAB_SU:
      FluxInviscidStrong(gas, s, ds, R);
      break;
    case STAB_SUPG:
      FluxInviscidStrong(gas, s, ds, R);
      for (CeedInt j = 0; j < 5; j++) R[j] += U_dot[j] - body_force[j];
      break;
  }
  StabilizationMatrix(gas, s, Tau_d, R, x, stab);
}

// *****************************************************************************
// Helper function for computing Tau elements (stabilization constant)
//   Model from:
//     PHASTA
//
//   Tau[i] = itau=0 which is diagonal-Shakib (3 values still but not spatial)
//
// *****************************************************************************
CEED_QFUNCTION_HELPER void Tau_diagPrim(NewtonianIdealGasContext gas, State s, const CeedScalar dXdx[3][3], const CeedScalar dt,
                                        CeedScalar Tau_d[3]) {
  // Context
  const CeedScalar Ctau_t = gas->Ctau_t;
  const CeedScalar Ctau_v = gas->Ctau_v;
  const CeedScalar Ctau_C = gas->Ctau_C;
  const CeedScalar Ctau_M = gas->Ctau_M;
  const CeedScalar Ctau_E = gas->Ctau_E;
  const CeedScalar cv     = gas->cv;
  const CeedScalar mu     = gas->mu;
  const CeedScalar u[3]   = {s.Y.velocity[0], s.Y.velocity[1], s.Y.velocity[2]};
  const CeedScalar rho    = s.U.density;

  CeedScalar gijd[6];
  CeedScalar tau;
  CeedScalar dts;
  CeedScalar fact;

  //*INDENT-OFF*
  gijd[0] = dXdx[0][0] * dXdx[0][0] + dXdx[1][0] * dXdx[1][0] + dXdx[2][0] * dXdx[2][0];

  gijd[1] = dXdx[0][0] * dXdx[0][1] + dXdx[1][0] * dXdx[1][1] + dXdx[2][0] * dXdx[2][1];

  gijd[2] = dXdx[0][1] * dXdx[0][1] + dXdx[1][1] * dXdx[1][1] + dXdx[2][1] * dXdx[2][1];

  gijd[3] = dXdx[0][0] * dXdx[0][2] + dXdx[1][0] * dXdx[1][2] + dXdx[2][0] * dXdx[2][2];

  gijd[4] = dXdx[0][1] * dXdx[0][2] + dXdx[1][1] * dXdx[1][2] + dXdx[2][1] * dXdx[2][2];

  gijd[5] = dXdx[0][2] * dXdx[0][2] + dXdx[1][2] * dXdx[1][2] + dXdx[2][2] * dXdx[2][2];
  //*INDENT-ON*

  dts = Ctau_t / dt;

  tau = rho * rho *
            ((4. * dts * dts) + u[0] * (u[0] * gijd[0] + 2. * (u[1] * gijd[1] + u[2] * gijd[3])) + u[1] * (u[1] * gijd[2] + 2. * u[2] * gijd[4]) +
             u[2] * u[2] * gijd[5]) +
        Ctau_v * mu * mu *
            (gijd[0] * gijd[0] + gijd[2] * gijd[2] + gijd[5] * gijd[5] + +2. * (gijd[1] * gijd[1] + gijd[3] * gijd[3] + gijd[4] * gijd[4]));

  fact = sqrt(tau);

  Tau_d[0] = Ctau_C * fact / (rho * (gijd[0] + gijd[2] + gijd[5])) * 0.125;

  Tau_d[1] = Ctau_M / fact;
  Tau_d[2] = Ctau_E / (fact * cv);

  // consider putting back the way I initially had it  Ctau_E * Tau_d[1] /cv
  //  to avoid a division if the compiler is smart enough to see that cv IS
  // a constant that it could invert once for all elements
  // but in that case energy tau is scaled by the product of Ctau_E * Ctau_M
  // OR we could absorb cv into Ctau_E but this puts more burden on user to
  // know how to change constants with a change of fluid or units.  Same for
  // Ctau_v * mu * mu IF AND ONLY IF we don't add viscosity law =f(T)
}

// *****************************************************************************

#endif  // stabilization_h
