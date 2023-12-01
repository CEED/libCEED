// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

#include "newtonian_state.h"
#include "utils.h"

CEED_QFUNCTION_HELPER int DivDiffusiveFluxRHS(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)          = in[1];
  const CeedScalar(*q_data)          = in[2];
  CeedScalar(*Grad_v)[4][CEED_Q_VLA] = (CeedScalar(*)[4][CEED_Q_VLA])out[0];

  const NewtonianIdealGasContext context  = (NewtonianIdealGasContext)ctx;
  const StateConservative        ZeroFlux = {
             .density = 0., .momentum = {0., 0., 0.},
                  .E_total = 0.
  };
  const StateConservative ZeroInviscidFluxes[3] = {ZeroFlux, ZeroFlux, ZeroFlux};
  // const StateConservative ZeroInviscidFluxes[3] = {{0.}, {0.}, {0.}};

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(context, qi, state_var);

    CeedScalar wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    State grad_s[3];
    StatePhysicalGradientFromReference(Q, i, context, s, state_var, Grad_q, dXdx, grad_s);

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate_State(grad_s, strain_rate);
    NewtonianStress(context, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(context, s.Y, grad_s, stress, Fe);

    // Total flux
    CeedScalar DiffFlux[5][3];
    FluxTotal(ZeroInviscidFluxes, stress, Fe, DiffFlux);

    // Continuity has no diffusive flux, therefore skip
    for (CeedInt j = 1; j < 5; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        Grad_v[k][j - 1][i] = -wdetJ * (dXdx[k][0] * DiffFlux[j][0] + dXdx[k][1] * DiffFlux[j][1] + dXdx[k][2] * DiffFlux[j][2]);
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(DivDiffusiveFluxRHS_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxRHS(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(DivDiffusiveFluxRHS_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxRHS(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}
