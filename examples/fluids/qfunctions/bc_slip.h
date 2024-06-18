// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// QFunctions for the `bc_slip` boundary conditions
#include "bc_freestream_type.h"
#include "newtonian_state.h"
#include "newtonian_types.h"
#include "riemann_solver.h"

CEED_QFUNCTION_HELPER int Slip(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const NewtonianIdealGasContext newt_ctx = (const NewtonianIdealGasContext)ctx;
  const CeedScalar(*q)[CEED_Q_VLA]        = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)           = in[2];
  CeedScalar(*v)[CEED_Q_VLA]              = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)               = newt_ctx->is_implicit ? out[1] : NULL;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    State            s     = StateFromQ(newt_ctx, qi, state_var);

    CeedScalar wdetJb, norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, norm);
    wdetJb *= newt_ctx->is_implicit ? -1. : 1.;

    CeedScalar       vel_reflect[3];
    const CeedScalar vel_normal = Dot3(s.Y.velocity, norm);
    for (CeedInt j = 0; j < 3; j++) vel_reflect[j] = s.Y.velocity[j] - 2. * norm[j] * vel_normal;
    const CeedScalar Y_reflect[5] = {s.Y.pressure, vel_reflect[0], vel_reflect[1], vel_reflect[2], s.Y.temperature};
    State            s_reflect    = StateFromY(newt_ctx, Y_reflect);

    StateConservative flux = RiemannFlux_HLLC(newt_ctx, s, s_reflect, norm);

    CeedScalar Flux[5];
    UnpackState_U(flux, Flux);
    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    if (newt_ctx->is_implicit) {
      CeedScalar zeros[6] = {0.};
      StoredValuesPack(Q, i, 0, 5, qi, jac_data_sur);
      StoredValuesPack(Q, i, 5, 6, zeros, jac_data_sur);  // Every output value must be set
    }
  }
  return 0;
}

CEED_QFUNCTION(Slip_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Slip(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(Slip_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Slip(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(Slip_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Slip(ctx, Q, in, out, STATEVAR_ENTROPY);
}

CEED_QFUNCTION_HELPER int Slip_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)     = in[2];
  const CeedScalar(*jac_data_sur)   = in[4];

  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const NewtonianIdealGasContext newt_ctx = (const NewtonianIdealGasContext)ctx;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, norm);
    wdetJb *= newt_ctx->is_implicit ? -1. : 1.;

    CeedScalar qi[5], dqi[5];
    StoredValuesUnpack(Q, i, 0, 5, jac_data_sur, qi);
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];
    State s  = StateFromQ(newt_ctx, qi, state_var);
    State ds = StateFromQ_fwd(newt_ctx, s, dqi, state_var);

    CeedScalar       vel_reflect[3];
    const CeedScalar vel_normal = Dot3(s.Y.velocity, norm);
    for (CeedInt j = 0; j < 3; j++) vel_reflect[j] = s.Y.velocity[j] - 2. * norm[j] * vel_normal;
    const CeedScalar Y_reflect[5] = {s.Y.pressure, vel_reflect[0], vel_reflect[1], vel_reflect[2], s.Y.temperature};
    State            s_reflect    = StateFromY(newt_ctx, Y_reflect);

    CeedScalar       dvel_reflect[3];
    const CeedScalar dvel_normal = Dot3(ds.Y.velocity, norm);
    for (CeedInt j = 0; j < 3; j++) dvel_reflect[j] = ds.Y.velocity[j] - 2. * norm[j] * dvel_normal;
    const CeedScalar dY_reflect[5] = {ds.Y.pressure, dvel_reflect[0], dvel_reflect[1], dvel_reflect[2], ds.Y.temperature};
    State            ds_reflect    = StateFromY_fwd(newt_ctx, s_reflect, dY_reflect);

    StateConservative dflux = RiemannFlux_HLLC_fwd(newt_ctx, s, ds, s_reflect, ds_reflect, norm);

    CeedScalar dFlux[5];
    UnpackState_U(dflux, dFlux);
    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * dFlux[j];
  }
  return 0;
}

CEED_QFUNCTION(Slip_Jacobian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Slip_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(Slip_Jacobian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Slip_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(Slip_Jacobian_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Slip_Jacobian(ctx, Q, in, out, STATEVAR_ENTROPY);
}
