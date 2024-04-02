// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// QFunctions for the `bc_freestream` and `bc_outflow` boundary conditions
#include "bc_freestream_type.h"
#include "newtonian_state.h"
#include "newtonian_types.h"
#include "riemann_solver.h"

// *****************************************************************************
// Freestream Boundary Condition
// *****************************************************************************
CEED_QFUNCTION_HELPER int Freestream(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var,
                                     RiemannFluxType flux_type) {
  const FreestreamContext context  = (FreestreamContext)ctx;
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)    = in[2];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)        = context->newtonian_ctx.is_implicit ? out[1] : NULL;

  const NewtonianIdealGasContext newt_ctx    = &context->newtonian_ctx;
  const bool                     is_implicit = newt_ctx->is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(newt_ctx, qi, state_var);

    CeedScalar wdetJb, norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    StateConservative flux;
    switch (flux_type) {
      case RIEMANN_HLL:
        flux = RiemannFlux_HLL(newt_ctx, s, context->S_infty, norm);
        break;
      case RIEMANN_HLLC:
        flux = RiemannFlux_HLLC(newt_ctx, s, context->S_infty, norm);
        break;
    }
    CeedScalar Flux[5];
    UnpackState_U(flux, Flux);
    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    if (is_implicit) {
      CeedScalar zeros[6] = {0.};
      StoredValuesPack(Q, i, 0, 5, qi, jac_data_sur);
      StoredValuesPack(Q, i, 5, 6, zeros, jac_data_sur);  // Every output value must be set
    }
  }
  return 0;
}

CEED_QFUNCTION(Freestream_Conserv_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Prim_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Conserv_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLLC);
}

CEED_QFUNCTION(Freestream_Prim_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLLC);
}

CEED_QFUNCTION_HELPER int Freestream_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var,
                                              RiemannFluxType flux_type) {
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)     = in[2];
  const CeedScalar(*jac_data_sur)   = in[4];

  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const FreestreamContext        context     = (FreestreamContext)ctx;
  const NewtonianIdealGasContext newt_ctx    = &context->newtonian_ctx;
  const bool                     is_implicit = newt_ctx->is_implicit;
  const State                    dS_infty    = {0};

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    CeedScalar qi[5], dqi[5];
    StoredValuesUnpack(Q, i, 0, 5, jac_data_sur, qi);
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];
    State s  = StateFromQ(newt_ctx, qi, state_var);
    State ds = StateFromQ_fwd(newt_ctx, s, dqi, state_var);

    StateConservative dflux;
    switch (flux_type) {
      case RIEMANN_HLL:
        dflux = RiemannFlux_HLL_fwd(newt_ctx, s, ds, context->S_infty, dS_infty, norm);
        break;
      case RIEMANN_HLLC:
        dflux = RiemannFlux_HLLC_fwd(newt_ctx, s, ds, context->S_infty, dS_infty, norm);
        break;
    }
    CeedScalar dFlux[5];
    UnpackState_U(dflux, dFlux);
    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * dFlux[j];
  }
  return 0;
}

CEED_QFUNCTION(Freestream_Jacobian_Conserv_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Jacobian_Prim_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Jacobian_Conserv_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLLC);
}

CEED_QFUNCTION(Freestream_Jacobian_Prim_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLLC);
}

// Note the identity
//
// softplus(x) - x = log(1 + exp(x)) - x
//                 = log(1 + exp(x)) + log(exp(-x))
//                 = log((1 + exp(x)) * exp(-x))
//                 = log(exp(-x) + 1)
//                 = softplus(-x)
CEED_QFUNCTION_HELPER CeedScalar Softplus(CeedScalar x, CeedScalar width) {
  if (x > 40 * width) return x;
  return width * log1p(exp(x / width));
}

CEED_QFUNCTION_HELPER CeedScalar Softplus_fwd(CeedScalar x, CeedScalar dx, CeedScalar width) {
  if (x > 40 * width) return 1;
  const CeedScalar t = exp(x / width);
  return t / (1 + t);
}

// Viscous Outflow boundary condition, setting a constant exterior pressure and
// temperature as input for a Riemann solve. This condition is stable even in
// recirculating flow so long as the exterior temperature is sensible.
//
// The velocity in the exterior state has optional softplus regularization to
// keep it outflow. These parameters have been finnicky in practice and provide
// little or no benefit in the tests we've run thus far, thus we recommend
// skipping this feature and just allowing recirculation.
CEED_QFUNCTION_HELPER int RiemannOutflow(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const OutflowContext outflow     = (OutflowContext)ctx;
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)        = in[1];
  const CeedScalar(*q_data_sur)    = in[2];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)        = outflow->gas.is_implicit ? out[1] : NULL;

  const NewtonianIdealGasContext gas         = &outflow->gas;
  const bool                     is_implicit = gas->is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, dXdx[2][3], norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, norm);
    wdetJb *= is_implicit ? -1. : 1.;
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s_int = StateFromQ(gas, qi, state_var);

    StatePrimitive y_ext      = s_int.Y;
    y_ext.pressure            = outflow->pressure;
    y_ext.temperature         = outflow->temperature;
    const CeedScalar u_normal = Dot3(y_ext.velocity, norm);
    const CeedScalar proj     = (1 - outflow->recirc) * Softplus(-u_normal, outflow->softplus_velocity);
    for (CeedInt j = 0; j < 3; j++) {
      y_ext.velocity[j] += norm[j] * proj;  // (I - n n^T) projects into the plane tangent to the normal
    }
    State s_ext = StateFromPrimitive(gas, y_ext);

    State grad_s[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, gas, s_int, state_var, Grad_q, dXdx, grad_s);

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate_State(grad_s, strain_rate);
    NewtonianStress(gas, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(gas, s_int.Y, grad_s, stress, Fe);

    StateConservative F_inviscid_normal = RiemannFlux_HLLC(gas, s_int, s_ext, norm);

    CeedScalar Flux[5];
    FluxTotal_RiemannBoundary(F_inviscid_normal, stress, Fe, norm, Flux);

    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    // Save values for Jacobian
    if (is_implicit) {
      StoredValuesPack(Q, i, 0, 5, qi, jac_data_sur);
      StoredValuesPack(Q, i, 5, 6, kmstress, jac_data_sur);
    }
  }
  return 0;
}

CEED_QFUNCTION(RiemannOutflow_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return RiemannOutflow(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(RiemannOutflow_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return RiemannOutflow(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

// *****************************************************************************
// Jacobian for Riemann pressure/temperature outflow boundary condition
// *****************************************************************************
CEED_QFUNCTION_HELPER int RiemannOutflow_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                  StateVariable state_var) {
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_dq)        = in[1];
  const CeedScalar(*q_data_sur)     = in[2];
  const CeedScalar(*jac_data_sur)   = in[4];
  CeedScalar(*v)[CEED_Q_VLA]        = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const OutflowContext           outflow     = (OutflowContext)ctx;
  const NewtonianIdealGasContext gas         = &outflow->gas;
  const bool                     is_implicit = gas->is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, dXdx[2][3], norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    CeedScalar qi[5], kmstress[6], dqi[5];
    StoredValuesUnpack(Q, i, 0, 5, jac_data_sur, qi);
    StoredValuesUnpack(Q, i, 5, 6, jac_data_sur, kmstress);
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];

    State          s_int  = StateFromQ(gas, qi, state_var);
    const State    ds_int = StateFromQ_fwd(gas, s_int, dqi, state_var);
    StatePrimitive y_ext = s_int.Y, dy_ext = ds_int.Y;
    y_ext.pressure             = outflow->pressure;
    y_ext.temperature          = outflow->temperature;
    dy_ext.pressure            = 0;
    dy_ext.temperature         = 0;
    const CeedScalar u_normal  = Dot3(s_int.Y.velocity, norm);
    const CeedScalar du_normal = Dot3(ds_int.Y.velocity, norm);
    const CeedScalar proj      = (1 - outflow->recirc) * Softplus(-u_normal, outflow->softplus_velocity);
    const CeedScalar dproj     = (1 - outflow->recirc) * Softplus_fwd(-u_normal, -du_normal, outflow->softplus_velocity);
    for (CeedInt j = 0; j < 3; j++) {
      y_ext.velocity[j] += norm[j] * proj;
      dy_ext.velocity[j] += norm[j] * dproj;
    }

    State s_ext  = StateFromPrimitive(gas, y_ext);
    State ds_ext = StateFromPrimitive_fwd(gas, s_ext, dy_ext);

    State grad_ds[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, gas, s_int, state_var, Grad_dq, dXdx, grad_ds);

    CeedScalar dstrain_rate[6], dkmstress[6], stress[3][3], dstress[3][3], dFe[3];
    KMStrainRate_State(grad_ds, dstrain_rate);
    NewtonianStress(gas, dstrain_rate, dkmstress);
    KMUnpack(dkmstress, dstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux_fwd(gas, s_int.Y, ds_int.Y, grad_ds, stress, dstress, dFe);

    StateConservative dF_inviscid_normal = RiemannFlux_HLLC_fwd(gas, s_int, ds_int, s_ext, ds_ext, norm);

    CeedScalar dFlux[5];
    FluxTotal_RiemannBoundary(dF_inviscid_normal, dstress, dFe, norm, dFlux);

    for (int j = 0; j < 5; j++) v[j][i] = -wdetJb * dFlux[j];
  }
  return 0;
}

CEED_QFUNCTION(RiemannOutflow_Jacobian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return RiemannOutflow_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(RiemannOutflow_Jacobian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return RiemannOutflow_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

// *****************************************************************************
// Outflow boundary condition, weakly setting a constant pressure. This is the
// classic outflow condition used by PHASTA-C and retained largely for
// comparison. In our experiments, it is never better than RiemannOutflow, and
// will crash if outflow ever becomes an inflow, as occurs with strong
// acoustics, vortices, etc.
// *****************************************************************************
CEED_QFUNCTION_HELPER int PressureOutflow(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const OutflowContext outflow     = (OutflowContext)ctx;
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)        = in[1];
  const CeedScalar(*q_data_sur)    = in[2];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)        = outflow->gas.is_implicit ? out[1] : NULL;

  const NewtonianIdealGasContext gas         = &outflow->gas;
  const bool                     is_implicit = gas->is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    State            s     = StateFromQ(gas, qi, state_var);
    s.Y.pressure           = outflow->pressure;

    CeedScalar wdetJb, dXdx[2][3], norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    State grad_s[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, gas, s, state_var, Grad_q, dXdx, grad_s);

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate_State(grad_s, strain_rate);
    NewtonianStress(gas, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(gas, s.Y, grad_s, stress, Fe);

    StateConservative F_inviscid[3];
    FluxInviscid(gas, s, F_inviscid);

    CeedScalar Flux[5];
    FluxTotal_Boundary(F_inviscid, stress, Fe, norm, Flux);

    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    // Save values for Jacobian
    if (is_implicit) {
      StoredValuesPack(Q, i, 0, 5, qi, jac_data_sur);
      StoredValuesPack(Q, i, 5, 6, kmstress, jac_data_sur);
    }
  }
  return 0;
}

CEED_QFUNCTION(PressureOutflow_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(PressureOutflow_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

// *****************************************************************************
// Jacobian for weak-pressure outflow boundary condition
// *****************************************************************************
CEED_QFUNCTION_HELPER int PressureOutflow_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                   StateVariable state_var) {
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_dq)        = in[1];
  const CeedScalar(*q_data_sur)     = in[2];
  const CeedScalar(*jac_data_sur)   = in[4];
  CeedScalar(*v)[CEED_Q_VLA]        = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const OutflowContext           outflow     = (OutflowContext)ctx;
  const NewtonianIdealGasContext gas         = &outflow->gas;
  const bool                     is_implicit = gas->is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, dXdx[2][3], norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    CeedScalar qi[5], kmstress[6], dqi[5];
    StoredValuesUnpack(Q, i, 0, 5, jac_data_sur, qi);
    StoredValuesUnpack(Q, i, 5, 6, jac_data_sur, kmstress);
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];

    State s       = StateFromQ(gas, qi, state_var);
    State ds      = StateFromQ_fwd(gas, s, dqi, state_var);
    s.Y.pressure  = outflow->pressure;
    ds.Y.pressure = 0.;

    State grad_ds[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, gas, s, state_var, Grad_dq, dXdx, grad_ds);

    CeedScalar dstrain_rate[6], dkmstress[6], stress[3][3], dstress[3][3], dFe[3];
    KMStrainRate_State(grad_ds, dstrain_rate);
    NewtonianStress(gas, dstrain_rate, dkmstress);
    KMUnpack(dkmstress, dstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux_fwd(gas, s.Y, ds.Y, grad_ds, stress, dstress, dFe);

    StateConservative dF_inviscid[3];
    FluxInviscid_fwd(gas, s, ds, dF_inviscid);

    CeedScalar dFlux[5];
    FluxTotal_Boundary(dF_inviscid, dstress, dFe, norm, dFlux);

    for (int j = 0; j < 5; j++) v[j][i] = -wdetJb * dFlux[j];
  }
  return 0;
}

CEED_QFUNCTION(PressureOutflow_Jacobian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(PressureOutflow_Jacobian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}
