// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Structs and helper functions regarding the state of a newtonian simulation

#ifndef newtonian_state_h
#define newtonian_state_h

#include <ceed.h>
#include <math.h>

#include "newtonian_types.h"
#include "utils.h"

typedef struct {
  CeedScalar pressure;
  CeedScalar velocity[3];
  CeedScalar temperature;
} StatePrimitive;

typedef struct {
  CeedScalar density;
  CeedScalar momentum[3];
  CeedScalar E_total;
} StateConservative;

typedef struct {
  StateConservative U;
  StatePrimitive    Y;
} State;

CEED_QFUNCTION_HELPER void UnpackState_U(StateConservative s, CeedScalar U[5]) {
  U[0] = s.density;
  for (int i = 0; i < 3; i++) U[i + 1] = s.momentum[i];
  U[4] = s.E_total;
}

CEED_QFUNCTION_HELPER void UnpackState_Y(StatePrimitive s, CeedScalar Y[5]) {
  Y[0] = s.pressure;
  for (int i = 0; i < 3; i++) Y[i + 1] = s.velocity[i];
  Y[4] = s.temperature;
}

CEED_QFUNCTION_HELPER CeedScalar HeatCapacityRatio(NewtonianIdealGasContext gas) { return gas->cp / gas->cv; }

CEED_QFUNCTION_HELPER CeedScalar GasConstant(NewtonianIdealGasContext gas) { return gas->cp - gas->cv; }

CEED_QFUNCTION_HELPER CeedScalar Prandtl(NewtonianIdealGasContext gas) { return gas->cp * gas->mu / gas->k; }

CEED_QFUNCTION_HELPER CeedScalar SoundSpeed(NewtonianIdealGasContext gas, CeedScalar T) { return sqrt(gas->cp * (HeatCapacityRatio(gas) - 1.) * T); }

CEED_QFUNCTION_HELPER CeedScalar Mach(NewtonianIdealGasContext gas, CeedScalar T, CeedScalar u) { return u / SoundSpeed(gas, T); }

CEED_QFUNCTION_HELPER CeedScalar TotalSpecificEnthalpy(NewtonianIdealGasContext gas, const State s) {
  // Ignoring potential energy
  CeedScalar e_internal = gas->cv * s.Y.temperature;
  CeedScalar e_kinetic  = 0.5 * Dot3(s.Y.velocity, s.Y.velocity);
  return e_internal + e_kinetic + s.Y.pressure / s.U.density;
}

CEED_QFUNCTION_HELPER CeedScalar TotalSpecificEnthalpy_fwd(NewtonianIdealGasContext gas, const State s, const State ds) {
  // Ignoring potential energy
  CeedScalar de_kinetic  = Dot3(ds.Y.velocity, s.Y.velocity);
  CeedScalar de_internal = gas->cv * ds.Y.temperature;
  return de_internal + de_kinetic + ds.Y.pressure / s.U.density - s.Y.pressure / Square(s.U.density) * ds.U.density;
}

CEED_QFUNCTION_HELPER StatePrimitive StatePrimitiveFromConservative(NewtonianIdealGasContext gas, StateConservative U, const CeedScalar x[3]) {
  StatePrimitive Y;
  for (CeedInt i = 0; i < 3; i++) Y.velocity[i] = U.momentum[i] / U.density;
  CeedScalar e_kinetic   = .5 * Dot3(Y.velocity, Y.velocity);
  CeedScalar e_potential = -Dot3(gas->g, x);
  CeedScalar e_total     = U.E_total / U.density;
  CeedScalar e_internal  = e_total - e_kinetic - e_potential;
  Y.temperature          = e_internal / gas->cv;
  Y.pressure             = (HeatCapacityRatio(gas) - 1) * U.density * e_internal;
  return Y;
}

CEED_QFUNCTION_HELPER StatePrimitive StatePrimitiveFromConservative_fwd(NewtonianIdealGasContext gas, State s, StateConservative dU,
                                                                        const CeedScalar x[3], const CeedScalar dx[3]) {
  StatePrimitive dY;
  for (CeedInt i = 0; i < 3; i++) {
    dY.velocity[i] = (dU.momentum[i] - s.Y.velocity[i] * dU.density) / s.U.density;
  }
  CeedScalar e_kinetic    = .5 * Dot3(s.Y.velocity, s.Y.velocity);
  CeedScalar de_kinetic   = Dot3(dY.velocity, s.Y.velocity);
  CeedScalar e_potential  = -Dot3(gas->g, x);
  CeedScalar de_potential = -Dot3(gas->g, dx);
  CeedScalar e_total      = s.U.E_total / s.U.density;
  CeedScalar de_total     = (dU.E_total - e_total * dU.density) / s.U.density;
  CeedScalar e_internal   = e_total - e_kinetic - e_potential;
  CeedScalar de_internal  = de_total - de_kinetic - de_potential;
  dY.temperature          = de_internal / gas->cv;
  dY.pressure             = (HeatCapacityRatio(gas) - 1) * (dU.density * e_internal + s.U.density * de_internal);
  return dY;
}

CEED_QFUNCTION_HELPER StateConservative StateConservativeFromPrimitive(NewtonianIdealGasContext gas, StatePrimitive Y, const CeedScalar x[3]) {
  StateConservative U;
  U.density = Y.pressure / (GasConstant(gas) * Y.temperature);
  for (int i = 0; i < 3; i++) U.momentum[i] = U.density * Y.velocity[i];
  CeedScalar e_internal  = gas->cv * Y.temperature;
  CeedScalar e_kinetic   = .5 * Dot3(Y.velocity, Y.velocity);
  CeedScalar e_potential = -Dot3(gas->g, x);
  CeedScalar e_total     = e_internal + e_kinetic + e_potential;
  U.E_total              = U.density * e_total;
  return U;
}

CEED_QFUNCTION_HELPER StateConservative StateConservativeFromPrimitive_fwd(NewtonianIdealGasContext gas, State s, StatePrimitive dY,
                                                                           const CeedScalar x[3], const CeedScalar dx[3]) {
  StateConservative dU;
  dU.density = (dY.pressure * s.Y.temperature - s.Y.pressure * dY.temperature) / (GasConstant(gas) * s.Y.temperature * s.Y.temperature);
  for (int i = 0; i < 3; i++) {
    dU.momentum[i] = dU.density * s.Y.velocity[i] + s.U.density * dY.velocity[i];
  }
  CeedScalar e_kinetic    = .5 * Dot3(s.Y.velocity, s.Y.velocity);
  CeedScalar de_kinetic   = Dot3(dY.velocity, s.Y.velocity);
  CeedScalar e_potential  = -Dot3(gas->g, x);
  CeedScalar de_potential = -Dot3(gas->g, dx);
  CeedScalar e_internal   = gas->cv * s.Y.temperature;
  CeedScalar de_internal  = gas->cv * dY.temperature;
  CeedScalar e_total      = e_internal + e_kinetic + e_potential;
  CeedScalar de_total     = de_internal + de_kinetic + de_potential;
  dU.E_total              = dU.density * e_total + s.U.density * de_total;
  return dU;
}

// Function pointer types for generic state array -> State struct functions
typedef State (*StateFromQi_t)(NewtonianIdealGasContext gas, const CeedScalar qi[5], const CeedScalar x[3]);
typedef State (*StateFromQi_fwd_t)(NewtonianIdealGasContext gas, State s, const CeedScalar dqi[5], const CeedScalar x[3], const CeedScalar dx[3]);

CEED_QFUNCTION_HELPER State StateFromU(NewtonianIdealGasContext gas, const CeedScalar U[5], const CeedScalar x[3]) {
  State s;
  s.U.density     = U[0];
  s.U.momentum[0] = U[1];
  s.U.momentum[1] = U[2];
  s.U.momentum[2] = U[3];
  s.U.E_total     = U[4];
  s.Y             = StatePrimitiveFromConservative(gas, s.U, x);
  return s;
}

CEED_QFUNCTION_HELPER State StateFromU_fwd(NewtonianIdealGasContext gas, State s, const CeedScalar dU[5], const CeedScalar x[3],
                                           const CeedScalar dx[3]) {
  State ds;
  ds.U.density     = dU[0];
  ds.U.momentum[0] = dU[1];
  ds.U.momentum[1] = dU[2];
  ds.U.momentum[2] = dU[3];
  ds.U.E_total     = dU[4];
  ds.Y             = StatePrimitiveFromConservative_fwd(gas, s, ds.U, x, dx);
  return ds;
}

CEED_QFUNCTION_HELPER State StateFromY(NewtonianIdealGasContext gas, const CeedScalar Y[5], const CeedScalar x[3]) {
  State s;
  s.Y.pressure    = Y[0];
  s.Y.velocity[0] = Y[1];
  s.Y.velocity[1] = Y[2];
  s.Y.velocity[2] = Y[3];
  s.Y.temperature = Y[4];
  s.U             = StateConservativeFromPrimitive(gas, s.Y, x);
  return s;
}

CEED_QFUNCTION_HELPER State StateFromY_fwd(NewtonianIdealGasContext gas, State s, const CeedScalar dY[5], const CeedScalar x[3],
                                           const CeedScalar dx[3]) {
  State ds;
  ds.Y.pressure    = dY[0];
  ds.Y.velocity[0] = dY[1];
  ds.Y.velocity[1] = dY[2];
  ds.Y.velocity[2] = dY[3];
  ds.Y.temperature = dY[4];
  ds.U             = StateConservativeFromPrimitive_fwd(gas, s, ds.Y, x, dx);
  return ds;
}

// Function pointer types for State struct -> generic state array
typedef void (*StateToQi_t)(NewtonianIdealGasContext gas, const State input, CeedScalar qi[5]);

CEED_QFUNCTION_HELPER void StateToU(NewtonianIdealGasContext gas, const State input, CeedScalar U[5]) { UnpackState_U(input.U, U); }

CEED_QFUNCTION_HELPER void StateToY(NewtonianIdealGasContext gas, const State input, CeedScalar Y[5]) { UnpackState_Y(input.Y, Y); }

CEED_QFUNCTION_HELPER void FluxInviscid(NewtonianIdealGasContext gas, State s, StateConservative Flux[3]) {
  for (CeedInt i = 0; i < 3; i++) {
    Flux[i].density = s.U.momentum[i];
    for (CeedInt j = 0; j < 3; j++) Flux[i].momentum[j] = s.U.momentum[i] * s.Y.velocity[j] + s.Y.pressure * (i == j);
    Flux[i].E_total = (s.U.E_total + s.Y.pressure) * s.Y.velocity[i];
  }
}

CEED_QFUNCTION_HELPER void FluxInviscid_fwd(NewtonianIdealGasContext gas, State s, State ds, StateConservative dFlux[3]) {
  for (CeedInt i = 0; i < 3; i++) {
    dFlux[i].density = ds.U.momentum[i];
    for (CeedInt j = 0; j < 3; j++) {
      dFlux[i].momentum[j] = ds.U.momentum[i] * s.Y.velocity[j] + s.U.momentum[i] * ds.Y.velocity[j] + ds.Y.pressure * (i == j);
    }
    dFlux[i].E_total = (ds.U.E_total + ds.Y.pressure) * s.Y.velocity[i] + (s.U.E_total + s.Y.pressure) * ds.Y.velocity[i];
  }
}

CEED_QFUNCTION_HELPER StateConservative FluxInviscidDotNormal(NewtonianIdealGasContext gas, State s, const CeedScalar normal[3]) {
  StateConservative Flux[3], Flux_dot_n = {0};
  FluxInviscid(gas, s, Flux);
  for (CeedInt i = 0; i < 3; i++) {
    Flux_dot_n.density += Flux[i].density * normal[i];
    for (CeedInt j = 0; j < 3; j++) Flux_dot_n.momentum[j] += Flux[i].momentum[j] * normal[i];
    Flux_dot_n.E_total += Flux[i].E_total * normal[i];
  }
  return Flux_dot_n;
}

CEED_QFUNCTION_HELPER StateConservative FluxInviscidDotNormal_fwd(NewtonianIdealGasContext gas, State s, State ds, const CeedScalar normal[3]) {
  StateConservative dFlux[3], Flux_dot_n = {0};
  FluxInviscid_fwd(gas, s, ds, dFlux);
  for (CeedInt i = 0; i < 3; i++) {
    Flux_dot_n.density += dFlux[i].density * normal[i];
    for (CeedInt j = 0; j < 3; j++) Flux_dot_n.momentum[j] += dFlux[i].momentum[j] * normal[i];
    Flux_dot_n.E_total += dFlux[i].E_total * normal[i];
  }
  return Flux_dot_n;
}

CEED_QFUNCTION_HELPER void FluxInviscidStrong(NewtonianIdealGasContext gas, State s, State ds[3], CeedScalar strong_conv[5]) {
  for (CeedInt i = 0; i < 5; i++) strong_conv[i] = 0;
  for (CeedInt i = 0; i < 3; i++) {
    StateConservative dF[3];
    FluxInviscid_fwd(gas, s, ds[i], dF);
    CeedScalar dF_i[5];
    UnpackState_U(dF[i], dF_i);
    for (CeedInt j = 0; j < 5; j++) strong_conv[j] += dF_i[j];
  }
}

CEED_QFUNCTION_HELPER void FluxTotal(const StateConservative F_inviscid[3], CeedScalar stress[3][3], CeedScalar Fe[3], CeedScalar Flux[5][3]) {
  for (CeedInt j = 0; j < 3; j++) {
    Flux[0][j] = F_inviscid[j].density;
    for (CeedInt k = 0; k < 3; k++) Flux[k + 1][j] = F_inviscid[j].momentum[k] - stress[k][j];
    Flux[4][j] = F_inviscid[j].E_total + Fe[j];
  }
}

CEED_QFUNCTION_HELPER void FluxTotal_Boundary(const StateConservative F_inviscid[3], const CeedScalar stress[3][3], const CeedScalar Fe[3],
                                              const CeedScalar normal[3], CeedScalar Flux[5]) {
  for (CeedInt j = 0; j < 5; j++) Flux[j] = 0.;
  for (CeedInt j = 0; j < 3; j++) {
    Flux[0] += F_inviscid[j].density * normal[j];
    for (CeedInt k = 0; k < 3; k++) {
      Flux[k + 1] += (F_inviscid[j].momentum[k] - stress[k][j]) * normal[j];
    }
    Flux[4] += (F_inviscid[j].E_total + Fe[j]) * normal[j];
  }
}

// Kelvin-Mandel notation
CEED_QFUNCTION_HELPER void KMStrainRate(const State grad_s[3], CeedScalar strain_rate[6]) {
  const CeedScalar weight = 1 / sqrt(2.);
  strain_rate[0]          = grad_s[0].Y.velocity[0];
  strain_rate[1]          = grad_s[1].Y.velocity[1];
  strain_rate[2]          = grad_s[2].Y.velocity[2];
  strain_rate[3]          = weight * (grad_s[2].Y.velocity[1] + grad_s[1].Y.velocity[2]);
  strain_rate[4]          = weight * (grad_s[2].Y.velocity[0] + grad_s[0].Y.velocity[2]);
  strain_rate[5]          = weight * (grad_s[1].Y.velocity[0] + grad_s[0].Y.velocity[1]);
}

CEED_QFUNCTION_HELPER void NewtonianStress(NewtonianIdealGasContext gas, const CeedScalar strain_rate[6], CeedScalar stress[6]) {
  CeedScalar div_u = strain_rate[0] + strain_rate[1] + strain_rate[2];
  for (CeedInt i = 0; i < 6; i++) {
    stress[i] = gas->mu * (2 * strain_rate[i] + gas->lambda * div_u * (i < 3));
  }
}

CEED_QFUNCTION_HELPER void ViscousEnergyFlux(NewtonianIdealGasContext gas, StatePrimitive Y, const State grad_s[3], const CeedScalar stress[3][3],
                                             CeedScalar Fe[3]) {
  for (CeedInt i = 0; i < 3; i++) {
    Fe[i] = -Y.velocity[0] * stress[0][i] - Y.velocity[1] * stress[1][i] - Y.velocity[2] * stress[2][i] - gas->k * grad_s[i].Y.temperature;
  }
}

CEED_QFUNCTION_HELPER void ViscousEnergyFlux_fwd(NewtonianIdealGasContext gas, StatePrimitive Y, StatePrimitive dY, const State grad_ds[3],
                                                 const CeedScalar stress[3][3], const CeedScalar dstress[3][3], CeedScalar dFe[3]) {
  for (CeedInt i = 0; i < 3; i++) {
    dFe[i] = -Y.velocity[0] * dstress[0][i] - dY.velocity[0] * stress[0][i] - Y.velocity[1] * dstress[1][i] - dY.velocity[1] * stress[1][i] -
             Y.velocity[2] * dstress[2][i] - dY.velocity[2] * stress[2][i] - gas->k * grad_ds[i].Y.temperature;
  }
}

#endif  // newtonian_state_h
