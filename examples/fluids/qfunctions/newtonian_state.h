// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Structs and helper functions regarding the state of a newtonian simulation
#pragma once

#include <ceed/types.h>
#ifndef CEED_RUNNING_JIT_PASS
#include <math.h>
#endif

#include "newtonian_types.h"
#include "utils.h"

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

CEED_QFUNCTION_HELPER void UnpackState_V(StateEntropy s, CeedScalar V[5]) {
  V[0] = s.S_density;
  for (int i = 0; i < 3; i++) V[i + 1] = s.S_momentum[i];
  V[4] = s.S_energy;
}

CEED_QFUNCTION_HELPER CeedScalar HeatCapacityRatio(NewtonianIdealGasContext gas) { return gas->cp / gas->cv; }

CEED_QFUNCTION_HELPER CeedScalar GasConstant(NewtonianIdealGasContext gas) { return gas->cp - gas->cv; }

CEED_QFUNCTION_HELPER CeedScalar Prandtl(NewtonianIdealGasContext gas) { return gas->cp * gas->mu / gas->k; }

CEED_QFUNCTION_HELPER CeedScalar SoundSpeed(NewtonianIdealGasContext gas, CeedScalar T) { return sqrt(gas->cp * (HeatCapacityRatio(gas) - 1.) * T); }

CEED_QFUNCTION_HELPER CeedScalar Mach(NewtonianIdealGasContext gas, CeedScalar T, CeedScalar u) { return u / SoundSpeed(gas, T); }

CEED_QFUNCTION_HELPER CeedScalar TotalSpecificEnthalpy(NewtonianIdealGasContext gas, const State s) {
  CeedScalar e_kinetic  = 0.5 * Dot3(s.Y.velocity, s.Y.velocity);
  CeedScalar e_internal = gas->cv * s.Y.temperature;
  return e_internal + e_kinetic + s.Y.pressure / s.U.density;
}

CEED_QFUNCTION_HELPER CeedScalar TotalSpecificEnthalpy_fwd(NewtonianIdealGasContext gas, const State s, const State ds) {
  CeedScalar de_kinetic  = Dot3(ds.Y.velocity, s.Y.velocity);
  CeedScalar de_internal = gas->cv * ds.Y.temperature;
  return de_internal + de_kinetic + ds.Y.pressure / s.U.density - s.Y.pressure / Square(s.U.density) * ds.U.density;
}

CEED_QFUNCTION_HELPER StatePrimitive StatePrimitiveFromConservative(NewtonianIdealGasContext gas, StateConservative U) {
  StatePrimitive Y;
  for (CeedInt i = 0; i < 3; i++) Y.velocity[i] = U.momentum[i] / U.density;
  CeedScalar e_kinetic  = .5 * Dot3(Y.velocity, Y.velocity);
  CeedScalar e_total    = U.E_total / U.density;
  CeedScalar e_internal = e_total - e_kinetic;
  Y.temperature         = e_internal / gas->cv;
  Y.pressure            = (HeatCapacityRatio(gas) - 1) * U.density * e_internal;
  return Y;
}

CEED_QFUNCTION_HELPER StatePrimitive StatePrimitiveFromConservative_fwd(NewtonianIdealGasContext gas, State s, StateConservative dU) {
  StatePrimitive dY;
  for (CeedInt i = 0; i < 3; i++) {
    dY.velocity[i] = (dU.momentum[i] - s.Y.velocity[i] * dU.density) / s.U.density;
  }
  CeedScalar e_kinetic   = .5 * Dot3(s.Y.velocity, s.Y.velocity);
  CeedScalar de_kinetic  = Dot3(dY.velocity, s.Y.velocity);
  CeedScalar e_total     = s.U.E_total / s.U.density;
  CeedScalar de_total    = (dU.E_total - e_total * dU.density) / s.U.density;
  CeedScalar e_internal  = e_total - e_kinetic;
  CeedScalar de_internal = de_total - de_kinetic;
  dY.temperature         = de_internal / gas->cv;
  dY.pressure            = (HeatCapacityRatio(gas) - 1) * (dU.density * e_internal + s.U.density * de_internal);
  return dY;
}

CEED_QFUNCTION_HELPER StateEntropy StateEntropyFromPrimitive(NewtonianIdealGasContext gas, StatePrimitive Y) {
  StateEntropy     V;
  const CeedScalar gamma     = HeatCapacityRatio(gas);
  const CeedScalar rho       = Y.pressure / (GasConstant(gas) * Y.temperature);
  const CeedScalar entropy   = log(Y.pressure) - gamma * log(rho);
  const CeedScalar rho_div_p = rho / Y.pressure;
  const CeedScalar e_kinetic = 0.5 * Dot3(Y.velocity, Y.velocity);

  V.S_density = (gamma - entropy) / (gamma - 1) - rho_div_p * e_kinetic;
  for (int i = 0; i < 3; i++) V.S_momentum[i] = rho_div_p * Y.velocity[i];
  V.S_energy = -rho_div_p;
  return V;
}

CEED_QFUNCTION_HELPER StateEntropy StateEntropyFromPrimitive_fwd(NewtonianIdealGasContext gas, State s, StatePrimitive dY) {
  StateEntropy     dV;
  const CeedScalar gamma = HeatCapacityRatio(gas);
  CeedScalar       drho  = (dY.pressure * s.Y.temperature - s.Y.pressure * dY.temperature) / (GasConstant(gas) * s.Y.temperature * s.Y.temperature);

  const CeedScalar e_kinetic  = .5 * Dot3(s.Y.velocity, s.Y.velocity);
  const CeedScalar de_kinetic = Dot3(dY.velocity, s.Y.velocity);
  const CeedScalar rho_div_p  = s.U.density / s.Y.pressure;
  const CeedScalar drho_div_p = (drho * s.Y.pressure - s.U.density * dY.pressure) / Square(s.Y.pressure);

  CeedScalar dentropy = dY.pressure / s.Y.pressure - gamma * drho / s.U.density;

  dV.S_density = -dentropy / (gamma - 1) - de_kinetic * rho_div_p - e_kinetic * drho_div_p;
  for (CeedInt i = 0; i < 3; i++) dV.S_momentum[i] = rho_div_p * dY.velocity[i] + drho_div_p * s.Y.velocity[i];
  dV.S_energy = -drho_div_p;
  return dV;
}

CEED_QFUNCTION_HELPER StatePrimitive StatePrimitiveFromEntropy(NewtonianIdealGasContext gas, StateEntropy V) {
  StatePrimitive Y;
  for (int i = 0; i < 3; i++) Y.velocity[i] = -V.S_momentum[i] / V.S_energy;
  Y.temperature              = -1 / (GasConstant(gas) * V.S_energy);
  const CeedScalar gamma     = HeatCapacityRatio(gas);
  const CeedScalar e_kinetic = 0.5 * Dot3(Y.velocity, Y.velocity);
  const CeedScalar entropy   = gamma - (gamma - 1) * (V.S_density - e_kinetic * V.S_energy);
  const CeedScalar log_P     = -(entropy + gamma * log(-V.S_energy)) / (gamma - 1);
  Y.pressure                 = exp(log_P);
  return Y;
}

CEED_QFUNCTION_HELPER StatePrimitive StatePrimitiveFromEntropy_fwd(NewtonianIdealGasContext gas, State s, StateEntropy dV) {
  StatePrimitive dY;
  StateEntropy   V = StateEntropyFromPrimitive(gas, s.Y);
  for (int i = 0; i < 3; i++) dY.velocity[i] = -(dV.S_momentum[i] - V.S_momentum[i] * dV.S_energy / V.S_energy) / V.S_energy;
  dY.temperature              = dV.S_energy / (GasConstant(gas) * V.S_energy * V.S_energy);
  const CeedScalar gamma      = HeatCapacityRatio(gas);
  const CeedScalar e_kinetic  = 0.5 * Dot3(s.Y.velocity, s.Y.velocity);
  const CeedScalar de_kinetic = Dot3(dY.velocity, s.Y.velocity);
  const CeedScalar dentropy   = (1 - gamma) * (dV.S_density - e_kinetic * dV.S_energy - de_kinetic * V.S_energy);
  dY.pressure                 = s.Y.pressure * (-dentropy - gamma * dV.S_energy / V.S_energy) / (gamma - 1);
  return dY;
}

CEED_QFUNCTION_HELPER StateConservative StateConservativeFromPrimitive(NewtonianIdealGasContext gas, StatePrimitive Y) {
  StateConservative U;
  U.density = Y.pressure / (GasConstant(gas) * Y.temperature);
  for (int i = 0; i < 3; i++) U.momentum[i] = U.density * Y.velocity[i];
  CeedScalar e_internal = gas->cv * Y.temperature;
  CeedScalar e_kinetic  = .5 * Dot3(Y.velocity, Y.velocity);
  CeedScalar e_total    = e_internal + e_kinetic;
  U.E_total             = U.density * e_total;
  return U;
}

CEED_QFUNCTION_HELPER StateConservative StateConservativeFromPrimitive_fwd(NewtonianIdealGasContext gas, State s, StatePrimitive dY) {
  StateConservative dU;
  dU.density = (dY.pressure * s.Y.temperature - s.Y.pressure * dY.temperature) / (GasConstant(gas) * s.Y.temperature * s.Y.temperature);
  for (int i = 0; i < 3; i++) {
    dU.momentum[i] = dU.density * s.Y.velocity[i] + s.U.density * dY.velocity[i];
  }
  CeedScalar e_kinetic   = .5 * Dot3(s.Y.velocity, s.Y.velocity);
  CeedScalar de_kinetic  = Dot3(dY.velocity, s.Y.velocity);
  CeedScalar e_internal  = gas->cv * s.Y.temperature;
  CeedScalar de_internal = gas->cv * dY.temperature;
  CeedScalar e_total     = e_internal + e_kinetic;
  CeedScalar de_total    = de_internal + de_kinetic;
  dU.E_total             = dU.density * e_total + s.U.density * de_total;
  return dU;
}

CEED_QFUNCTION_HELPER StateEntropy StateEntropyFromConservative(NewtonianIdealGasContext gas, StateConservative U) {
  StateEntropy     V;
  const CeedScalar gamma      = HeatCapacityRatio(gas);
  const CeedScalar e_kinetic  = .5 * Dot3(U.momentum, U.momentum) / U.density;
  const CeedScalar e_internal = U.E_total - e_kinetic;
  const CeedScalar p          = (gamma - 1) * e_internal;
  const CeedScalar entropy    = log(p) - gamma * log(U.density);

  V.S_density = (gamma - entropy) / (gamma - 1) - e_kinetic / p;
  for (int i = 0; i < 3; i++) V.S_momentum[i] = U.momentum[i] / p;
  V.S_energy = -U.density / p;
  return V;
}

CEED_QFUNCTION_HELPER StateEntropy StateEntropyFromConservative_fwd(NewtonianIdealGasContext gas, State s, StateConservative dU) {
  StateEntropy     dV;
  const CeedScalar gamma       = HeatCapacityRatio(gas);
  const CeedScalar e_kinetic   = .5 * Dot3(s.U.momentum, s.U.momentum) / s.U.density;
  const CeedScalar de_kinetic  = (Dot3(s.U.momentum, dU.momentum) - e_kinetic * dU.density) / s.U.density;
  const CeedScalar de_internal = dU.E_total - de_kinetic;
  const CeedScalar p           = s.Y.pressure;
  const CeedScalar dp          = (gamma - 1) * de_internal;

  CeedScalar dentropy = dp / p - gamma * dU.density / s.U.density;

  dV.S_density = -dentropy / (gamma - 1) - de_kinetic / p + dp * e_kinetic / Square(p);
  for (CeedInt i = 0; i < 3; i++) {
    dV.S_momentum[i] = (dU.momentum[i] - s.U.momentum[i] * dp / p) / p;
  }
  dV.S_energy = -(dU.density - s.U.density * dp / p) / p;
  return dV;
}

CEED_QFUNCTION_HELPER StateConservative StateConservativeFromEntropy(NewtonianIdealGasContext gas, StateEntropy V) {
  StateConservative U;
  CeedScalar        velocity[3];
  for (int i = 0; i < 3; i++) velocity[i] = -V.S_momentum[i] / V.S_energy;
  const CeedScalar gamma     = HeatCapacityRatio(gas);
  const CeedScalar e_kinetic = 0.5 * Dot3(velocity, velocity);
  const CeedScalar entropy   = gamma - (gamma - 1) * (V.S_density - e_kinetic * V.S_energy);
  const CeedScalar log_rho   = -(entropy + log(-V.S_energy)) / (gamma - 1);
  U.density                  = exp(log_rho);
  for (int i = 0; i < 3; i++) U.momentum[i] = U.density * velocity[i];

  const CeedScalar e_internal = -gas->cv / (GasConstant(gas) * V.S_energy);
  U.E_total                   = U.density * (e_internal + e_kinetic);
  return U;
}

CEED_QFUNCTION_HELPER StateConservative StateConservativeFromEntropy_fwd(NewtonianIdealGasContext gas, State s, StateEntropy dV) {
  StateConservative dU;
  CeedScalar        dvelocity[3];
  StateEntropy      V = StateEntropyFromPrimitive(gas, s.Y);
  for (int i = 0; i < 3; i++) dvelocity[i] = (-dV.S_momentum[i] - s.Y.velocity[i] * dV.S_energy) / V.S_energy;
  const CeedScalar gamma      = HeatCapacityRatio(gas);
  const CeedScalar e_kinetic  = 0.5 * Dot3(s.Y.velocity, s.Y.velocity);
  const CeedScalar de_kinetic = Dot3(dvelocity, s.Y.velocity);
  const CeedScalar entropy    = gamma - (gamma - 1) * (V.S_density - e_kinetic * V.S_energy);
  const CeedScalar dentropy   = -(gamma - 1) * (dV.S_density - (de_kinetic * V.S_energy + e_kinetic * dV.S_energy));
  const CeedScalar log_rho    = -(entropy + log(-V.S_energy)) / (gamma - 1);
  const CeedScalar rho        = exp(log_rho);
  dU.density                  = -rho / (gamma - 1) * (dentropy + dV.S_energy / V.S_energy);
  for (int i = 0; i < 3; i++) dU.momentum[i] = dU.density * s.Y.velocity[i] + s.U.density * dvelocity[i];

  const CeedScalar e_internal  = -gas->cv / (GasConstant(gas) * V.S_energy);
  const CeedScalar de_internal = gas->cv * dV.S_energy / (GasConstant(gas) * V.S_energy * V.S_energy);
  const CeedScalar e_total     = e_internal + e_kinetic;
  dU.E_total                   = dU.density * e_total + s.U.density * (de_internal + de_kinetic);
  return dU;
}

CEED_QFUNCTION_HELPER State StateFromPrimitive(NewtonianIdealGasContext gas, StatePrimitive Y) {
  StateConservative U = StateConservativeFromPrimitive(gas, Y);
  State             s;
  s.U = U;
  s.Y = Y;
  return s;
}

CEED_QFUNCTION_HELPER State StateFromPrimitive_fwd(NewtonianIdealGasContext gas, State s, StatePrimitive dY) {
  StateConservative dU = StateConservativeFromPrimitive_fwd(gas, s, dY);
  State             ds;
  ds.U = dU;
  ds.Y = dY;
  return ds;
}

// linear combination of n states
CEED_QFUNCTION_HELPER StateConservative StateConservativeMult(CeedInt n, const CeedScalar a[], const StateConservative X[]) {
  StateConservative R = {0};
  for (CeedInt i = 0; i < n; i++) {
    R.density += a[i] * X[i].density;
    for (int j = 0; j < 3; j++) R.momentum[j] += a[i] * X[i].momentum[j];
    R.E_total += a[i] * X[i].E_total;
  }
  return R;
}

CEED_QFUNCTION_HELPER StateConservative StateConservativeAXPBYPCZ(CeedScalar a, StateConservative X, CeedScalar b, StateConservative Y, CeedScalar c,
                                                                  StateConservative Z) {
  StateConservative R;
  R.density = a * X.density + b * Y.density + c * Z.density;
  for (int i = 0; i < 3; i++) R.momentum[i] = a * X.momentum[i] + b * Y.momentum[i] + c * Z.momentum[i];
  R.E_total = a * X.E_total + b * Y.E_total + c * Z.E_total;
  return R;
}

CEED_QFUNCTION_HELPER void StateToU(NewtonianIdealGasContext gas, const State input, CeedScalar U[5]) { UnpackState_U(input.U, U); }

CEED_QFUNCTION_HELPER void StateToY(NewtonianIdealGasContext gas, const State input, CeedScalar Y[5]) { UnpackState_Y(input.Y, Y); }

CEED_QFUNCTION_HELPER void StateToV(NewtonianIdealGasContext gas, const State input, CeedScalar V[5]) {
  StateEntropy state_V = StateEntropyFromPrimitive(gas, input.Y);
  UnpackState_V(state_V, V);
}

CEED_QFUNCTION_HELPER void StateToQ(NewtonianIdealGasContext gas, const State input, CeedScalar Q[5], StateVariable state_var) {
  switch (state_var) {
    case STATEVAR_CONSERVATIVE:
      StateToU(gas, input, Q);
      break;
    case STATEVAR_PRIMITIVE:
      StateToY(gas, input, Q);
      break;
    case STATEVAR_ENTROPY:
      StateToV(gas, input, Q);
      break;
  }
}

CEED_QFUNCTION_HELPER void StateToQ_fwd(NewtonianIdealGasContext gas, const State input, const State dinput, CeedScalar dQ[5],
                                        StateVariable state_var) {
  switch (state_var) {
    case STATEVAR_CONSERVATIVE:
    case STATEVAR_PRIMITIVE:
      StateToQ(gas, dinput, dQ, state_var);
      break;
    case STATEVAR_ENTROPY: {
      StateEntropy dstate_v;

      dstate_v = StateEntropyFromPrimitive_fwd(gas, input, dinput.Y);
      UnpackState_V(dstate_v, dQ);
    } break;
  }
}

CEED_QFUNCTION_HELPER State StateFromU(NewtonianIdealGasContext gas, const CeedScalar U[5]) {
  State s;
  s.U.density     = U[0];
  s.U.momentum[0] = U[1];
  s.U.momentum[1] = U[2];
  s.U.momentum[2] = U[3];
  s.U.E_total     = U[4];
  s.Y             = StatePrimitiveFromConservative(gas, s.U);
  return s;
}

CEED_QFUNCTION_HELPER State StateFromU_fwd(NewtonianIdealGasContext gas, State s, const CeedScalar dU[5]) {
  State ds;
  ds.U.density     = dU[0];
  ds.U.momentum[0] = dU[1];
  ds.U.momentum[1] = dU[2];
  ds.U.momentum[2] = dU[3];
  ds.U.E_total     = dU[4];
  ds.Y             = StatePrimitiveFromConservative_fwd(gas, s, ds.U);
  return ds;
}

CEED_QFUNCTION_HELPER State StateFromY(NewtonianIdealGasContext gas, const CeedScalar Y[5]) {
  State s;
  s.Y.pressure    = Y[0];
  s.Y.velocity[0] = Y[1];
  s.Y.velocity[1] = Y[2];
  s.Y.velocity[2] = Y[3];
  s.Y.temperature = Y[4];
  s.U             = StateConservativeFromPrimitive(gas, s.Y);
  return s;
}

CEED_QFUNCTION_HELPER State StateFromY_fwd(NewtonianIdealGasContext gas, State s, const CeedScalar dY[5]) {
  State ds;
  ds.Y.pressure    = dY[0];
  ds.Y.velocity[0] = dY[1];
  ds.Y.velocity[1] = dY[2];
  ds.Y.velocity[2] = dY[3];
  ds.Y.temperature = dY[4];
  ds.U             = StateConservativeFromPrimitive_fwd(gas, s, ds.Y);
  return ds;
}

CEED_QFUNCTION_HELPER State StateFromV(NewtonianIdealGasContext gas, const CeedScalar V[5]) {
  State        s;
  StateEntropy state_V;
  state_V.S_density     = V[0];
  state_V.S_momentum[0] = V[1];
  state_V.S_momentum[1] = V[2];
  state_V.S_momentum[2] = V[3];
  state_V.S_energy      = V[4];
  s.U                   = StateConservativeFromEntropy(gas, state_V);
  s.Y                   = StatePrimitiveFromEntropy(gas, state_V);
  return s;
}

CEED_QFUNCTION_HELPER State StateFromV_fwd(NewtonianIdealGasContext gas, State s, const CeedScalar dV[5]) {
  State        ds;
  StateEntropy state_dV;
  state_dV.S_density     = dV[0];
  state_dV.S_momentum[0] = dV[1];
  state_dV.S_momentum[1] = dV[2];
  state_dV.S_momentum[2] = dV[3];
  state_dV.S_energy      = dV[4];
  ds.U                   = StateConservativeFromEntropy_fwd(gas, s, state_dV);
  ds.Y                   = StatePrimitiveFromEntropy_fwd(gas, s, state_dV);
  return ds;
}

CEED_QFUNCTION_HELPER State StateFromQ(NewtonianIdealGasContext gas, const CeedScalar Q[5], StateVariable state_var) {
  State s;
  switch (state_var) {
    case STATEVAR_CONSERVATIVE:
      s = StateFromU(gas, Q);
      break;
    case STATEVAR_PRIMITIVE:
      s = StateFromY(gas, Q);
      break;
    case STATEVAR_ENTROPY:
      s = StateFromV(gas, Q);
      break;
  }
  return s;
}

CEED_QFUNCTION_HELPER State StateFromQ_fwd(NewtonianIdealGasContext gas, State s, const CeedScalar dQ[5], StateVariable state_var) {
  State ds;
  switch (state_var) {
    case STATEVAR_CONSERVATIVE:
      ds = StateFromU_fwd(gas, s, dQ);
      break;
    case STATEVAR_PRIMITIVE:
      ds = StateFromY_fwd(gas, s, dQ);
      break;
    case STATEVAR_ENTROPY:
      ds = StateFromV_fwd(gas, s, dQ);
      break;
  }
  return ds;
}

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

CEED_QFUNCTION_HELPER void FluxTotal_RiemannBoundary(const StateConservative F_inviscid_normal, const CeedScalar stress[3][3], const CeedScalar Fe[3],
                                                     const CeedScalar normal[3], CeedScalar Flux[5]) {
  Flux[0] = F_inviscid_normal.density;
  for (CeedInt k = 0; k < 3; k++) Flux[k + 1] = F_inviscid_normal.momentum[k];
  Flux[4] = F_inviscid_normal.E_total;
  for (CeedInt j = 0; j < 3; j++) {
    for (CeedInt k = 0; k < 3; k++) {
      Flux[k + 1] -= stress[k][j] * normal[j];
    }
    Flux[4] += Fe[j] * normal[j];
  }
}

CEED_QFUNCTION_HELPER void VelocityGradient(const State grad_s[3], CeedScalar grad_velocity[3][3]) {
  grad_velocity[0][0] = grad_s[0].Y.velocity[0];
  grad_velocity[0][1] = grad_s[1].Y.velocity[0];
  grad_velocity[0][2] = grad_s[2].Y.velocity[0];
  grad_velocity[1][0] = grad_s[0].Y.velocity[1];
  grad_velocity[1][1] = grad_s[1].Y.velocity[1];
  grad_velocity[1][2] = grad_s[2].Y.velocity[1];
  grad_velocity[2][0] = grad_s[0].Y.velocity[2];
  grad_velocity[2][1] = grad_s[1].Y.velocity[2];
  grad_velocity[2][2] = grad_s[2].Y.velocity[2];
}

CEED_QFUNCTION_HELPER void KMStrainRate(const CeedScalar grad_velocity[3][3], CeedScalar strain_rate[6]) {
  const CeedScalar weight = 1 / sqrt(2.);  // Really sqrt(2.) / 2
  strain_rate[0]          = grad_velocity[0][0];
  strain_rate[1]          = grad_velocity[1][1];
  strain_rate[2]          = grad_velocity[2][2];
  strain_rate[3]          = weight * (grad_velocity[1][2] + grad_velocity[2][1]);
  strain_rate[4]          = weight * (grad_velocity[0][2] + grad_velocity[2][0]);
  strain_rate[5]          = weight * (grad_velocity[0][1] + grad_velocity[1][0]);
}

// Kelvin-Mandel notation
CEED_QFUNCTION_HELPER void KMStrainRate_State(const State grad_s[3], CeedScalar strain_rate[6]) {
  CeedScalar grad_velocity[3][3];
  VelocityGradient(grad_s, grad_velocity);
  KMStrainRate(grad_velocity, strain_rate);
}

//@brief Given velocity gradient du_i/dx_j, return 0.5*(du_i/dx_j - du_j/dx_i)
CEED_QFUNCTION_HELPER void RotationRate(const CeedScalar grad_velocity[3][3], CeedScalar rotation_rate[3][3]) {
  rotation_rate[0][0] = 0;
  rotation_rate[1][1] = 0;
  rotation_rate[2][2] = 0;
  rotation_rate[1][2] = 0.5 * (grad_velocity[1][2] - grad_velocity[2][1]);
  rotation_rate[0][2] = 0.5 * (grad_velocity[0][2] - grad_velocity[2][0]);
  rotation_rate[0][1] = 0.5 * (grad_velocity[0][1] - grad_velocity[1][0]);
  rotation_rate[2][1] = -rotation_rate[1][2];
  rotation_rate[2][0] = -rotation_rate[0][2];
  rotation_rate[1][0] = -rotation_rate[0][1];
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

CEED_QFUNCTION_HELPER void Vorticity(const State grad_s[3], CeedScalar vorticity[3]) {
  CeedScalar grad_velocity[3][3];
  VelocityGradient(grad_s, grad_velocity);
  Curl3(grad_velocity, vorticity);
}

CEED_QFUNCTION_HELPER void StatePhysicalGradientFromReference(CeedInt Q, CeedInt i, NewtonianIdealGasContext gas, State s, StateVariable state_var,
                                                              const CeedScalar *grad_q, const CeedScalar dXdx[3][3], State grad_s[3]) {
  for (CeedInt k = 0; k < 3; k++) {
    CeedScalar dqi[5];
    for (CeedInt j = 0; j < 5; j++) {
      dqi[j] =
          grad_q[(Q * 5) * 0 + Q * j + i] * dXdx[0][k] + grad_q[(Q * 5) * 1 + Q * j + i] * dXdx[1][k] + grad_q[(Q * 5) * 2 + Q * j + i] * dXdx[2][k];
    }
    grad_s[k] = StateFromQ_fwd(gas, s, dqi, state_var);
  }
}

CEED_QFUNCTION_HELPER void StatePhysicalGradientFromReference_Boundary(CeedInt Q, CeedInt i, NewtonianIdealGasContext gas, State s,
                                                                       StateVariable state_var, const CeedScalar *grad_q, const CeedScalar dXdx[2][3],
                                                                       State grad_s[3]) {
  for (CeedInt k = 0; k < 3; k++) {
    CeedScalar dqi[5];
    for (CeedInt j = 0; j < 5; j++) {
      dqi[j] = grad_q[(Q * 5) * 0 + Q * j + i] * dXdx[0][k] + grad_q[(Q * 5) * 1 + Q * j + i] * dXdx[1][k];
    }
    grad_s[k] = StateFromQ_fwd(gas, s, dqi, state_var);
  }
}
