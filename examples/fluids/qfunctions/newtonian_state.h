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

#include <math.h>
#include <ceed.h>
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
  StatePrimitive Y;
} State;

CEED_QFUNCTION_HELPER StatePrimitive StatePrimitiveFromConservative(
  NewtonianIdealGasContext gas, StateConservative U, const CeedScalar x[3]) {
  StatePrimitive Y;
  for (CeedInt i=0; i<3; i++) Y.velocity[i] = U.momentum[i] / U.density;
  CeedScalar e_kinetic = .5 * Dot3(Y.velocity, Y.velocity);
  CeedScalar e_potential = -Dot3(gas->g, x);
  CeedScalar e_total = U.E_total / U.density;
  CeedScalar e_internal = e_total - e_kinetic - e_potential;
  Y.temperature = e_internal / gas->cv;
  Y.pressure = (gas->cp / gas->cv - 1) * U.density * e_internal;
  return Y;
}

CEED_QFUNCTION_HELPER StatePrimitive StatePrimitiveFromConservative_fwd(
  NewtonianIdealGasContext gas, State s, StateConservative dU,
  const CeedScalar x[3], const CeedScalar dx[3]) {
  StatePrimitive dY;
  for (CeedInt i=0; i<3; i++) {
    dY.velocity[i] = (dU.momentum[i] - s.Y.velocity[i] * dU.density) / s.U.density;
  }
  CeedScalar e_kinetic = .5 * Dot3(s.Y.velocity, s.Y.velocity);
  CeedScalar de_kinetic = Dot3(dY.velocity, s.Y.velocity);
  CeedScalar e_potential = -Dot3(gas->g, x);
  CeedScalar de_potential = -Dot3(gas->g, dx);
  CeedScalar e_total = s.U.E_total / s.U.density;
  CeedScalar de_total = (dU.E_total - e_total * dU.density) / s.U.density;
  CeedScalar e_internal = e_total - e_kinetic - e_potential;
  CeedScalar de_internal = de_total - de_kinetic - de_potential;
  dY.temperature = de_internal / gas->cv;
  dY.pressure = (gas->cp / gas->cv - 1)
                * (dU.density * e_internal + s.U.density * de_internal);
  return dY;
}

CEED_QFUNCTION_HELPER StateConservative StateConservativeFromPrimitive(
  NewtonianIdealGasContext gas, StatePrimitive Y, const CeedScalar x[3]) {
  StateConservative U;
  CeedScalar R = gas->cp - gas->cv;
  U.density = Y.pressure / (R * Y.temperature);
  for (int i=0; i<3; i++) U.momentum[i] = U.density*Y.velocity[i];
  CeedScalar e_internal = gas->cv * Y.temperature;
  CeedScalar e_kinetic = .5 * Dot3(Y.velocity, Y.velocity);
  CeedScalar e_potential = -Dot3(gas->g, x);
  CeedScalar e_total = e_internal + e_kinetic + e_potential;
  U.E_total = U.density*e_total;
  return U;
}

CEED_QFUNCTION_HELPER StateConservative StateConservativeFromPrimitive_fwd(
  NewtonianIdealGasContext gas, State s, StatePrimitive dY,
  const CeedScalar x[3], const CeedScalar dx[3]) {
  StateConservative dU;
  CeedScalar R = gas->cp - gas->cv;
  dU.density = (dY.pressure * s.Y.temperature - s.Y.pressure * dY.temperature) /
               (R * s.Y.temperature * s.Y.temperature);
  for (int i=0; i<3; i++) {
    dU.momentum[i] = dU.density * s.Y.velocity[i] + s.U.density * dY.velocity[i];
  }
  CeedScalar e_kinetic = .5 * Dot3(s.Y.velocity, s.Y.velocity);
  CeedScalar de_kinetic = Dot3(dY.velocity, s.Y.velocity);
  CeedScalar e_potential = -Dot3(gas->g, x);
  CeedScalar de_potential = -Dot3(gas->g, dx);
  CeedScalar e_internal = gas->cv * s.Y.temperature;
  CeedScalar de_internal = gas->cv * dY.temperature;
  CeedScalar e_total = e_internal + e_kinetic + e_potential;
  CeedScalar de_total = de_internal + de_kinetic + de_potential;
  dU.E_total = dU.density*e_total + s.U.density*de_total;
  return dU;
}

CEED_QFUNCTION_HELPER State StateFromU(NewtonianIdealGasContext gas,
                                       const CeedScalar U[5], const CeedScalar x[3]) {
  State s;
  s.U.density = U[0];
  s.U.momentum[0] = U[1];
  s.U.momentum[1] = U[2];
  s.U.momentum[2] = U[3];
  s.U.E_total = U[4];
  s.Y = StatePrimitiveFromConservative(gas, s.U, x);
  return s;
}

CEED_QFUNCTION_HELPER State StateFromU_fwd(NewtonianIdealGasContext gas,
    State s, const CeedScalar dU[5],
    const CeedScalar x[3], const CeedScalar dx[3]) {
  State ds;
  ds.U.density = dU[0];
  ds.U.momentum[0] = dU[1];
  ds.U.momentum[1] = dU[2];
  ds.U.momentum[2] = dU[3];
  ds.U.E_total = dU[4];
  ds.Y = StatePrimitiveFromConservative_fwd(gas, s, ds.U, x, dx);
  return ds;
}

CEED_QFUNCTION_HELPER State StateFromY(NewtonianIdealGasContext gas,
                                       const CeedScalar Y[5], const CeedScalar x[3]) {
  State s;
  s.Y.pressure    = Y[0];
  s.Y.velocity[0] = Y[1];
  s.Y.velocity[1] = Y[2];
  s.Y.velocity[2] = Y[3];
  s.Y.temperature = Y[4];
  s.U = StateConservativeFromPrimitive(gas, s.Y, x);
  return s;
}

CEED_QFUNCTION_HELPER State StateFromY_fwd(NewtonianIdealGasContext gas,
    State s, const CeedScalar dY[5],
    const CeedScalar x[3], const CeedScalar dx[3]) {
  State ds;
  ds.Y.pressure    = dY[0];
  ds.Y.velocity[0] = dY[1];
  ds.Y.velocity[1] = dY[2];
  ds.Y.velocity[2] = dY[3];
  ds.Y.temperature = dY[4];
  ds.U = StateConservativeFromPrimitive_fwd(gas, s, ds.Y, x, dx);
  return ds;
}

CEED_QFUNCTION_HELPER void FluxInviscid(NewtonianIdealGasContext gas, State s,
                                        StateConservative Flux[3]) {
  for (CeedInt i=0; i<3; i++) {
    Flux[i].density = s.U.momentum[i];
    for (CeedInt j=0; j<3; j++)
      Flux[i].momentum[j] = s.U.momentum[i] * s.Y.velocity[j]
                            + s.Y.pressure * (i == j);
    Flux[i].E_total = (s.U.E_total + s.Y.pressure) * s.Y.velocity[i];
  }
}

CEED_QFUNCTION_HELPER void FluxInviscid_fwd(NewtonianIdealGasContext gas,
    State s, State ds, StateConservative dFlux[3]) {
  for (CeedInt i=0; i<3; i++) {
    dFlux[i].density = ds.U.momentum[i];
    for (CeedInt j=0; j<3; j++)
      dFlux[i].momentum[j] = ds.U.momentum[i] * s.Y.velocity[j] +
                             s.U.momentum[i] * ds.Y.velocity[j] + ds.Y.pressure * (i == j);
    dFlux[i].E_total = (ds.U.E_total + ds.Y.pressure) * s.Y.velocity[i] +
                       (s.U.E_total + s.Y.pressure) * ds.Y.velocity[i];
  }
}

// Kelvin-Mandel notation
CEED_QFUNCTION_HELPER void KMStrainRate(const State grad_s[3],
                                        CeedScalar strain_rate[6]) {
  const CeedScalar weight = 1 / sqrt(2.);
  strain_rate[0] = grad_s[0].Y.velocity[0];
  strain_rate[1] = grad_s[1].Y.velocity[1];
  strain_rate[2] = grad_s[2].Y.velocity[2];
  strain_rate[3] = weight * (grad_s[2].Y.velocity[1] + grad_s[1].Y.velocity[2]);
  strain_rate[4] = weight * (grad_s[2].Y.velocity[0] + grad_s[0].Y.velocity[2]);
  strain_rate[5] = weight * (grad_s[1].Y.velocity[0] + grad_s[0].Y.velocity[1]);
}

CEED_QFUNCTION_HELPER void NewtonianStress(NewtonianIdealGasContext gas,
    const CeedScalar strain_rate[6], CeedScalar stress[6]) {
  CeedScalar div_u = strain_rate[0] + strain_rate[1] + strain_rate[2];
  for (CeedInt i=0; i<6; i++) {
    stress[i] = gas->mu * (2 * strain_rate[i] + gas->lambda * div_u * (i < 3));
  }
}

CEED_QFUNCTION_HELPER void ViscousEnergyFlux(NewtonianIdealGasContext gas,
    StatePrimitive Y, const State grad_s[3], const CeedScalar stress[3][3],
    CeedScalar Fe[3]) {
  for (CeedInt i=0; i<3; i++) {
    Fe[i] = - Y.velocity[0] * stress[0][i]
            - Y.velocity[1] * stress[1][i]
            - Y.velocity[2] * stress[2][i]
            - gas->k * grad_s[i].Y.temperature;
  }
}

CEED_QFUNCTION_HELPER void ViscousEnergyFlux_fwd(NewtonianIdealGasContext gas,
    StatePrimitive Y, StatePrimitive dY, const State grad_ds[3],
    const CeedScalar stress[3][3],
    const CeedScalar dstress[3][3],
    CeedScalar dFe[3]) {
  for (CeedInt i=0; i<3; i++) {
    dFe[i] = - Y.velocity[0] * dstress[0][i] - dY.velocity[0] * stress[0][i]
             - Y.velocity[1] * dstress[1][i] - dY.velocity[1] * stress[1][i]
             - Y.velocity[2] * dstress[2][i] - dY.velocity[2] * stress[2][i]
             - gas->k * grad_ds[i].Y.temperature;
  }
}

#endif // newtonian_state_h
