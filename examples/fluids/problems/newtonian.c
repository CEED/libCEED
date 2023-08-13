// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up problems using the Newtonian Qfunction

#include "../qfunctions/newtonian.h"

#include <ceed.h>
#include <petscdm.h>

#include "../navierstokes.h"
#include "../qfunctions/setupgeo.h"

// For use with PetscOptionsEnum
static const char *const StateVariables[] = {"CONSERVATIVE", "PRIMITIVE", "ENTROPY", "StateVariable", "STATEVAR_", NULL};

// Compute relative error |a - b|/|s|
static PetscErrorCode CheckConservativeWithTolerance(StateConservative sU, StateConservative aU, StateConservative bU, const char *name,
                                                     PetscReal rtol_density, PetscReal rtol_momentum, PetscReal rtol_E_total) {
  PetscFunctionBeginUser;
  StateConservative eU;  // relative error
  eU.density    = (aU.density - bU.density) / sU.density;
  PetscScalar u = sqrt(Square(sU.momentum[0]) + Square(sU.momentum[1]) + Square(sU.momentum[2]));
  for (int j = 0; j < 3; j++) eU.momentum[j] = (aU.momentum[j] - bU.momentum[j]) / u;
  eU.E_total = (aU.E_total - bU.E_total) / sU.E_total;
  if (fabs(eU.density) > rtol_density) printf("%s: density error %g (expected %g, got %g)\n", name, eU.density, sU.density, aU.density);
  for (int j = 0; j < 3; j++) {
    if (fabs(eU.momentum[j]) > rtol_momentum) {
      printf("%s: momentum[%d] error %g (expected %g, got %g)\n", name, j, eU.momentum[j], sU.momentum[j], aU.momentum[j]);
    }
  }
  if (fabs(eU.E_total) > rtol_E_total) printf("%s: E_total error %g (expected %g, got %g)\n", name, eU.E_total, sU.E_total, aU.E_total);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckPrimitiveWithTolerance(StatePrimitive sY, StatePrimitive aY, StatePrimitive bY, const char *name, PetscReal rtol_pressure,
                                                  PetscReal rtol_velocity, PetscReal rtol_temperature) {
  PetscFunctionBeginUser;
  StatePrimitive eY;  // relative error
  eY.pressure   = (aY.pressure - bY.pressure) / sY.pressure;
  PetscScalar u = sqrt(Square(sY.velocity[0]) + Square(sY.velocity[1]) + Square(sY.velocity[2]));
  for (int j = 0; j < 3; j++) eY.velocity[j] = (aY.velocity[j] - bY.velocity[j]) / u;
  eY.temperature = (aY.temperature - bY.temperature) / sY.temperature;
  if (fabs(eY.pressure) > rtol_pressure) printf("%s: pressure error %g (expected %g, got %g)\n", name, eY.pressure, sY.pressure, aY.pressure);
  for (int j = 0; j < 3; j++) {
    if (fabs(eY.velocity[j]) > rtol_velocity) {
      printf("%s: velocity[%d] error %g (expected %g, got %g)\n", name, j, eY.velocity[j], sY.velocity[j], aY.velocity[j]);
    }
  }
  if (fabs(eY.temperature) > rtol_temperature)
    printf("%s: temperature error %g (expected %g, got %g)\n", name, eY.temperature, sY.temperature, aY.temperature);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckEntropyWithTolerance(StateEntropy sV, StateEntropy aV, StateEntropy bV, const char *name, PetscReal rtol_v1,
                                                PetscReal rtol_velocity, PetscReal rtol_temperature_inv) {
  PetscFunctionBeginUser;
  StateEntropy eV;  // relative error
  eV.S_density  = (aV.S_density - bV.S_density) / (fabs(sV.S_density) > 1e-16 ? sV.S_density : 1);
  PetscScalar u = sqrt(Square(sV.S_momentum[0]) + Square(sV.S_momentum[1]) + Square(sV.S_momentum[2]));
  for (int j = 0; j < 3; j++) eV.S_momentum[j] = (aV.S_momentum[j] - bV.S_momentum[j]) / (fabs(u) > 1e-16 ? u : 1);
  eV.S_energy = (aV.S_energy - bV.S_energy) / (fabs(sV.S_energy) > 1e-16 ? sV.S_energy : 1);
  if (fabs(eV.S_density) > rtol_v1) {
    printf("%s: V1 error %g (expected %g, got %g)\n", name, eV.S_density, sV.S_density, aV.S_density);
  }
  for (int j = 0; j < 3; j++) {
    if (fabs(eV.S_momentum[j]) > rtol_velocity) {
      printf("%s: S_momentum[%d] error %g (expected %g, got %g)\n", name, j, eV.S_momentum[j], sV.S_momentum[j], aV.S_momentum[j]);
    }
  }
  if (fabs(eV.S_energy) > rtol_temperature_inv) {
    printf("%s: S_energy error %g (expected %g, got %g)\n", name, eV.S_energy, sV.S_energy, aV.S_energy);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UnitTests_Newtonian(User user, NewtonianIdealGasContext gas) {
  Units            units = user->units;
  const CeedScalar eps   = 1e-6;
  const CeedScalar rtol  = 1e-4;
  const CeedScalar kg = units->kilogram, m = units->meter, sec = units->second, K = units->Kelvin;
  PetscFunctionBeginUser;
  const CeedScalar T   = 200 * K;
  const CeedScalar rho = 1.2 * kg / Cube(m), u_base = 40 * m / sec;
  const CeedScalar P           = (HeatCapacityRatio(gas) - 1) * rho * gas->cv * T;
  const CeedScalar u[3]        = {u_base, u_base * 1.1, u_base * 1.2};
  const CeedScalar e_kinetic   = 0.5 * Dot3(u, u);
  const CeedScalar x[3]        = {3 * m, 4 * m, 5 * m};
  const CeedScalar e_potential = -Dot3(gas->g, x);
  const CeedScalar e_internal  = gas->cv * T;
  const CeedScalar e_total     = e_kinetic + e_potential + e_internal;
  {
    const CeedScalar U[5] = {rho, rho * u[0], rho * u[1], rho * u[2], rho * e_total};
    const State      s    = StateFromU(gas, U, x);
    for (int i = 0; i < 8; i++) {
      CeedScalar dU[5] = {0}, dx[3] = {0};
      if (i < 5) dU[i] = U[i];
      else dx[i - 5] = x[i - 5];
      State ds = StateFromU_fwd(gas, s, dU, x, dx);
      for (int j = 0; j < 5; j++) dU[j] = (1 + eps * (i == j)) * U[j];
      for (int j = 0; j < 3; j++) dx[j] = (1 + eps * (i == 5 + j)) * x[j];
      State          t = StateFromU(gas, dU, dx);
      StatePrimitive dY;
      dY.pressure = (t.Y.pressure - s.Y.pressure) / eps;
      for (int j = 0; j < 3; j++) dY.velocity[j] = (t.Y.velocity[j] - s.Y.velocity[j]) / eps;
      dY.temperature = (t.Y.temperature - s.Y.temperature) / eps;
      char buf[128];
      snprintf(buf, sizeof buf, "U->Y: StateFromU_fwd i=%d", i);
      PetscCall(CheckPrimitiveWithTolerance(dY, ds.Y, dY, buf, rtol, rtol, rtol));
    }
    for (int i = 0; i < 8; i++) {
      CeedScalar dU[5] = {0}, dx[3] = {0};
      if (i < 5) dU[i] = U[i];
      else dx[i - 5] = x[i - 5];
      State ds = StateFromU_fwd(gas, s, dU, x, dx);
      for (int j = 0; j < 5; j++) dU[j] = (1 + eps * (i == j)) * U[j];
      for (int j = 0; j < 3; j++) dx[j] = (1 + eps * (i == 5 + j)) * x[j];
      State        t = StateFromU(gas, dU, dx);
      StateEntropy dV;
      dV.S_density = (t.V.S_density - s.V.S_density) / eps;
      for (int j = 0; j < 3; j++) dV.S_momentum[j] = (t.V.S_momentum[j] - s.V.S_momentum[j]) / eps;
      dV.S_energy = (t.V.S_energy - s.V.S_energy) / eps;
      char buf[128];
      snprintf(buf, sizeof buf, "U->V: StateFromU_fwd i=%d", i);
      PetscCall(CheckEntropyWithTolerance(dV, ds.V, dV, buf, 5 * rtol, rtol, rtol));
    }
  }
  {
    CeedScalar  Y[5] = {P, u[0], u[1], u[2], T};
    const State s    = StateFromY(gas, Y, x);
    for (int i = 0; i < 8; i++) {
      CeedScalar dY[5] = {0}, dx[3] = {0};
      if (i < 5) dY[i] = Y[i];
      else dx[i - 5] = x[i - 5];
      State ds = StateFromY_fwd(gas, s, dY, x, dx);
      for (int j = 0; j < 5; j++) dY[j] = (1 + eps * (i == j)) * Y[j];
      for (int j = 0; j < 3; j++) dx[j] = (1 + eps * (i == 5 + j)) * x[j];
      State        t = StateFromY(gas, dY, dx);
      StateEntropy dV;
      dV.S_density = (t.V.S_density - s.V.S_density) / eps;
      for (int j = 0; j < 3; j++) dV.S_momentum[j] = (t.V.S_momentum[j] - s.V.S_momentum[j]) / eps;
      dV.S_energy = (t.V.S_energy - s.V.S_energy) / eps;
      char buf[128];
      snprintf(buf, sizeof buf, "Y->V: StateFromY_fwd i=%d", i);
      PetscCall(CheckEntropyWithTolerance(dV, ds.V, dV, buf, rtol, rtol, rtol));
    }
    for (int i = 0; i < 8; i++) {
      CeedScalar dY[5] = {0}, dx[3] = {0};
      if (i < 5) dY[i] = Y[i];
      else dx[i - 5] = x[i - 5];
      State ds = StateFromY_fwd(gas, s, dY, x, dx);
      for (int j = 0; j < 5; j++) dY[j] = (1 + eps * (i == j)) * Y[j];
      for (int j = 0; j < 3; j++) dx[j] = (1 + eps * (i == 5 + j)) * x[j];
      State             t = StateFromY(gas, dY, dx);
      StateConservative dU;
      dU.density = (t.U.density - s.U.density) / eps;
      for (int j = 0; j < 3; j++) dU.momentum[j] = (t.U.momentum[j] - s.U.momentum[j]) / eps;
      dU.E_total = (t.U.E_total - s.U.E_total) / eps;
      char buf[128];
      snprintf(buf, sizeof buf, "Y->U: StateFromY_fwd i=%d", i);
      PetscCall(CheckConservativeWithTolerance(dU, ds.U, dU, buf, rtol, rtol, rtol));
    }
  }
  {
    const CeedScalar gamma     = HeatCapacityRatio(gas);
    const CeedScalar entropy   = log(P) - gamma * log(rho);
    const CeedScalar rho_div_p = rho / P;

    const CeedScalar V[5] = {(gamma - entropy) / (gamma - 1) - rho_div_p * (e_kinetic + e_potential), rho_div_p * u[0], rho_div_p * u[1],
                             rho_div_p * u[2], -rho_div_p};
    const State      s    = StateFromV(gas, V, x);
    for (int i = 0; i < 8; i++) {
      CeedScalar dV[5] = {0}, dx[3] = {0};
      if (i < 5) dV[i] = V[i];
      else dx[i - 5] = x[i - 5];
      State ds = StateFromV_fwd(gas, s, dV, x, dx);
      for (int j = 0; j < 5; j++) dV[j] = (1 + eps * (i == j)) * V[j];
      for (int j = 0; j < 3; j++) dx[j] = (1 + eps * (i == 5 + j)) * x[j];
      State          t = StateFromV(gas, dV, dx);
      StatePrimitive dY;
      dY.pressure = (t.Y.pressure - s.Y.pressure) / eps;
      for (int j = 0; j < 3; j++) dY.velocity[j] = (t.Y.velocity[j] - s.Y.velocity[j]) / eps;
      dY.temperature = (t.Y.temperature - s.Y.temperature) / eps;
      char buf[128];
      snprintf(buf, sizeof buf, "V->Y: StateFromV_fwd i=%d", i);
      PetscCall(CheckPrimitiveWithTolerance(dY, ds.Y, dY, buf, rtol, rtol, rtol));
    }
    for (int i = 0; i < 8; i++) {
      CeedScalar dV[5] = {0}, dx[3] = {0};
      if (i < 5) dV[i] = V[i];
      else dx[i - 5] = x[i - 5];
      State ds = StateFromV_fwd(gas, s, dV, x, dx);
      for (int j = 0; j < 5; j++) dV[j] = (1 + eps * (i == j)) * V[j];
      for (int j = 0; j < 3; j++) dx[j] = (1 + eps * (i == 5 + j)) * x[j];
      State             t = StateFromV(gas, dV, dx);
      StateConservative dU;
      dU.density = (t.U.density - s.U.density) / eps;
      for (int j = 0; j < 3; j++) dU.momentum[j] = (t.U.momentum[j] - s.U.momentum[j]) / eps;
      dU.E_total = (t.U.E_total - s.U.E_total) / eps;
      char buf[128];
      snprintf(buf, sizeof buf, "V->U: StateFromV_fwd i=%d", i);
      PetscCall(CheckConservativeWithTolerance(dU, ds.U, dU, buf, rtol, rtol, rtol));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NS_NEWTONIAN_IG(ProblemData *problem, DM dm, void *ctx, SimpleBC bc) {
  SetupContext             setup_context;
  User                     user   = *(User *)ctx;
  CeedInt                  degree = user->app_ctx->degree;
  StabilizationType        stab;
  StateVariable            state_var;
  MPI_Comm                 comm = user->comm;
  Ceed                     ceed = user->ceed;
  PetscBool                implicit;
  PetscBool                has_curr_time = PETSC_FALSE, unit_tests;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext     newtonian_ig_context;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(1, &setup_context));
  PetscCall(PetscCalloc1(1, &newtonian_ig_ctx));

  // ------------------------------------------------------
  //           Setup Generic Newtonian IG Problem
  // ------------------------------------------------------
  problem->dim                     = 3;
  problem->q_data_size_vol         = 10;
  problem->q_data_size_sur         = 10;
  problem->jac_data_size_sur       = 11;
  problem->setup_vol.qfunction     = Setup;
  problem->setup_vol.qfunction_loc = Setup_loc;
  problem->setup_sur.qfunction     = SetupBoundary;
  problem->setup_sur.qfunction_loc = SetupBoundary_loc;
  problem->non_zero_time           = PETSC_FALSE;
  problem->print_info              = PRINT_NEWTONIAN;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar cv         = 717.;           // J/(kg K)
  CeedScalar cp         = 1004.;          // J/(kg K)
  CeedScalar g[3]       = {0, 0, -9.81};  // m/s^2
  CeedScalar lambda     = -2. / 3.;       // -
  CeedScalar mu         = 1.8e-5;         // Pa s, dynamic viscosity
  CeedScalar k          = 0.02638;        // W/(m K)
  CeedScalar c_tau      = 0.5 / degree;   // -
  CeedScalar Ctau_t     = 1.0;            // -
  CeedScalar Cv_func[3] = {36, 60, 128};
  CeedScalar Ctau_v     = Cv_func[(CeedInt)Min(3, degree) - 1];
  CeedScalar Ctau_C     = 0.25 / degree;
  CeedScalar Ctau_M     = 0.25 / degree;
  CeedScalar Ctau_E     = 0.125;
  PetscReal  domain_min[3], domain_max[3], domain_size[3];
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  StatePrimitive reference      = {.pressure = 1.01e5, .velocity = {0}, .temperature = 288.15};
  CeedScalar     idl_decay_time = -1, idl_start = 0, idl_length = 0;
  PetscBool      idl_enable = PETSC_FALSE;

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter    = 1;  // 1 meter in scaled length units
  PetscScalar kilogram = 1;  // 1 kilogram in scaled mass units
  PetscScalar second   = 1;  // 1 second in scaled time units
  PetscScalar Kelvin   = 1;  // 1 Kelvin in scaled temperature units
  PetscScalar W_per_m_K, Pascal, J_per_kg_K, m_per_squared_s;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for Newtonian Ideal Gas based problem", NULL);
  // -- Conservative vs Primitive variables
  PetscCall(PetscOptionsEnum("-state_var", "State variables used", NULL, StateVariables, (PetscEnum)(state_var = STATEVAR_CONSERVATIVE),
                             (PetscEnum *)&state_var, NULL));

  switch (state_var) {
    case STATEVAR_CONSERVATIVE:
      problem->ics.qfunction                       = ICsNewtonianIG_Conserv;
      problem->ics.qfunction_loc                   = ICsNewtonianIG_Conserv_loc;
      problem->apply_vol_rhs.qfunction             = RHSFunction_Newtonian;
      problem->apply_vol_rhs.qfunction_loc         = RHSFunction_Newtonian_loc;
      problem->apply_vol_ifunction.qfunction       = IFunction_Newtonian_Conserv;
      problem->apply_vol_ifunction.qfunction_loc   = IFunction_Newtonian_Conserv_loc;
      problem->apply_vol_ijacobian.qfunction       = IJacobian_Newtonian_Conserv;
      problem->apply_vol_ijacobian.qfunction_loc   = IJacobian_Newtonian_Conserv_loc;
      problem->apply_inflow.qfunction              = BoundaryIntegral_Conserv;
      problem->apply_inflow.qfunction_loc          = BoundaryIntegral_Conserv_loc;
      problem->apply_inflow_jacobian.qfunction     = BoundaryIntegral_Jacobian_Conserv;
      problem->apply_inflow_jacobian.qfunction_loc = BoundaryIntegral_Jacobian_Conserv_loc;
      break;

    case STATEVAR_PRIMITIVE:
      problem->ics.qfunction                       = ICsNewtonianIG_Prim;
      problem->ics.qfunction_loc                   = ICsNewtonianIG_Prim_loc;
      problem->apply_vol_ifunction.qfunction       = IFunction_Newtonian_Prim;
      problem->apply_vol_ifunction.qfunction_loc   = IFunction_Newtonian_Prim_loc;
      problem->apply_vol_ijacobian.qfunction       = IJacobian_Newtonian_Prim;
      problem->apply_vol_ijacobian.qfunction_loc   = IJacobian_Newtonian_Prim_loc;
      problem->apply_inflow.qfunction              = BoundaryIntegral_Prim;
      problem->apply_inflow.qfunction_loc          = BoundaryIntegral_Prim_loc;
      problem->apply_inflow_jacobian.qfunction     = BoundaryIntegral_Jacobian_Prim;
      problem->apply_inflow_jacobian.qfunction_loc = BoundaryIntegral_Jacobian_Prim_loc;
      break;

    case STATEVAR_ENTROPY:
      problem->ics.qfunction                       = ICsNewtonianIG_Entropy;
      problem->ics.qfunction_loc                   = ICsNewtonianIG_Entropy_loc;
      problem->apply_vol_ifunction.qfunction       = IFunction_Newtonian_Entropy;
      problem->apply_vol_ifunction.qfunction_loc   = IFunction_Newtonian_Entropy_loc;
      problem->apply_vol_ijacobian.qfunction       = IJacobian_Newtonian_Entropy;
      problem->apply_vol_ijacobian.qfunction_loc   = IJacobian_Newtonian_Entropy_loc;
      problem->apply_inflow.qfunction              = BoundaryIntegral_Entropy;
      problem->apply_inflow.qfunction_loc          = BoundaryIntegral_Entropy_loc;
      problem->apply_inflow_jacobian.qfunction     = BoundaryIntegral_Jacobian_Entropy;
      problem->apply_inflow_jacobian.qfunction_loc = BoundaryIntegral_Jacobian_Entropy_loc;
      break;
  }

  // -- Physics
  PetscCall(PetscOptionsScalar("-cv", "Heat capacity at constant volume", NULL, cv, &cv, NULL));
  PetscCall(PetscOptionsScalar("-cp", "Heat capacity at constant pressure", NULL, cp, &cp, NULL));
  PetscCall(PetscOptionsScalar("-lambda", "Stokes hypothesis second viscosity coefficient", NULL, lambda, &lambda, NULL));
  PetscCall(PetscOptionsScalar("-mu", "Shear dynamic viscosity coefficient", NULL, mu, &mu, NULL));
  PetscCall(PetscOptionsScalar("-k", "Thermal conductivity", NULL, k, &k, NULL));

  PetscInt dim = problem->dim;
  PetscCall(PetscOptionsRealArray("-g", "Gravitational acceleration", NULL, g, &dim, NULL));
  PetscCall(PetscOptionsEnum("-stab", "Stabilization method", NULL, StabilizationTypes, (PetscEnum)(stab = STAB_NONE), (PetscEnum *)&stab, NULL));
  PetscCall(PetscOptionsScalar("-c_tau", "Stabilization constant", NULL, c_tau, &c_tau, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_t", "Stabilization time constant", NULL, Ctau_t, &Ctau_t, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_v", "Stabilization viscous constant", NULL, Ctau_v, &Ctau_v, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_C", "Stabilization continuity constant", NULL, Ctau_C, &Ctau_C, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_M", "Stabilization momentum constant", NULL, Ctau_M, &Ctau_M, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_E", "Stabilization energy constant", NULL, Ctau_E, &Ctau_E, NULL));
  PetscCall(PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation", NULL, implicit = PETSC_FALSE, &implicit, NULL));
  PetscCall(PetscOptionsBool("-newtonian_unit_tests", "Run Newtonian unit tests", NULL, unit_tests = PETSC_FALSE, &unit_tests, NULL));

  dim = 3;
  PetscCall(PetscOptionsScalar("-reference_pressure", "Reference/initial pressure", NULL, reference.pressure, &reference.pressure, NULL));
  PetscCall(PetscOptionsScalarArray("-reference_velocity", "Reference/initial velocity", NULL, reference.velocity, &dim, NULL));
  PetscCall(PetscOptionsScalar("-reference_temperature", "Reference/initial temperature", NULL, reference.temperature, &reference.temperature, NULL));

  // -- Units
  PetscCall(PetscOptionsScalar("-units_meter", "1 meter in scaled length units", NULL, meter, &meter, NULL));
  meter = fabs(meter);
  PetscCall(PetscOptionsScalar("-units_kilogram", "1 kilogram in scaled mass units", NULL, kilogram, &kilogram, NULL));
  kilogram = fabs(kilogram);
  PetscCall(PetscOptionsScalar("-units_second", "1 second in scaled time units", NULL, second, &second, NULL));
  second = fabs(second);
  PetscCall(PetscOptionsScalar("-units_Kelvin", "1 Kelvin in scaled temperature units", NULL, Kelvin, &Kelvin, NULL));
  Kelvin = fabs(Kelvin);

  // -- Warnings
  if (stab == STAB_SUPG && !implicit) {
    PetscCall(PetscPrintf(comm, "Warning! Use -stab supg only with -implicit\n"));
  }
  PetscCheck(!(state_var == STATEVAR_PRIMITIVE && !implicit), comm, PETSC_ERR_SUP,
             "RHSFunction is not provided for primitive variables (use -state_var primitive only with -implicit)\n");

  PetscCall(PetscOptionsScalar("-idl_decay_time", "Characteristic timescale of the pressure deviance decay. The timestep is good starting point",
                               NULL, idl_decay_time, &idl_decay_time, &idl_enable));
  if (idl_enable && idl_decay_time == 0) SETERRQ(comm, PETSC_ERR_SUP, "idl_decay_time may not be equal to zero.");
  else if (idl_decay_time < 0) idl_enable = PETSC_FALSE;
  PetscCall(PetscOptionsScalar("-idl_start", "Start of IDL in the x direction", NULL, idl_start, &idl_start, NULL));
  PetscCall(PetscOptionsScalar("-idl_length", "Length of IDL in the positive x direction", NULL, idl_length, &idl_length, NULL));
  PetscOptionsEnd();

  // ------------------------------------------------------
  //           Set up the PETSc context
  // ------------------------------------------------------
  // -- Define derived units
  Pascal          = kilogram / (meter * PetscSqr(second));
  J_per_kg_K      = PetscSqr(meter) / (PetscSqr(second) * Kelvin);
  m_per_squared_s = meter / PetscSqr(second);
  W_per_m_K       = kilogram * meter / (pow(second, 3) * Kelvin);

  user->units->meter           = meter;
  user->units->kilogram        = kilogram;
  user->units->second          = second;
  user->units->Kelvin          = Kelvin;
  user->units->Pascal          = Pascal;
  user->units->J_per_kg_K      = J_per_kg_K;
  user->units->m_per_squared_s = m_per_squared_s;
  user->units->W_per_m_K       = W_per_m_K;

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  cv *= J_per_kg_K;
  cp *= J_per_kg_K;
  mu *= Pascal * second;
  k *= W_per_m_K;
  for (PetscInt i = 0; i < 3; i++) domain_size[i] *= meter;
  for (PetscInt i = 0; i < 3; i++) g[i] *= m_per_squared_s;
  reference.pressure *= Pascal;
  for (PetscInt i = 0; i < 3; i++) reference.velocity[i] *= meter / second;
  reference.temperature *= Kelvin;
  problem->dm_scale = meter;

  // -- Solver Settings
  user->phys->stab          = stab;
  user->phys->implicit      = implicit;
  user->phys->state_var     = state_var;
  user->phys->has_curr_time = has_curr_time;

  // -- QFunction Context
  newtonian_ig_ctx->lambda        = lambda;
  newtonian_ig_ctx->mu            = mu;
  newtonian_ig_ctx->k             = k;
  newtonian_ig_ctx->cv            = cv;
  newtonian_ig_ctx->cp            = cp;
  newtonian_ig_ctx->c_tau         = c_tau;
  newtonian_ig_ctx->Ctau_t        = Ctau_t;
  newtonian_ig_ctx->Ctau_v        = Ctau_v;
  newtonian_ig_ctx->Ctau_C        = Ctau_C;
  newtonian_ig_ctx->Ctau_M        = Ctau_M;
  newtonian_ig_ctx->Ctau_E        = Ctau_E;
  newtonian_ig_ctx->P0            = reference.pressure;
  newtonian_ig_ctx->stabilization = stab;
  newtonian_ig_ctx->P0            = reference.pressure;
  newtonian_ig_ctx->is_implicit   = implicit;
  newtonian_ig_ctx->state_var     = state_var;
  newtonian_ig_ctx->idl_enable    = idl_enable;
  newtonian_ig_ctx->idl_amplitude = 1 / (idl_decay_time * second);
  newtonian_ig_ctx->idl_start     = idl_start * meter;
  newtonian_ig_ctx->idl_length    = idl_length * meter;
  PetscCall(PetscArraycpy(newtonian_ig_ctx->g, g, 3));

  // -- Setup Context
  setup_context->reference = reference;
  setup_context->gas       = *newtonian_ig_ctx;
  setup_context->lx        = domain_size[0];
  setup_context->ly        = domain_size[1];
  setup_context->lz        = domain_size[2];
  setup_context->time      = 0;

  if (bc->num_freestream > 0) PetscCall(FreestreamBCSetup(problem, dm, ctx, newtonian_ig_ctx, &reference));
  if (bc->num_outflow > 0) PetscCall(OutflowBCSetup(problem, dm, ctx, newtonian_ig_ctx, &reference));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &problem->ics.qfunction_context));
  PetscCallCeed(ceed,
                CeedQFunctionContextSetData(problem->ics.qfunction_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*setup_context), setup_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(problem->ics.qfunction_context, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(problem->ics.qfunction_context, "evaluation time", offsetof(struct SetupContext_, time), 1,
                                                         "Time of evaluation"));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &newtonian_ig_context));
  PetscCallCeed(ceed,
                CeedQFunctionContextSetData(newtonian_ig_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*newtonian_ig_ctx), newtonian_ig_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(newtonian_ig_context, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(newtonian_ig_context, "timestep size", offsetof(struct NewtonianIdealGasContext_, dt), 1,
                                                         "Size of timestep, delta t"));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(newtonian_ig_context, "ijacobian time shift",
                                                         offsetof(struct NewtonianIdealGasContext_, ijacobian_time_shift), 1,
                                                         "Shift for mass matrix in IJacobian"));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(newtonian_ig_context, "solution time", offsetof(struct NewtonianIdealGasContext_, time), 1,
                                                         "Current solution time"));

  problem->apply_vol_rhs.qfunction_context = newtonian_ig_context;
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_context, &problem->apply_vol_ifunction.qfunction_context));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_context, &problem->apply_vol_ijacobian.qfunction_context));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_context, &problem->apply_inflow.qfunction_context));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_context, &problem->apply_inflow_jacobian.qfunction_context));

  if (unit_tests) {
    PetscCall(UnitTests_Newtonian(user, newtonian_ig_ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PRINT_NEWTONIAN(User user, ProblemData *problem, AppCtx app_ctx) {
  MPI_Comm                 comm = user->comm;
  Ceed                     ceed = user->ceed;
  NewtonianIdealGasContext newtonian_ctx;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context, CEED_MEM_HOST, &newtonian_ctx));
  PetscCall(PetscPrintf(comm,
                        "  Problem:\n"
                        "    Problem Name                       : %s\n"
                        "    Stabilization                      : %s\n",
                        app_ctx->problem_name, StabilizationTypes[newtonian_ctx->stabilization]));
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context, &newtonian_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
