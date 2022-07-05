// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up problems using the Newtonian Qfunction

#include "../navierstokes.h"
#include "../qfunctions/setupgeo.h"
#include "../qfunctions/newtonian.h"

// Compute relative error |a - b|/|s|
static PetscErrorCode CheckPrimitiveWithTolerance(StatePrimitive sY,
    StatePrimitive aY, StatePrimitive bY, const char *name, PetscReal rtol_pressure,
    PetscReal rtol_velocity, PetscReal rtol_temperature) {

  PetscFunctionBeginUser;
  StatePrimitive eY; // relative error
  eY.pressure = (aY.pressure - bY.pressure) / sY.pressure;
  PetscScalar u = sqrt(Square(sY.velocity[0]) + Square(sY.velocity[1]) + Square(
                         sY.velocity[2]));
  for (int j=0; j<3; j++) eY.velocity[j] = (aY.velocity[j] - bY.velocity[j]) / u;
  eY.temperature = (aY.temperature - bY.temperature) / sY.temperature;
  if (fabs(eY.pressure) > rtol_pressure)
    printf("%s: pressure error %g\n", name, eY.pressure);
  for (int j=0; j<3; j++)
    if (fabs(eY.velocity[j]) > rtol_velocity)
      printf("%s: velocity[%d] error %g\n", name, j, eY.velocity[j]);
  if (fabs(eY.temperature) > rtol_temperature)
    printf("%s: temperature error %g\n", name, eY.temperature);
  PetscFunctionReturn(0);
}

static PetscErrorCode UnitTests_Newtonian(User user,
    NewtonianIdealGasContext gas) {

  Units units = user->units;
  const CeedScalar eps    = 1e-6;
  const CeedScalar kg     = units->kilogram,
                   m      = units->meter,
                   sec    = units->second,
                   Pascal = units->Pascal;
  PetscFunctionBeginUser;
  const CeedScalar rho = 1.2 * kg / (m*m*m),
                   u   = 40 * m/sec;
  CeedScalar U[5] = {rho, rho*u, rho *u*1.1, rho *u*1.2, 250e3*Pascal + .5*rho *u*u};
  const CeedScalar x[3] = {.1, .2, .3};
  State s = StateFromU(gas, U, x);
  for (int i=0; i<8; i++) {
    CeedScalar dU[5] = {0}, dx[3] = {0};
    if (i < 5) dU[i] = U[i];
    else dx[i-5] = x[i-5];
    State ds = StateFromU_fwd(gas, s, dU, x, dx);
    for (int j=0; j<5; j++) dU[j] = (1 + eps * (i == j)) * U[j];
    for (int j=0; j<3; j++) dx[j] = (1 + eps * (i == 5 + j)) * x[j];
    State t = StateFromU(gas, dU, dx);
    StatePrimitive dY;
    dY.pressure = (t.Y.pressure - s.Y.pressure) / eps;
    for (int j=0; j<3; j++)
      dY.velocity[j] = (t.Y.velocity[j] - s.Y.velocity[j]) / eps;
    dY.temperature = (t.Y.temperature - s.Y.temperature) / eps;
    char buf[128];
    snprintf(buf, sizeof buf, "StateFromU_fwd i=%d", i);
    PetscCall(CheckPrimitiveWithTolerance(dY, ds.Y, dY, buf, 5e-6, 1e-6, 1e-6));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode NS_NEWTONIAN_IG(ProblemData *problem, DM dm, void *ctx) {

  SetupContext      setup_context;
  User              user = *(User *)ctx;
  StabilizationType stab;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscBool         implicit;
  PetscBool         has_curr_time = PETSC_FALSE,
                    prim_var, unit_tests;
  PetscInt          ierr;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext newtonian_ig_context;

  PetscFunctionBeginUser;
  ierr = PetscCalloc1(1, &setup_context); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &newtonian_ig_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //           Setup Generic Newtonian IG Problem
  // ------------------------------------------------------
  problem->dim                                  = 3;
  problem->q_data_size_vol                      = 10;
  problem->q_data_size_sur                      = 10;
  problem->jac_data_size_sur                    = 11;
  problem->setup_vol.qfunction                  = Setup;
  problem->setup_vol.qfunction_loc              = Setup_loc;
  problem->ics.qfunction                        = ICsNewtonianIG;
  problem->ics.qfunction_loc                    = ICsNewtonianIG_loc;
  problem->setup_sur.qfunction                  = SetupBoundary;
  problem->setup_sur.qfunction_loc              = SetupBoundary_loc;
  problem->apply_vol_rhs.qfunction              = RHSFunction_Newtonian;
  problem->apply_vol_rhs.qfunction_loc          = RHSFunction_Newtonian_loc;
  problem->apply_vol_ifunction.qfunction        = IFunction_Newtonian;
  problem->apply_vol_ifunction.qfunction_loc    = IFunction_Newtonian_loc;
  problem->apply_vol_ijacobian.qfunction        = IJacobian_Newtonian;
  problem->apply_vol_ijacobian.qfunction_loc    = IJacobian_Newtonian_loc;
  problem->apply_inflow.qfunction               = BoundaryIntegral;
  problem->apply_inflow.qfunction_loc           = BoundaryIntegral_loc;
  problem->apply_inflow_jacobian.qfunction      = BoundaryIntegral_Jacobian;
  problem->apply_inflow_jacobian.qfunction_loc  = BoundaryIntegral_Jacobian_loc;
  problem->apply_outflow.qfunction              = PressureOutflow;
  problem->apply_outflow.qfunction_loc          = PressureOutflow_loc;
  problem->apply_outflow_jacobian.qfunction     = PressureOutflow_Jacobian;
  problem->apply_outflow_jacobian.qfunction_loc = PressureOutflow_Jacobian_loc;
  problem->bc                                   = NULL;
  problem->bc_ctx                               = setup_context;
  problem->non_zero_time                        = PETSC_FALSE;
  problem->print_info                           = PRINT_NEWTONIAN;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar cv     = 717.;          // J/(kg K)
  CeedScalar cp     = 1004.;         // J/(kg K)
  CeedScalar g[3]   = {0, 0, -9.81}; // m/s^2
  CeedScalar lambda = -2./3.;        // -
  CeedScalar mu     = 1.8e-5;        // Pa s, dynamic viscosity
  CeedScalar k      = 0.02638;       // W/(m K)
  CeedScalar c_tau  = 0.5;           // -
  CeedScalar Ctau_t = 1.0;           // -
  CeedScalar Ctau_v = 36.0;          // TODO make function of degree
  CeedScalar Ctau_C = 1.0;           // TODO make function of degree
  CeedScalar Ctau_M = 1.0;           // TODO make function of degree
  CeedScalar Ctau_E = 1.0;           // TODO make function of degree
  PetscReal domain_min[3], domain_max[3], domain_size[3];
  ierr = DMGetBoundingBox(dm, domain_min, domain_max); CHKERRQ(ierr);
  for (PetscInt i=0; i<3; i++) domain_size[i] = domain_max[i] - domain_min[i];

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
  PetscOptionsBegin(comm, NULL, "Options for Newtonian Ideal Gas based problem",
                    NULL);
  // -- Conservative vs Primitive variables
  ierr = PetscOptionsBool("-primitive", "Use primitive variables",
                          NULL, prim_var=PETSC_FALSE, &prim_var, NULL); CHKERRQ(ierr);
  if (prim_var) {
    problem->ics.qfunction                     = ICsNewtonianIG_Prim;
    problem->ics.qfunction_loc                 = ICsNewtonianIG_Prim_loc;
    problem->apply_vol_rhs.qfunction           = RHSFunction_Newtonian_Prim;
    problem->apply_vol_rhs.qfunction_loc       = RHSFunction_Newtonian_Prim_loc;
    problem->apply_vol_ifunction.qfunction     = IFunction_Newtonian_Prim;
    problem->apply_vol_ifunction.qfunction_loc = IFunction_Newtonian_Prim_loc;
    problem->apply_vol_ijacobian.qfunction     = IJacobian_Newtonian_Prim;
    problem->apply_vol_ijacobian.qfunction_loc = IJacobian_Newtonian_Prim_loc;
  }

  // -- Physics
  ierr = PetscOptionsScalar("-cv", "Heat capacity at constant volume",
                            NULL, cv, &cv, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-cp", "Heat capacity at constant pressure",
                            NULL, cp, &cp, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lambda",
                            "Stokes hypothesis second viscosity coefficient",
                            NULL, lambda, &lambda, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-mu", "Shear dynamic viscosity coefficient",
                            NULL, mu, &mu, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-k", "Thermal conductivity",
                            NULL, k, &k, NULL); CHKERRQ(ierr);

  PetscInt dim = problem->dim;
  ierr = PetscOptionsRealArray("-g", "Gravitational acceleration",
                               NULL, g, &dim, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-stab", "Stabilization method", NULL,
                          StabilizationTypes, (PetscEnum)(stab = STAB_NONE),
                          (PetscEnum *)&stab, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-c_tau", "Stabilization constant",
                            NULL, c_tau, &c_tau, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-Ctau_t", "Stabilization time constant",
                            NULL, Ctau_t, &Ctau_t, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-Ctau_v", "Stabilization viscous constant",
                            NULL, Ctau_v, &Ctau_v, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-Ctau_C", "Stabilization continuity constant",
                            NULL, Ctau_C, &Ctau_C, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-Ctau_M", "Stabilization momentum constant",
                            NULL, Ctau_M, &Ctau_M, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-Ctau_E", "Stabilization energy constant",
                            NULL, Ctau_E, &Ctau_E, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation",
                          NULL, implicit=PETSC_FALSE, &implicit, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsBool("-newtonian_unit_tests", "Run Newtonian unit tests",
                          NULL, unit_tests=PETSC_FALSE, &unit_tests, NULL);
  CHKERRQ(ierr);

  // -- Units
  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, meter, &meter, NULL); CHKERRQ(ierr);
  meter = fabs(meter);
  ierr = PetscOptionsScalar("-units_kilogram","1 kilogram in scaled mass units",
                            NULL, kilogram, &kilogram, NULL); CHKERRQ(ierr);
  kilogram = fabs(kilogram);
  ierr = PetscOptionsScalar("-units_second","1 second in scaled time units",
                            NULL, second, &second, NULL); CHKERRQ(ierr);
  second = fabs(second);
  ierr = PetscOptionsScalar("-units_Kelvin",
                            "1 Kelvin in scaled temperature units",
                            NULL, Kelvin, &Kelvin, NULL); CHKERRQ(ierr);
  Kelvin = fabs(Kelvin);

  // -- Warnings
  if (stab == STAB_SUPG && !implicit) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -stab supg only with -implicit\n");
    CHKERRQ(ierr);
  }
  PetscOptionsEnd();

  // ------------------------------------------------------
  //           Set up the PETSc context
  // ------------------------------------------------------
  // -- Define derived units
  Pascal          = kilogram / (meter * PetscSqr(second));
  J_per_kg_K      =  PetscSqr(meter) / (PetscSqr(second) * Kelvin);
  m_per_squared_s = meter / PetscSqr(second);
  W_per_m_K       = kilogram * meter / (pow(second,3) * Kelvin);

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
  cv     *= J_per_kg_K;
  cp     *= J_per_kg_K;
  mu     *= Pascal * second;
  k      *= W_per_m_K;
  for (PetscInt i=0; i<3; i++) domain_size[i] *= meter;
  for (PetscInt i=0; i<3; i++) g[i]           *= m_per_squared_s;
  problem->dm_scale = meter;

  // -- Setup Context
  setup_context->cv         = cv;
  setup_context->cp         = cp;
  setup_context->lx         = domain_size[0];
  setup_context->ly         = domain_size[1];
  setup_context->lz         = domain_size[2];
  setup_context->time       = 0;
  ierr = PetscArraycpy(setup_context->g, g, 3); CHKERRQ(ierr);

  // -- Solver Settings
  user->phys->stab          = stab;
  user->phys->implicit      = implicit;
  user->phys->primitive     = prim_var;
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
  newtonian_ig_ctx->stabilization = stab;
  newtonian_ig_ctx->is_implicit   = implicit;
  ierr = PetscArraycpy(newtonian_ig_ctx->g, g, 3); CHKERRQ(ierr);

  CeedQFunctionContextCreate(user->ceed, &problem->ics.qfunction_context);
  CeedQFunctionContextSetData(problem->ics.qfunction_context, CEED_MEM_HOST,
                              CEED_USE_POINTER, sizeof(*setup_context), setup_context);
  CeedQFunctionContextSetDataDestroy(problem->ics.qfunction_context,
                                     CEED_MEM_HOST,
                                     FreeContextPetsc);
  CeedQFunctionContextRegisterDouble(problem->ics.qfunction_context,
                                     "evaluation time",
                                     (char *)&setup_context->time - (char *)setup_context, 1, "Time of evaluation");

  CeedQFunctionContextCreate(user->ceed, &newtonian_ig_context);
  CeedQFunctionContextSetData(newtonian_ig_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*newtonian_ig_ctx), newtonian_ig_ctx);
  CeedQFunctionContextSetDataDestroy(newtonian_ig_context, CEED_MEM_HOST,
                                     FreeContextPetsc);
  CeedQFunctionContextRegisterDouble(newtonian_ig_context, "timestep size",
                                     offsetof(struct NewtonianIdealGasContext_, dt), 1, "Size of timestep, delta t");
  CeedQFunctionContextRegisterDouble(newtonian_ig_context, "ijacobian time shift",
                                     offsetof(struct NewtonianIdealGasContext_, ijacobian_time_shift), 1,
                                     "Shift for mass matrix in IJacobian");
  problem->apply_vol_rhs.qfunction_context = newtonian_ig_context;
  CeedQFunctionContextReferenceCopy(newtonian_ig_context,
                                    &problem->apply_vol_ifunction.qfunction_context);
  CeedQFunctionContextReferenceCopy(newtonian_ig_context,
                                    &problem->apply_vol_ijacobian.qfunction_context);
  CeedQFunctionContextReferenceCopy(newtonian_ig_context,
                                    &problem->apply_inflow.qfunction_context);
  CeedQFunctionContextReferenceCopy(newtonian_ig_context,
                                    &problem->apply_inflow_jacobian.qfunction_context);
  CeedQFunctionContextReferenceCopy(newtonian_ig_context,
                                    &problem->apply_outflow.qfunction_context);
  CeedQFunctionContextReferenceCopy(newtonian_ig_context,
                                    &problem->apply_outflow_jacobian.qfunction_context);

  if (unit_tests) {
    PetscCall(UnitTests_Newtonian(user, newtonian_ig_ctx));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PRINT_NEWTONIAN(ProblemData *problem, AppCtx app_ctx) {

  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscErrorCode ierr;
  NewtonianIdealGasContext newtonian_ctx;

  PetscFunctionBeginUser;
  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context,
                              CEED_MEM_HOST, &newtonian_ctx);
  ierr = PetscPrintf(comm,
                     "  Problem:\n"
                     "    Problem Name                       : %s\n"
                     "    Stabilization                      : %s\n",
                     app_ctx->problem_name, StabilizationTypes[newtonian_ctx->stabilization]);
  CHKERRQ(ierr);
  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context,
                                  &newtonian_ctx);
  PetscFunctionReturn(0);
}
