// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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

// For use with PetscOptionsEnum
static const char *const StateVariables[] = {"conservative", "primitive", "entropy", "StateVariable", "STATEVAR_", NULL};

static PetscErrorCode CheckQWithTolerance(const CeedScalar Q_s[5], const CeedScalar Q_a[5], const CeedScalar Q_b[5], const char *name,
                                          PetscReal rtol_0, PetscReal rtol_u, PetscReal rtol_4) {
  CeedScalar relative_error[5];  // relative error
  CeedScalar divisor_threshold = 10 * CEED_EPSILON;

  PetscFunctionBeginUser;
  relative_error[0] = (Q_a[0] - Q_b[0]) / (fabs(Q_s[0]) > divisor_threshold ? Q_s[0] : 1);
  relative_error[4] = (Q_a[4] - Q_b[4]) / (fabs(Q_s[4]) > divisor_threshold ? Q_s[4] : 1);

  CeedScalar u_magnitude = sqrt(Square(Q_s[1]) + Square(Q_s[2]) + Square(Q_s[3]));
  CeedScalar u_divisor   = u_magnitude > divisor_threshold ? u_magnitude : 1;
  for (int i = 1; i < 4; i++) {
    relative_error[i] = (Q_a[i] - Q_b[i]) / u_divisor;
  }

  if (fabs(relative_error[0]) >= rtol_0) {
    printf("%s[0] error %g (expected %.10e, got %.10e)\n", name, relative_error[0], Q_s[0], Q_a[0]);
  }
  for (int i = 1; i < 4; i++) {
    if (fabs(relative_error[i]) >= rtol_u) {
      printf("%s[%d] error %g (expected %.10e, got %.10e)\n", name, i, relative_error[i], Q_s[i], Q_a[i]);
    }
  }
  if (fabs(relative_error[4]) >= rtol_4) {
    printf("%s[4] error %g (expected %.10e, got %.10e)\n", name, relative_error[4], Q_s[4], Q_a[4]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Verify `StateFromQ` by converting A0 -> B0 -> A0_test, where A0 should equal A0_test
static PetscErrorCode TestState(StateVariable state_var_A, StateVariable state_var_B, NewtonianIdealGasContext gas, const CeedScalar A0[5],
                                CeedScalar rtol_0, CeedScalar rtol_u, CeedScalar rtol_4) {
  CeedScalar        B0[5], A0_test[5];
  char              buf[128];
  const char *const StateVariables_Initial[] = {"U", "Y", "V"};

  PetscFunctionBeginUser;
  const char *A_initial = StateVariables_Initial[state_var_A];
  const char *B_initial = StateVariables_Initial[state_var_B];

  State state_A0 = StateFromQ(gas, A0, state_var_A);
  StateToQ(gas, state_A0, B0, state_var_B);
  State state_B0 = StateFromQ(gas, B0, state_var_B);
  StateToQ(gas, state_B0, A0_test, state_var_A);

  snprintf(buf, sizeof buf, "%s->%s->%s: %s", A_initial, B_initial, A_initial, A_initial);
  PetscCall(CheckQWithTolerance(A0, A0_test, A0, buf, rtol_0, rtol_u, rtol_4));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Verify `StateFromQ_fwd` via a finite difference approximation
static PetscErrorCode TestState_fwd(StateVariable state_var_A, StateVariable state_var_B, NewtonianIdealGasContext gas, const CeedScalar A0[5],
                                    CeedScalar rtol_0, CeedScalar rtol_u, CeedScalar rtol_4) {
  CeedScalar        eps = 4e-7;  // Finite difference step
  char              buf[128];
  const char *const StateVariables_Initial[] = {"U", "Y", "V"};

  PetscFunctionBeginUser;
  const char *A_initial = StateVariables_Initial[state_var_A];
  const char *B_initial = StateVariables_Initial[state_var_B];
  State       state_0   = StateFromQ(gas, A0, state_var_A);

  for (int i = 0; i < 5; i++) {
    CeedScalar dB[5] = {0.}, dB_fd[5] = {0.};
    {  // Calculate dB using State functions
      CeedScalar dA[5] = {0};

      dA[i]    = A0[i];
      State ds = StateFromQ_fwd(gas, state_0, dA, state_var_A);
      StateToQ(gas, ds, dB, state_var_B);
    }

    {  // Calculate dB_fd via finite difference approximation
      CeedScalar A1[5], B0[5], B1[5];

      for (int j = 0; j < 5; j++) A1[j] = (1 + eps * (i == j)) * A0[j];
      State state_1 = StateFromQ(gas, A1, state_var_A);
      StateToQ(gas, state_0, B0, state_var_B);
      StateToQ(gas, state_1, B1, state_var_B);
      for (int j = 0; j < 5; j++) dB_fd[j] = (B1[j] - B0[j]) / eps;
    }

    snprintf(buf, sizeof buf, "d%s->d%s: StateFrom%s_fwd i=%d: d%s", A_initial, B_initial, A_initial, i, B_initial);
    PetscCall(CheckQWithTolerance(dB_fd, dB, dB_fd, buf, rtol_0, rtol_u, rtol_4));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Test the Newtonian State transformation functions, `StateFrom*`
static PetscErrorCode UnitTests_Newtonian(User user, NewtonianIdealGasContext gas) {
  Units            units = user->units;
  const CeedScalar kg = units->kilogram, m = units->meter, sec = units->second, K = units->Kelvin;

  PetscFunctionBeginUser;
  const CeedScalar T          = 200 * K;
  const CeedScalar rho        = 1.2 * kg / Cube(m);
  const CeedScalar P          = (HeatCapacityRatio(gas) - 1) * rho * gas->cv * T;
  const CeedScalar u_base     = 40 * m / sec;
  const CeedScalar u[3]       = {u_base, u_base * 1.1, u_base * 1.2};
  const CeedScalar e_kinetic  = 0.5 * Dot3(u, u);
  const CeedScalar e_internal = gas->cv * T;
  const CeedScalar e_total    = e_kinetic + e_internal;
  const CeedScalar gamma      = HeatCapacityRatio(gas);
  const CeedScalar entropy    = log(P) - gamma * log(rho);
  const CeedScalar rho_div_p  = rho / P;
  const CeedScalar Y0[5]      = {P, u[0], u[1], u[2], T};
  const CeedScalar U0[5]      = {rho, rho * u[0], rho * u[1], rho * u[2], rho * e_total};
  const CeedScalar V0[5]      = {(gamma - entropy) / (gamma - 1) - rho_div_p * (e_kinetic), rho_div_p * u[0], rho_div_p * u[1], rho_div_p * u[2],
                                 -rho_div_p};

  {
    CeedScalar rtol = 20 * CEED_EPSILON;

    PetscCall(TestState(STATEVAR_PRIMITIVE, STATEVAR_CONSERVATIVE, gas, Y0, rtol, rtol, rtol));
    PetscCall(TestState(STATEVAR_PRIMITIVE, STATEVAR_ENTROPY, gas, Y0, rtol, rtol, rtol));
    PetscCall(TestState(STATEVAR_CONSERVATIVE, STATEVAR_PRIMITIVE, gas, U0, rtol, rtol, rtol));
    PetscCall(TestState(STATEVAR_CONSERVATIVE, STATEVAR_ENTROPY, gas, U0, rtol, rtol, rtol));
    PetscCall(TestState(STATEVAR_ENTROPY, STATEVAR_CONSERVATIVE, gas, V0, rtol, rtol, rtol));
    PetscCall(TestState(STATEVAR_ENTROPY, STATEVAR_PRIMITIVE, gas, V0, rtol, rtol, rtol));
  }

  {
    CeedScalar rtol = 5e-6;

    PetscCall(TestState_fwd(STATEVAR_PRIMITIVE, STATEVAR_CONSERVATIVE, gas, Y0, rtol, rtol, rtol));
    PetscCall(TestState_fwd(STATEVAR_PRIMITIVE, STATEVAR_ENTROPY, gas, Y0, rtol, rtol, rtol));
    PetscCall(TestState_fwd(STATEVAR_CONSERVATIVE, STATEVAR_PRIMITIVE, gas, U0, rtol, rtol, rtol));
    PetscCall(TestState_fwd(STATEVAR_CONSERVATIVE, STATEVAR_ENTROPY, gas, U0, 10 * rtol, rtol, rtol));
    PetscCall(TestState_fwd(STATEVAR_ENTROPY, STATEVAR_CONSERVATIVE, gas, V0, 5 * rtol, rtol, rtol));
    PetscCall(TestState_fwd(STATEVAR_ENTROPY, STATEVAR_PRIMITIVE, gas, V0, 5 * rtol, 5 * rtol, 5 * rtol));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create CeedOperator for stabilized mass KSP for explicit timestepping
//
// Only used for SUPG stabilization
PetscErrorCode CreateKSPMassOperator_NewtonianStabilized(User user, CeedOperator *op_mass) {
  Ceed                 ceed = user->ceed;
  CeedInt              num_comp_q, q_data_size;
  CeedQFunction        qf_mass;
  CeedElemRestriction  elem_restr_q, elem_restr_qd_i;
  CeedBasis            basis_q;
  CeedVector           q_data;
  CeedQFunctionContext qf_ctx = NULL;
  PetscInt             dim    = 3;

  PetscFunctionBeginUser;
  {  // Get restriction and basis from the RHS function
    CeedOperator     *sub_ops;
    CeedOperatorField field;
    PetscInt          sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedCompositeOperatorGetSubList(user->op_rhs_ctx->op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "q", &field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(field, &elem_restr_q));
    PetscCallCeed(ceed, CeedOperatorFieldGetBasis(field, &basis_q));

    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "qdata", &field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(field, &elem_restr_qd_i));
    PetscCallCeed(ceed, CeedOperatorFieldGetVector(field, &q_data));

    PetscCallCeed(ceed, CeedOperatorGetContext(sub_ops[sub_op_index], &qf_ctx));
  }

  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_qd_i, &q_data_size));

  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MassFunction_Newtonian_Conserv, MassFunction_Newtonian_Conserv_loc, &qf_mass));

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_mass, qf_ctx));
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_mass, 0));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "q_dot", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "q", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_mass, "v", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_mass, "Grad_v", 5 * dim, CEED_EVAL_GRAD));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "q_dot", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "q", elem_restr_q, basis_q, user->q_ceed));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "qdata", elem_restr_qd_i, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "Grad_v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NS_NEWTONIAN_IG(ProblemData problem, DM dm, void *ctx, SimpleBC bc) {
  SetupContext             setup_context;
  User                     user   = *(User *)ctx;
  CeedInt                  degree = user->app_ctx->degree;
  StabilizationType        stab;
  StateVariable            state_var;
  MPI_Comm                 comm = user->comm;
  Ceed                     ceed = user->ceed;
  PetscBool                implicit;
  PetscBool                unit_tests;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext     newtonian_ig_context;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(1, &setup_context));
  PetscCall(PetscCalloc1(1, &newtonian_ig_ctx));

  // ------------------------------------------------------
  //           Setup Generic Newtonian IG Problem
  // ------------------------------------------------------
  problem->dim               = 3;
  problem->jac_data_size_sur = 11;
  problem->non_zero_time     = PETSC_FALSE;
  problem->print_info        = PRINT_NEWTONIAN;
  problem->uses_newtonian    = PETSC_TRUE;

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar cv         = 717.;          // J/(kg K)
  CeedScalar cp         = 1004.;         // J/(kg K)
  CeedScalar g[3]       = {0, 0, 0};     // m/s^2
  CeedScalar lambda     = -2. / 3.;      // -
  CeedScalar mu         = 1.8e-5;        // Pa s, dynamic viscosity
  CeedScalar k          = 0.02638;       // W/(m K)
  CeedScalar c_tau      = 0.5 / degree;  // -
  CeedScalar Ctau_t     = 1.0;           // -
  CeedScalar Cv_func[3] = {36, 60, 128};
  CeedScalar Ctau_v     = Cv_func[(CeedInt)Min(3, degree) - 1];
  CeedScalar Ctau_C     = 0.25 / degree;
  CeedScalar Ctau_M     = 0.25 / degree;
  CeedScalar Ctau_E     = 0.125;
  PetscReal  domain_min[3], domain_max[3], domain_size[3];
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  StatePrimitive reference      = {.pressure = 1.01e5, .velocity = {0}, .temperature = 288.15};
  CeedScalar     idl_decay_time = -1, idl_start = 0, idl_length = 0, idl_pressure = reference.pressure;
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
  PetscBool given_option = PETSC_FALSE;
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
      break;
  }

  // -- Physics
  PetscCall(PetscOptionsScalar("-cv", "Heat capacity at constant volume", NULL, cv, &cv, NULL));
  PetscCall(PetscOptionsScalar("-cp", "Heat capacity at constant pressure", NULL, cp, &cp, NULL));
  PetscCall(PetscOptionsScalar("-lambda", "Stokes hypothesis second viscosity coefficient", NULL, lambda, &lambda, NULL));
  PetscCall(PetscOptionsScalar("-mu", "Shear dynamic viscosity coefficient", NULL, mu, &mu, NULL));
  PetscCall(PetscOptionsScalar("-k", "Thermal conductivity", NULL, k, &k, NULL));

  PetscInt dim = problem->dim;
  PetscCall(PetscOptionsDeprecated("-g", "-gravity", "libCEED 0.11.1", NULL));
  PetscCall(PetscOptionsRealArray("-gravity", "Gravitational acceleration vector", NULL, g, &dim, &given_option));
  if (given_option) PetscCheck(dim == 3, comm, PETSC_ERR_ARG_SIZ, "Gravity vector must be size 3, %" PetscInt_FMT " values given", dim);

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
  PetscCheck(!(state_var == STATEVAR_PRIMITIVE && !implicit), comm, PETSC_ERR_SUP,
             "RHSFunction is not provided for primitive variables (use -state_var primitive only with -implicit)\n");

  PetscCall(PetscOptionsScalar("-idl_decay_time", "Characteristic timescale of the pressure deviance decay. The timestep is good starting point",
                               NULL, idl_decay_time, &idl_decay_time, &idl_enable));
  PetscCheck(!(idl_enable && idl_decay_time == 0), comm, PETSC_ERR_SUP, "idl_decay_time may not be equal to zero.");
  if (idl_decay_time < 0) idl_enable = PETSC_FALSE;
  PetscCall(PetscOptionsScalar("-idl_start", "Start of IDL in the x direction", NULL, idl_start, &idl_start, NULL));
  PetscCall(PetscOptionsScalar("-idl_length", "Length of IDL in the positive x direction", NULL, idl_length, &idl_length, NULL));
  idl_pressure = reference.pressure;
  PetscCall(PetscOptionsScalar("-idl_pressure", "Pressure IDL uses as reference (default is `-reference_pressure`)", NULL, idl_pressure,
                               &idl_pressure, NULL));
  PetscOptionsEnd();

  if (stab == STAB_SUPG && !implicit) problem->create_mass_operator = CreateKSPMassOperator_NewtonianStabilized;

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
  user->phys->implicit  = implicit;
  user->phys->state_var = state_var;

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
  newtonian_ig_ctx->state_var     = state_var;
  newtonian_ig_ctx->idl_enable    = idl_enable;
  newtonian_ig_ctx->idl_amplitude = 1 / (idl_decay_time * second);
  newtonian_ig_ctx->idl_start     = idl_start * meter;
  newtonian_ig_ctx->idl_length    = idl_length * meter;
  newtonian_ig_ctx->idl_pressure  = idl_pressure;
  PetscCall(PetscArraycpy(newtonian_ig_ctx->g, g, 3));

  // -- Setup Context
  setup_context->reference = reference;
  setup_context->gas       = *newtonian_ig_ctx;
  setup_context->lx        = domain_size[0];
  setup_context->ly        = domain_size[1];
  setup_context->lz        = domain_size[2];
  setup_context->time      = 0;

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

  if (bc->num_freestream > 0) PetscCall(FreestreamBCSetup(problem, dm, ctx, newtonian_ig_ctx, &reference));
  if (bc->num_outflow > 0) PetscCall(OutflowBCSetup(problem, dm, ctx, newtonian_ig_ctx, &reference));
  if (bc->num_slip > 0) PetscCall(SlipBCSetup(problem, dm, ctx, newtonian_ig_context));

  if (unit_tests) {
    PetscCall(UnitTests_Newtonian(user, newtonian_ig_ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PRINT_NEWTONIAN(User user, ProblemData problem, AppCtx app_ctx) {
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
