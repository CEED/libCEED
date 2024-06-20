// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Freestream boundary condition

#include "../qfunctions/bc_freestream.h"

#include <ceed.h>
#include <petscdm.h>

#include "../navierstokes.h"
#include "../qfunctions/newtonian_types.h"

static const char *const RiemannSolverTypes[] = {"HLL", "HLLC", "RiemannSolverTypes", "RIEMANN_", NULL};

static PetscErrorCode RiemannSolverUnitTests(NewtonianIdealGasContext gas, CeedScalar rtol);

PetscErrorCode FreestreamBCSetup(ProblemData problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx, const StatePrimitive *reference) {
  User                 user = *(User *)ctx;
  MPI_Comm             comm = user->comm;
  Ceed                 ceed = user->ceed;
  FreestreamContext    freestream_ctx;
  CeedQFunctionContext freestream_context;
  RiemannFluxType      riemann = RIEMANN_HLLC;
  PetscScalar          meter   = user->units->meter;
  PetscScalar          second  = user->units->second;
  PetscScalar          Kelvin  = user->units->Kelvin;
  PetscScalar          Pascal  = user->units->Pascal;

  PetscFunctionBeginUser;
  // Freestream inherits reference state. We re-dimensionalize so the defaults
  // in -help will be visible in SI units.
  StatePrimitive Y_inf = {.pressure = reference->pressure / Pascal, .velocity = {0}, .temperature = reference->temperature / Kelvin};
  for (int i = 0; i < 3; i++) Y_inf.velocity[i] = reference->velocity[i] * second / meter;

  PetscOptionsBegin(comm, NULL, "Options for Freestream boundary condition", NULL);
  PetscCall(PetscOptionsEnum("-freestream_riemann", "Riemann solver to use in freestream boundary condition", NULL, RiemannSolverTypes,
                             (PetscEnum)riemann, (PetscEnum *)&riemann, NULL));
  PetscCall(PetscOptionsScalar("-freestream_pressure", "Pressure at freestream condition", NULL, Y_inf.pressure, &Y_inf.pressure, NULL));
  PetscInt narray = 3;
  PetscCall(PetscOptionsScalarArray("-freestream_velocity", "Velocity at freestream condition", NULL, Y_inf.velocity, &narray, NULL));
  PetscCall(PetscOptionsScalar("-freestream_temperature", "Temperature at freestream condition", NULL, Y_inf.temperature, &Y_inf.temperature, NULL));
  PetscOptionsEnd();

  switch (user->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      switch (riemann) {
        case RIEMANN_HLL:
          problem->apply_freestream.qfunction              = Freestream_Conserv_HLL;
          problem->apply_freestream.qfunction_loc          = Freestream_Conserv_HLL_loc;
          problem->apply_freestream_jacobian.qfunction     = Freestream_Jacobian_Conserv_HLL;
          problem->apply_freestream_jacobian.qfunction_loc = Freestream_Jacobian_Conserv_HLL_loc;
          break;
        case RIEMANN_HLLC:
          problem->apply_freestream.qfunction              = Freestream_Conserv_HLLC;
          problem->apply_freestream.qfunction_loc          = Freestream_Conserv_HLLC_loc;
          problem->apply_freestream_jacobian.qfunction     = Freestream_Jacobian_Conserv_HLLC;
          problem->apply_freestream_jacobian.qfunction_loc = Freestream_Jacobian_Conserv_HLLC_loc;
          break;
      }
      break;
    case STATEVAR_PRIMITIVE:
      switch (riemann) {
        case RIEMANN_HLL:
          problem->apply_freestream.qfunction              = Freestream_Prim_HLL;
          problem->apply_freestream.qfunction_loc          = Freestream_Prim_HLL_loc;
          problem->apply_freestream_jacobian.qfunction     = Freestream_Jacobian_Prim_HLL;
          problem->apply_freestream_jacobian.qfunction_loc = Freestream_Jacobian_Prim_HLL_loc;
          break;
        case RIEMANN_HLLC:
          problem->apply_freestream.qfunction              = Freestream_Prim_HLLC;
          problem->apply_freestream.qfunction_loc          = Freestream_Prim_HLLC_loc;
          problem->apply_freestream_jacobian.qfunction     = Freestream_Jacobian_Prim_HLLC;
          problem->apply_freestream_jacobian.qfunction_loc = Freestream_Jacobian_Prim_HLLC_loc;
          break;
      }
      break;
    case STATEVAR_ENTROPY:
      switch (riemann) {
        case RIEMANN_HLL:
          problem->apply_freestream.qfunction              = Freestream_Entropy_HLL;
          problem->apply_freestream.qfunction_loc          = Freestream_Entropy_HLL_loc;
          problem->apply_freestream_jacobian.qfunction     = Freestream_Jacobian_Entropy_HLL;
          problem->apply_freestream_jacobian.qfunction_loc = Freestream_Jacobian_Entropy_HLL_loc;
          break;
        case RIEMANN_HLLC:
          problem->apply_freestream.qfunction              = Freestream_Entropy_HLLC;
          problem->apply_freestream.qfunction_loc          = Freestream_Entropy_HLLC_loc;
          problem->apply_freestream_jacobian.qfunction     = Freestream_Jacobian_Entropy_HLLC;
          problem->apply_freestream_jacobian.qfunction_loc = Freestream_Jacobian_Entropy_HLLC_loc;
          break;
      }
      break;
  }

  Y_inf.pressure *= Pascal;
  for (int i = 0; i < 3; i++) Y_inf.velocity[i] *= meter / second;
  Y_inf.temperature *= Kelvin;

  State S_infty = StateFromPrimitive(newtonian_ig_ctx, Y_inf);

  // -- Set freestream_ctx struct values
  PetscCall(PetscCalloc1(1, &freestream_ctx));
  freestream_ctx->newtonian_ctx = *newtonian_ig_ctx;
  freestream_ctx->S_infty       = S_infty;

  PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &freestream_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(freestream_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*freestream_ctx), freestream_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(freestream_context, CEED_MEM_HOST, FreeContextPetsc));
  problem->apply_freestream.qfunction_context = freestream_context;
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(freestream_context, &problem->apply_freestream_jacobian.qfunction_context));

  {
    PetscBool run_unit_tests = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-riemann_solver_unit_tests", &run_unit_tests, NULL));
    if (run_unit_tests) PetscCall(RiemannSolverUnitTests(newtonian_ig_ctx, 5e-7));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const char *const OutflowTypes[] = {"RIEMANN", "PRESSURE", "OutflowType", "OUTFLOW_", NULL};
typedef enum {
  OUTFLOW_RIEMANN,
  OUTFLOW_PRESSURE,
} OutflowType;

PetscErrorCode OutflowBCSetup(ProblemData problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx, const StatePrimitive *reference) {
  User                 user = *(User *)ctx;
  Ceed                 ceed = user->ceed;
  OutflowContext       outflow_ctx;
  OutflowType          outflow_type = OUTFLOW_RIEMANN;
  CeedQFunctionContext outflow_context;
  const PetscScalar    Kelvin = user->units->Kelvin;
  const PetscScalar    Pascal = user->units->Pascal;

  PetscFunctionBeginUser;
  CeedScalar pressure    = reference->pressure / Pascal;
  CeedScalar temperature = reference->temperature / Kelvin;
  CeedScalar recirc = 1, softplus_velocity = 1e-2;
  PetscOptionsBegin(user->comm, NULL, "Options for Outflow boundary condition", NULL);
  PetscCall(
      PetscOptionsEnum("-outflow_type", "Type of outflow condition", NULL, OutflowTypes, (PetscEnum)outflow_type, (PetscEnum *)&outflow_type, NULL));
  PetscCall(PetscOptionsScalar("-outflow_pressure", "Pressure at outflow condition", NULL, pressure, &pressure, NULL));
  if (outflow_type == OUTFLOW_RIEMANN) {
    PetscCall(PetscOptionsScalar("-outflow_temperature", "Temperature at outflow condition", NULL, temperature, &temperature, NULL));
    PetscCall(
        PetscOptionsReal("-outflow_recirc", "Fraction of recirculation to allow in exterior velocity state [0,1]", NULL, recirc, &recirc, NULL));
    PetscCall(PetscOptionsReal("-outflow_softplus_velocity", "Characteristic velocity of softplus regularization", NULL, softplus_velocity,
                               &softplus_velocity, NULL));
  }
  PetscOptionsEnd();
  pressure *= Pascal;
  temperature *= Kelvin;

  switch (outflow_type) {
    case OUTFLOW_RIEMANN:
      switch (user->phys->state_var) {
        case STATEVAR_CONSERVATIVE:
          problem->apply_outflow.qfunction              = RiemannOutflow_Conserv;
          problem->apply_outflow.qfunction_loc          = RiemannOutflow_Conserv_loc;
          problem->apply_outflow_jacobian.qfunction     = RiemannOutflow_Jacobian_Conserv;
          problem->apply_outflow_jacobian.qfunction_loc = RiemannOutflow_Jacobian_Conserv_loc;
          break;
        case STATEVAR_PRIMITIVE:
          problem->apply_outflow.qfunction              = RiemannOutflow_Prim;
          problem->apply_outflow.qfunction_loc          = RiemannOutflow_Prim_loc;
          problem->apply_outflow_jacobian.qfunction     = RiemannOutflow_Jacobian_Prim;
          problem->apply_outflow_jacobian.qfunction_loc = RiemannOutflow_Jacobian_Prim_loc;
          break;
        case STATEVAR_ENTROPY:
          problem->apply_outflow.qfunction              = RiemannOutflow_Entropy;
          problem->apply_outflow.qfunction_loc          = RiemannOutflow_Entropy_loc;
          problem->apply_outflow_jacobian.qfunction     = RiemannOutflow_Jacobian_Entropy;
          problem->apply_outflow_jacobian.qfunction_loc = RiemannOutflow_Jacobian_Entropy_loc;
          break;
      }
      break;
    case OUTFLOW_PRESSURE:
      switch (user->phys->state_var) {
        case STATEVAR_CONSERVATIVE:
          problem->apply_outflow.qfunction              = PressureOutflow_Conserv;
          problem->apply_outflow.qfunction_loc          = PressureOutflow_Conserv_loc;
          problem->apply_outflow_jacobian.qfunction     = PressureOutflow_Jacobian_Conserv;
          problem->apply_outflow_jacobian.qfunction_loc = PressureOutflow_Jacobian_Conserv_loc;
          break;
        case STATEVAR_PRIMITIVE:
          problem->apply_outflow.qfunction              = PressureOutflow_Prim;
          problem->apply_outflow.qfunction_loc          = PressureOutflow_Prim_loc;
          problem->apply_outflow_jacobian.qfunction     = PressureOutflow_Jacobian_Prim;
          problem->apply_outflow_jacobian.qfunction_loc = PressureOutflow_Jacobian_Prim_loc;
          break;
        case STATEVAR_ENTROPY:
          problem->apply_outflow.qfunction              = PressureOutflow_Entropy;
          problem->apply_outflow.qfunction_loc          = PressureOutflow_Entropy_loc;
          problem->apply_outflow_jacobian.qfunction     = PressureOutflow_Jacobian_Entropy;
          problem->apply_outflow_jacobian.qfunction_loc = PressureOutflow_Jacobian_Entropy_loc;
          break;
      }
      break;
  }
  PetscCall(PetscCalloc1(1, &outflow_ctx));
  outflow_ctx->gas               = *newtonian_ig_ctx;
  outflow_ctx->recirc            = recirc;
  outflow_ctx->softplus_velocity = softplus_velocity;
  outflow_ctx->pressure          = pressure;
  outflow_ctx->temperature       = temperature;

  PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &outflow_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(outflow_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*outflow_ctx), outflow_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(outflow_context, CEED_MEM_HOST, FreeContextPetsc));
  problem->apply_outflow.qfunction_context = outflow_context;
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(outflow_context, &problem->apply_outflow_jacobian.qfunction_context));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Calculate relative error, (A - B) / S
// If S < threshold, then set S=1
static inline CeedScalar RelativeError(CeedScalar S, CeedScalar A, CeedScalar B, CeedScalar threshold) {
  return (A - B) / (fabs(S) > threshold ? S : 1);
}

// @brief Check errors of a State vector and print if above tolerance
static PetscErrorCode CheckQWithTolerance(const CeedScalar Q_s[5], const CeedScalar Q_a[5], const CeedScalar Q_b[5], const char *name,
                                          PetscReal rtol_0, PetscReal rtol_u, PetscReal rtol_4) {
  CeedScalar relative_error[5];  // relative error
  CeedScalar divisor_threshold = 10 * CEED_EPSILON;

  PetscFunctionBeginUser;
  relative_error[0] = RelativeError(Q_s[0], Q_a[0], Q_b[0], divisor_threshold);
  relative_error[4] = RelativeError(Q_s[4], Q_a[4], Q_b[4], divisor_threshold);

  CeedScalar u_magnitude = sqrt(Square(Q_s[1]) + Square(Q_s[2]) + Square(Q_s[3]));
  for (int i = 1; i < 4; i++) {
    relative_error[i] = RelativeError(u_magnitude, Q_a[i], Q_b[i], divisor_threshold);
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

// @brief Verify RiemannFlux_HLL_fwd function against finite-difference approximation
static PetscErrorCode TestRiemannHLL_fwd(NewtonianIdealGasContext gas, CeedScalar rtol_0, CeedScalar rtol_u, CeedScalar rtol_4) {
  CeedScalar       eps = 4e-7;  // Finite difference step
  char             buf[128];
  const CeedScalar T           = 200;
  const CeedScalar rho         = 1.2;
  const CeedScalar p           = (HeatCapacityRatio(gas) - 1) * rho * gas->cv * T;
  const CeedScalar u_base      = 40;
  const CeedScalar u[3]        = {u_base, u_base * 1.1, u_base * 1.2};
  const CeedScalar Y0_left[5]  = {p, u[0], u[1], u[2], T};
  const CeedScalar Y0_right[5] = {1.2 * p, 1.2 * u[0], 1.2 * u[1], 1.2 * u[2], 1.2 * T};
  CeedScalar       normal[3]   = {1, 2, 3};

  PetscFunctionBeginUser;
  State left0  = StateFromY(gas, Y0_left);
  State right0 = StateFromY(gas, Y0_right);
  ScaleN(normal, 1 / sqrt(Dot3(normal, normal)), 3);

  for (int i = 0; i < 10; i++) {
    CeedScalar dFlux[5] = {0.}, dFlux_fd[5] = {0.};
    {  // Calculate dFlux using *_fwd function
      CeedScalar dY_right[5] = {0};
      CeedScalar dY_left[5]  = {0};

      if (i < 5) {
        dY_left[i] = Y0_left[i];
      } else {
        dY_right[i % 5] = Y0_right[i % 5];
      }
      State dleft0  = StateFromY_fwd(gas, left0, dY_left);
      State dright0 = StateFromY_fwd(gas, right0, dY_right);

      StateConservative dFlux_state = RiemannFlux_HLL_fwd(gas, left0, dleft0, right0, dright0, normal);
      UnpackState_U(dFlux_state, dFlux);
    }

    {  // Calculate dFlux_fd via finite difference approximation
      CeedScalar Y1_left[5]  = {Y0_left[0], Y0_left[1], Y0_left[2], Y0_left[3], Y0_left[4]};
      CeedScalar Y1_right[5] = {Y0_right[0], Y0_right[1], Y0_right[2], Y0_right[3], Y0_right[4]};
      CeedScalar Flux0[5], Flux1[5];

      if (i < 5) {
        Y1_left[i] *= 1 + eps;
      } else {
        Y1_right[i % 5] *= 1 + eps;
      }
      State left1  = StateFromY(gas, Y1_left);
      State right1 = StateFromY(gas, Y1_right);

      StateConservative Flux0_state = RiemannFlux_HLL(gas, left0, right0, normal);
      StateConservative Flux1_state = RiemannFlux_HLL(gas, left1, right1, normal);
      UnpackState_U(Flux0_state, Flux0);
      UnpackState_U(Flux1_state, Flux1);
      for (int j = 0; j < 5; j++) dFlux_fd[j] = (Flux1[j] - Flux0[j]) / eps;
    }

    snprintf(buf, sizeof buf, "RiemannFlux_HLL i=%d: Flux", i);
    PetscCall(CheckQWithTolerance(dFlux_fd, dFlux, dFlux_fd, buf, rtol_0, rtol_u, rtol_4));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Verify RiemannFlux_HLLC_fwd function against finite-difference approximation
static PetscErrorCode TestRiemannHLLC_fwd(NewtonianIdealGasContext gas, CeedScalar rtol_0, CeedScalar rtol_u, CeedScalar rtol_4) {
  CeedScalar       eps = 4e-7;  // Finite difference step
  char             buf[128];
  const CeedScalar T           = 200;
  const CeedScalar rho         = 1.2;
  const CeedScalar p           = (HeatCapacityRatio(gas) - 1) * rho * gas->cv * T;
  const CeedScalar u_base      = 40;
  const CeedScalar u[3]        = {u_base, u_base * 1.1, u_base * 1.2};
  const CeedScalar Y0_left[5]  = {p, u[0], u[1], u[2], T};
  const CeedScalar Y0_right[5] = {1.2 * p, 1.2 * u[0], 1.2 * u[1], 1.2 * u[2], 1.2 * T};
  CeedScalar       normal[3]   = {1, 2, 3};

  PetscFunctionBeginUser;
  State left0  = StateFromY(gas, Y0_left);
  State right0 = StateFromY(gas, Y0_right);
  ScaleN(normal, 1 / sqrt(Dot3(normal, normal)), 3);

  for (int i = 0; i < 10; i++) {
    CeedScalar dFlux[5] = {0.}, dFlux_fd[5] = {0.};
    {  // Calculate dFlux using *_fwd function
      CeedScalar dY_right[5] = {0};
      CeedScalar dY_left[5]  = {0};

      if (i < 5) {
        dY_left[i] = Y0_left[i];
      } else {
        dY_right[i % 5] = Y0_right[i % 5];
      }
      State dleft0  = StateFromY_fwd(gas, left0, dY_left);
      State dright0 = StateFromY_fwd(gas, right0, dY_right);

      StateConservative dFlux_state = RiemannFlux_HLLC_fwd(gas, left0, dleft0, right0, dright0, normal);
      UnpackState_U(dFlux_state, dFlux);
    }

    {  // Calculate dFlux_fd via finite difference approximation
      CeedScalar Y1_left[5]  = {Y0_left[0], Y0_left[1], Y0_left[2], Y0_left[3], Y0_left[4]};
      CeedScalar Y1_right[5] = {Y0_right[0], Y0_right[1], Y0_right[2], Y0_right[3], Y0_right[4]};
      CeedScalar Flux0[5], Flux1[5];

      if (i < 5) {
        Y1_left[i] *= 1 + eps;
      } else {
        Y1_right[i % 5] *= 1 + eps;
      }
      State left1  = StateFromY(gas, Y1_left);
      State right1 = StateFromY(gas, Y1_right);

      StateConservative Flux0_state = RiemannFlux_HLLC(gas, left0, right0, normal);
      StateConservative Flux1_state = RiemannFlux_HLLC(gas, left1, right1, normal);
      UnpackState_U(Flux0_state, Flux0);
      UnpackState_U(Flux1_state, Flux1);
      for (int j = 0; j < 5; j++) dFlux_fd[j] = (Flux1[j] - Flux0[j]) / eps;
    }

    snprintf(buf, sizeof buf, "RiemannFlux_HLLC i=%d: Flux", i);
    PetscCall(CheckQWithTolerance(dFlux_fd, dFlux, dFlux_fd, buf, rtol_0, rtol_u, rtol_4));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Verify ComputeHLLSpeeds_Roe_fwd function against finite-difference approximation
static PetscErrorCode TestComputeHLLSpeeds_Roe_fwd(NewtonianIdealGasContext gas, CeedScalar rtol) {
  CeedScalar       eps = 4e-7;  // Finite difference step
  char             buf[128];
  const CeedScalar T           = 200;
  const CeedScalar rho         = 1.2;
  const CeedScalar p           = (HeatCapacityRatio(gas) - 1) * rho * gas->cv * T;
  const CeedScalar u_base      = 40;
  const CeedScalar u[3]        = {u_base, u_base * 1.1, u_base * 1.2};
  const CeedScalar Y0_left[5]  = {p, u[0], u[1], u[2], T};
  const CeedScalar Y0_right[5] = {1.2 * p, 1.2 * u[0], 1.2 * u[1], 1.2 * u[2], 1.2 * T};
  CeedScalar       normal[3]   = {1, 2, 3};

  PetscFunctionBeginUser;
  State left0  = StateFromY(gas, Y0_left);
  State right0 = StateFromY(gas, Y0_right);
  ScaleN(normal, 1 / sqrt(Dot3(normal, normal)), 3);
  CeedScalar u_left0  = Dot3(left0.Y.velocity, normal);
  CeedScalar u_right0 = Dot3(right0.Y.velocity, normal);

  for (int i = 0; i < 10; i++) {
    CeedScalar ds_left, ds_right, ds_left_fd, ds_right_fd;
    {  // Calculate ds_{left,right} using *_fwd function
      CeedScalar dY_right[5] = {0};
      CeedScalar dY_left[5]  = {0};

      if (i < 5) {
        dY_left[i] = Y0_left[i];
      } else {
        dY_right[i % 5] = Y0_right[i % 5];
      }
      State      dleft0   = StateFromY_fwd(gas, left0, dY_left);
      State      dright0  = StateFromY_fwd(gas, right0, dY_right);
      CeedScalar du_left  = Dot3(dleft0.Y.velocity, normal);
      CeedScalar du_right = Dot3(dright0.Y.velocity, normal);

      CeedScalar s_left, s_right;  // Throw away
      ComputeHLLSpeeds_Roe_fwd(gas, left0, dleft0, u_left0, du_left, right0, dright0, u_right0, du_right, &s_left, &ds_left, &s_right, &ds_right);
    }

    {  // Calculate ds_{left,right}_fd via finite difference approximation
      CeedScalar Y1_left[5]  = {Y0_left[0], Y0_left[1], Y0_left[2], Y0_left[3], Y0_left[4]};
      CeedScalar Y1_right[5] = {Y0_right[0], Y0_right[1], Y0_right[2], Y0_right[3], Y0_right[4]};

      if (i < 5) {
        Y1_left[i] *= 1 + eps;
      } else {
        Y1_right[i % 5] *= 1 + eps;
      }
      State      left1    = StateFromY(gas, Y1_left);
      State      right1   = StateFromY(gas, Y1_right);
      CeedScalar u_left1  = Dot3(left1.Y.velocity, normal);
      CeedScalar u_right1 = Dot3(right1.Y.velocity, normal);

      CeedScalar s_left0, s_right0, s_left1, s_right1;
      ComputeHLLSpeeds_Roe(gas, left0, u_left0, right0, u_right0, &s_left0, &s_right0);
      ComputeHLLSpeeds_Roe(gas, left1, u_left1, right1, u_right1, &s_left1, &s_right1);
      ds_left_fd  = (s_left1 - s_left0) / eps;
      ds_right_fd = (s_right1 - s_right0) / eps;
    }

    snprintf(buf, sizeof buf, "ComputeHLLSpeeds_Roe i=%d:", i);
    {
      CeedScalar divisor_threshold = 10 * CEED_EPSILON;
      CeedScalar ds_left_err, ds_right_err;

      ds_left_err  = RelativeError(ds_left_fd, ds_left, ds_left_fd, divisor_threshold);
      ds_right_err = RelativeError(ds_right_fd, ds_right, ds_right_fd, divisor_threshold);
      if (fabs(ds_left_err) >= rtol) printf("%s ds_left error %g (expected %.10e, got %.10e)\n", buf, ds_left_err, ds_left_fd, ds_left);
      if (fabs(ds_right_err) >= rtol) printf("%s ds_right error %g (expected %.10e, got %.10e)\n", buf, ds_right_err, ds_right_fd, ds_right);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Verify TotalSpecificEnthalpy_fwd function against finite-difference approximation
static PetscErrorCode TestTotalSpecificEnthalpy_fwd(NewtonianIdealGasContext gas, CeedScalar rtol) {
  CeedScalar       eps = 4e-7;  // Finite difference step
  char             buf[128];
  const CeedScalar T      = 200;
  const CeedScalar rho    = 1.2;
  const CeedScalar p      = (HeatCapacityRatio(gas) - 1) * rho * gas->cv * T;
  const CeedScalar u_base = 40;
  const CeedScalar u[3]   = {u_base, u_base * 1.1, u_base * 1.2};
  const CeedScalar Y0[5]  = {p, u[0], u[1], u[2], T};

  PetscFunctionBeginUser;
  State state0 = StateFromY(gas, Y0);

  for (int i = 0; i < 5; i++) {
    CeedScalar dH, dH_fd;
    {  // Calculate dH using *_fwd function
      CeedScalar dY[5] = {0};

      dY[i]         = Y0[i];
      State dstate0 = StateFromY_fwd(gas, state0, dY);
      dH            = TotalSpecificEnthalpy_fwd(gas, state0, dstate0);
    }

    {  // Calculate dH_fd via finite difference approximation
      CeedScalar H0, H1;
      CeedScalar Y1[5] = {Y0[0], Y0[1], Y0[2], Y0[3], Y0[4]};
      Y1[i] *= 1 + eps;
      State state1 = StateFromY(gas, Y1);

      H0    = TotalSpecificEnthalpy(gas, state0);
      H1    = TotalSpecificEnthalpy(gas, state1);
      dH_fd = (H1 - H0) / eps;
    }

    snprintf(buf, sizeof buf, "TotalSpecificEnthalpy i=%d:", i);
    {
      CeedScalar divisor_threshold = 10 * CEED_EPSILON;
      CeedScalar dH_err;

      dH_err = RelativeError(dH_fd, dH, dH_fd, divisor_threshold);
      if (fabs(dH_err) >= rtol) printf("%s dH error %g (expected %.10e, got %.10e)\n", buf, dH_err, dH_fd, dH);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Verify RoeSetup_fwd function against finite-difference approximation
static PetscErrorCode TestRowSetup_fwd(NewtonianIdealGasContext gas, CeedScalar rtol) {
  CeedScalar       eps = 4e-7;  // Finite difference step
  char             buf[128];
  const CeedScalar rho0[2] = {1.2, 1.4};

  PetscFunctionBeginUser;
  for (int i = 0; i < 2; i++) {
    RoeWeights dR, dR_fd;
    {  // Calculate using *_fwd function
      CeedScalar drho[5] = {0};

      drho[i] = rho0[i];
      dR      = RoeSetup_fwd(rho0[0], rho0[1], drho[0], drho[1]);
    }

    {  // Calculate via finite difference approximation
      RoeWeights R0, R1;
      CeedScalar rho1[5] = {rho0[0], rho0[1]};
      rho1[i] *= 1 + eps;

      R0          = RoeSetup(rho0[0], rho0[1]);
      R1          = RoeSetup(rho1[0], rho1[1]);
      dR_fd.left  = (R1.left - R0.left) / eps;
      dR_fd.right = (R1.right - R0.right) / eps;
    }

    snprintf(buf, sizeof buf, "RoeSetup i=%d:", i);
    {
      CeedScalar divisor_threshold = 10 * CEED_EPSILON;
      RoeWeights dR_err;

      dR_err.left  = RelativeError(dR_fd.left, dR.left, dR_fd.left, divisor_threshold);
      dR_err.right = RelativeError(dR_fd.right, dR.right, dR_fd.right, divisor_threshold);
      if (fabs(dR_err.left) >= rtol) printf("%s dR.left error %g (expected %.10e, got %.10e)\n", buf, dR_err.left, dR_fd.left, dR.left);
      if (fabs(dR_err.right) >= rtol) printf("%s dR.right error %g (expected %.10e, got %.10e)\n", buf, dR_err.right, dR_fd.right, dR.right);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Test Riemann solver related `*_fwd` functions via finite-difference approximation
static PetscErrorCode RiemannSolverUnitTests(NewtonianIdealGasContext gas, CeedScalar rtol) {
  PetscFunctionBeginUser;
  PetscCall(TestRiemannHLL_fwd(gas, rtol, rtol, rtol));
  PetscCall(TestRiemannHLLC_fwd(gas, rtol, rtol, rtol));
  PetscCall(TestComputeHLLSpeeds_Roe_fwd(gas, rtol));
  PetscCall(TestTotalSpecificEnthalpy_fwd(gas, rtol));
  PetscCall(TestRowSetup_fwd(gas, rtol));
  PetscFunctionReturn(PETSC_SUCCESS);
}
