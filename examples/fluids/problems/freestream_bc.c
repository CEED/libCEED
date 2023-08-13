// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Freestream boundary condition

#include "../qfunctions/freestream_bc.h"

#include <ceed.h>
#include <petscdm.h>

#include "../navierstokes.h"
#include "../qfunctions/newtonian_types.h"

static const char *const RiemannSolverTypes[] = {"hll", "hllc", "RiemannSolverTypes", "RIEMANN_", NULL};

PetscErrorCode FreestreamBCSetup(ProblemData *problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx, const StatePrimitive *reference) {
  User                 user = *(User *)ctx;
  MPI_Comm             comm = user->comm;
  Ceed                 ceed = user->ceed;
  FreestreamContext    freestream_ctx;
  CeedQFunctionContext freestream_context;
  RiemannFluxType      riemann = RIEMANN_HLLC;
  PetscFunctionBeginUser;
  PetscScalar meter  = user->units->meter;
  PetscScalar second = user->units->second;
  PetscScalar Kelvin = user->units->Kelvin;
  PetscScalar Pascal = user->units->Pascal;

  // -- Option Defaults

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

  const CeedScalar x[3]    = {0.};
  State            S_infty = StateFromPrimitive(newtonian_ig_ctx, Y_inf, x);

  // -- Set freestream_ctx struct values
  PetscCall(PetscCalloc1(1, &freestream_ctx));
  freestream_ctx->newtonian_ctx = *newtonian_ig_ctx;
  freestream_ctx->S_infty       = S_infty;

  PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &freestream_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(freestream_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*freestream_ctx), freestream_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(freestream_context, CEED_MEM_HOST, FreeContextPetsc));
  problem->apply_freestream.qfunction_context = freestream_context;
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(freestream_context, &problem->apply_freestream_jacobian.qfunction_context));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const char *const OutflowTypes[] = {"RIEMANN", "PRESSURE", "OutflowType", "OUTFLOW_", NULL};
typedef enum {
  OUTFLOW_RIEMANN,
  OUTFLOW_PRESSURE,
} OutflowType;

PetscErrorCode OutflowBCSetup(ProblemData *problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx, const StatePrimitive *reference) {
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
