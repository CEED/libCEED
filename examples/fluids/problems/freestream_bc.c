// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Freestream boundary condition

#include "../qfunctions/freestream_bc.h"

#include "../navierstokes.h"
#include "../qfunctions/newtonian_types.h"

typedef enum {
  RIEMANN_HLL,
  RIEMANN_HLLC,
} RiemannSolverType;
static const char *const RiemannSolverTypes[] = {"hll", "hllc", "RiemannSolverTypes", "RIEMANN_", NULL};

PetscErrorCode FreestreamBCSetup(ProblemData *problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx, const StatePrimitive *reference) {
  User                 user = *(User *)ctx;
  MPI_Comm             comm = PETSC_COMM_WORLD;
  FreestreamContext    freestream_ctx;
  CeedQFunctionContext freestream_context;
  RiemannSolverType    riemann = RIEMANN_HLLC;
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

  CeedQFunctionContextCreate(user->ceed, &freestream_context);
  CeedQFunctionContextSetData(freestream_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*freestream_ctx), freestream_ctx);
  CeedQFunctionContextSetDataDestroy(freestream_context, CEED_MEM_HOST, FreeContextPetsc);
  problem->apply_freestream.qfunction_context = freestream_context;
  CeedQFunctionContextReferenceCopy(freestream_context, &problem->apply_freestream_jacobian.qfunction_context);
  PetscFunctionReturn(0);
};
