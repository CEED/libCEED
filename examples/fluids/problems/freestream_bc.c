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

PetscErrorCode FreestreamBCSetup(ProblemData *problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx) {
  User                 user = *(User *)ctx;
  MPI_Comm             comm = PETSC_COMM_WORLD;
  FreestreamContext    freestream_ctx;
  CeedQFunctionContext freestream_context;
  RiemannSolverType    riemann = RIEMANN_HLLC;
  PetscFunctionBeginUser;

  // -- Option Defaults
  CeedScalar U_inf[3] = {0.};    // m/s
  CeedScalar T_inf    = 288.;    // K
  CeedScalar P_inf    = 1.01e5;  // Pa

  PetscOptionsBegin(comm, NULL, "Options for Freestream boundary condition", NULL);
  PetscCall(PetscOptionsEnum("-freestream_riemann", "Riemann solver to use in freestream boundary condition", NULL, RiemannSolverTypes,
                             (PetscEnum)riemann, (PetscEnum *)&riemann, NULL));
  PetscBool set    = PETSC_FALSE;
  PetscInt  narray = 3;
  PetscCall(PetscOptionsScalarArray("-freestream_velocity", "Velocity at freestream condition", NULL, U_inf, &narray, &set));
  PetscCheck(narray == 3 || !set, comm, PETSC_ERR_ARG_SIZ,
             "-freestream_velocity should recieve array of size 3, instead recieved size %" PetscInt_FMT ".", narray);

  PetscCall(PetscOptionsScalar("-freestream_temperature", "Temperature at freestream condition", NULL, T_inf, &T_inf, NULL));
  PetscCall(PetscOptionsScalar("-freestream_pressure", "Pressure at freestream condition", NULL, P_inf, &P_inf, NULL));
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

  PetscScalar meter  = user->units->meter;
  PetscScalar second = user->units->second;
  PetscScalar Kelvin = user->units->Kelvin;
  PetscScalar Pascal = user->units->Pascal;

  T_inf *= Kelvin;
  P_inf *= Pascal;
  for (int i = 0; i < 3; i++) U_inf[i] *= meter / second;

  State S_infty;
  {
    CeedScalar Y[5] = {P_inf, U_inf[0], U_inf[1], U_inf[2], T_inf};
    CeedScalar x[3] = {0.};
    S_infty         = StateFromY(newtonian_ig_ctx, Y, x);
  }

  // -- Set freestream_ctx struct values
  PetscCall(PetscCalloc1(1, &freestream_ctx));
  freestream_ctx->newtonian_ctx = *newtonian_ig_ctx;
  freestream_ctx->S_infty       = S_infty;

  CeedQFunctionContextCreate(user->ceed, &freestream_context);
  CeedQFunctionContextSetData(freestream_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*freestream_ctx), freestream_ctx);
  CeedQFunctionContextSetDataDestroy(freestream_context, CEED_MEM_HOST, FreeContextPetsc);
  problem->apply_freestream.qfunction_context          = freestream_context;
  problem->apply_freestream_jacobian.qfunction_context = freestream_context;

  PetscFunctionReturn(0);
};
