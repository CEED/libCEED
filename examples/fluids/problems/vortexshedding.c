// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up vortex shedding problem behind a cylinder

#include "../navierstokes.h"
#include "../qfunctions/vortexshedding.h"
#include "ceed/ceed-f64.h"
#include "stg_shur14.h"

PetscErrorCode NS_VORTEXSHEDDING(ProblemData *problem, DM dm, void *ctx) {

  PetscInt ierr;
  User      user    = *(User *)ctx;
  MPI_Comm  comm    = PETSC_COMM_WORLD;
  PetscBool use_stg = PETSC_FALSE;
  VortexsheddingContext vortexshedding_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext vortexshedding_context;

  PetscFunctionBeginUser;
  ierr = NS_NEWTONIAN_IG(problem, dm, ctx); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &vortexshedding_ctx); CHKERRQ(ierr);

  // ------------------------------------------------------
  //               SET UP Vortex Shedding
  // ------------------------------------------------------
  CeedQFunctionContextDestroy(&problem->ics.qfunction_context);
  problem->ics.qfunction        = ICsVortexshedding;
  problem->ics.qfunction_loc    = ICsVortexshedding_loc;
  if (user->phys->state_var == STATEVAR_CONSERVATIVE) {
    problem->apply_inflow.qfunction      = Vortexshedding_Inflow;
    problem->apply_inflow.qfunction_loc  = Vortexshedding_Inflow_loc;
    problem->apply_outflow.qfunction     = Vortexshedding_Outflow;
    problem->apply_outflow.qfunction_loc = Vortexshedding_Outflow_loc;
  }


  CeedScalar U_in               = 1.0;         // m/s
  CeedScalar T_in               = 300.;        // K
  CeedScalar P0                 = 1.e5;        // Pa
  PetscBool  weakT              = PETSC_FALSE; // weak density or temperature

  PetscOptionsBegin(comm, NULL, "Options for VORTEX SHEDDING problem", NULL);
  ierr = PetscOptionsScalar("-velocity_inflow",
                            "Velocity at inflow",
                            NULL, U_in, &U_in, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-P0", "Pressure at outflow",
                            NULL, P0, &P0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-temperature_inflow", "Temperature at inflow",
                            NULL, T_in, &T_in, NULL); CHKERRQ(ierr);

  PetscOptionsEnd();

  PetscScalar meter  = user->units->meter;
  PetscScalar second = user->units->second;
  PetscScalar Kelvin = user->units->Kelvin;
  PetscScalar Pascal = user->units->Pascal;

  T_in   *= Kelvin;
  P0     *= Pascal;
  U_in   *= meter / second;

  //-- Setup Problem information
  CeedScalar L, H;
  {
    PetscReal domain_min[3], domain_max[3], domain_size[3];
    ierr = DMGetBoundingBox(dm, domain_min, domain_max); CHKERRQ(ierr);
    // compute L and H

  }
  // Some properties depend on parameters from NewtonianIdealGas
  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context,
                              CEED_MEM_HOST, &newtonian_ig_ctx);

  vortexshedding_ctx->L        = L;
  vortexshedding_ctx->H        = H;
  vortexshedding_ctx->d        = d;
  vortexshedding_ctx->T_in     = T_in;
  vortexshedding_ctx->P0       = P0;
  vortexshedding_ctx->U_in     = U_in;
  vortexshedding_ctx->implicit = user->phys->implicit;



  vortexshedding_ctx->newtonian_ctx = *newtonian_ig_ctx;
  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context,
                                  &newtonian_ig_ctx);

  CeedQFunctionContextCreate(user->ceed, &vortexshedding_context);
  CeedQFunctionContextSetData(vortexshedding_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*vortexshedding_ctx), vortexshedding_ctx);
  CeedQFunctionContextSetDataDestroy(vortexshedding_context, CEED_MEM_HOST,
                                     FreeContextPetsc);
  problem->ics.qfunction_context = vortexshedding_context;
  CeedQFunctionContextReferenceCopy(vortexshedding_context,
                                    &problem->apply_inflow.qfunction_context);
  CeedQFunctionContextReferenceCopy(vortexshedding_context,
                                    &problem->apply_outflow.qfunction_context);


  PetscFunctionReturn(0);
}