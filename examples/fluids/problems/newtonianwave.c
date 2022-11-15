// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Newtonian Wave problem

#include "../navierstokes.h"
#include "../qfunctions/freestream_bc_type.h"
#include "../qfunctions/newtonwave.h"

PetscErrorCode NS_NEWTONIAN_WAVE(ProblemData *problem, DM dm, void *ctx) {
  User                     user = *(User *)ctx;
  MPI_Comm                 comm = PETSC_COMM_WORLD;
  NewtonWaveContext        newtwave_ctx;
  FreestreamContext        freestream_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext     newtwave_context;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx));

  // *INDENT-OFF*
  switch (user->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      problem->ics.qfunction     = IC_NewtonianWave_Conserv;
      problem->ics.qfunction_loc = IC_NewtonianWave_Conserv_loc;
    case STATEVAR_PRIMITIVE:
      problem->ics.qfunction     = IC_NewtonianWave_Prim;
      problem->ics.qfunction_loc = IC_NewtonianWave_Prim_loc;
  }
  // *INDENT-ON*

  // -- Option Defaults
  CeedScalar epicenter[3] = {0.};   // m
  CeedScalar width        = 0.002;  // m
  CeedScalar amplitude    = 0.1;    // -

  PetscOptionsBegin(comm, NULL, "Options for NEWTONIAN_WAVE problem", NULL);
  PetscInt narray = 3;
  PetscCall(PetscOptionsScalarArray("-epicenter", "Coordinates of center of wave", NULL, epicenter, &narray, NULL));
  PetscCheck(narray == 3, comm, PETSC_ERR_ARG_SIZ, "-epicenter should recieve array of size 3, instead recieved size %" PetscInt_FMT ".", narray);
  PetscCall(PetscOptionsScalar("-width", "Width parameter for perturbation size", NULL, width, &width, NULL));
  PetscCall(PetscOptionsScalar("-amplitude", "Amplitude of the perturbation", NULL, amplitude, &amplitude, NULL));
  PetscOptionsEnd();

  width *= user->units->meter;
  for (int i = 0; i < 3; i++) epicenter[i] *= user->units->meter;

  PetscCall(FreestreamBCSetup(problem, dm, ctx));

  // -- Set newtwave_ctx struct values
  PetscCall(PetscCalloc1(1, &newtwave_ctx));
  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context, CEED_MEM_HOST, &newtonian_ig_ctx);
  CeedQFunctionContextGetData(problem->apply_freestream.qfunction_context, CEED_MEM_HOST, &freestream_ctx);

  newtwave_ctx->amplitude = amplitude;
  newtwave_ctx->width     = width;
  newtwave_ctx->S_infty   = freestream_ctx->S_infty;
  newtwave_ctx->newt_ctx  = *newtonian_ig_ctx;
  PetscCall(PetscArraycpy(newtwave_ctx->epicenter, epicenter, 3));

  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context, &newtonian_ig_ctx);
  CeedQFunctionContextRestoreData(problem->apply_freestream.qfunction_context, &freestream_ctx);

  CeedQFunctionContextCreate(user->ceed, &newtwave_context);
  CeedQFunctionContextSetData(newtwave_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*newtwave_ctx), newtwave_ctx);
  CeedQFunctionContextSetDataDestroy(newtwave_context, CEED_MEM_HOST, FreeContextPetsc);
  CeedQFunctionContextDestroy(&problem->ics.qfunction_context);
  problem->ics.qfunction_context = newtwave_context;

  PetscFunctionReturn(0);
}
