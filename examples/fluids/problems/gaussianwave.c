// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Gaussian Wave problem

#include "../qfunctions/gaussianwave.h"

#include <ceed.h>
#include <petscdm.h>

#include "../navierstokes.h"
#include "../qfunctions/freestream_bc_type.h"

PetscErrorCode NS_GAUSSIAN_WAVE(ProblemData *problem, DM dm, void *ctx, SimpleBC bc) {
  User                     user = *(User *)ctx;
  MPI_Comm                 comm = user->comm;
  Ceed                     ceed = user->ceed;
  GaussianWaveContext      gausswave_ctx;
  FreestreamContext        freestream_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext     gausswave_context;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx, bc));

  switch (user->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      problem->ics.qfunction     = IC_GaussianWave_Conserv;
      problem->ics.qfunction_loc = IC_GaussianWave_Conserv_loc;
      break;
    case STATEVAR_PRIMITIVE:
      problem->ics.qfunction     = IC_GaussianWave_Prim;
      problem->ics.qfunction_loc = IC_GaussianWave_Prim_loc;
      break;
    case STATEVAR_ENTROPY:
      problem->ics.qfunction     = IC_GaussianWave_Entropy;
      problem->ics.qfunction_loc = IC_GaussianWave_Entropy_loc;
      break;
  }

  // -- Option Defaults
  CeedScalar epicenter[3] = {0.};   // m
  CeedScalar width        = 0.002;  // m
  CeedScalar amplitude    = 0.1;    // -

  PetscOptionsBegin(comm, NULL, "Options for GAUSSIAN_WAVE problem", NULL);
  PetscInt narray = 3;
  PetscCall(PetscOptionsScalarArray("-epicenter", "Coordinates of center of wave", NULL, epicenter, &narray, NULL));
  PetscCheck(narray == 3, comm, PETSC_ERR_ARG_SIZ, "-epicenter should recieve array of size 3, instead recieved size %" PetscInt_FMT ".", narray);
  PetscCall(PetscOptionsScalar("-width", "Width parameter for perturbation size", NULL, width, &width, NULL));
  PetscCall(PetscOptionsScalar("-amplitude", "Amplitude of the perturbation", NULL, amplitude, &amplitude, NULL));
  PetscOptionsEnd();

  width *= user->units->meter;
  for (int i = 0; i < 3; i++) epicenter[i] *= user->units->meter;

  // -- Set gausswave_ctx struct values
  PetscCall(PetscCalloc1(1, &gausswave_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context, CEED_MEM_HOST, &newtonian_ig_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_freestream.qfunction_context, CEED_MEM_HOST, &freestream_ctx));

  gausswave_ctx->amplitude = amplitude;
  gausswave_ctx->width     = width;
  gausswave_ctx->S_infty   = freestream_ctx->S_infty;
  gausswave_ctx->newt_ctx  = *newtonian_ig_ctx;
  PetscCall(PetscArraycpy(gausswave_ctx->epicenter, epicenter, 3));

  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context, &newtonian_ig_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_freestream.qfunction_context, &freestream_ctx));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &gausswave_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(gausswave_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*gausswave_ctx), gausswave_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(gausswave_context, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->ics.qfunction_context));
  problem->ics.qfunction_context = gausswave_context;

  PetscFunctionReturn(PETSC_SUCCESS);
}
