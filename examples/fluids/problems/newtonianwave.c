// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Blasius Boundary Layer

#include "../navierstokes.h"
#include "../qfunctions/newtonwave.h"
#include "../qfunctions/freestream_bc.h"

PetscErrorCode NS_NEWTONIAN_WAVE(ProblemData *problem, DM dm, void *ctx) {
  User      user    = *(User *)ctx;
  MPI_Comm  comm    = PETSC_COMM_WORLD;
  NewtonWaveContext newtwave_ctx;
  FreestreamContext freestream_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext newtwave_context, freestream_context;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx));

  // *INDENT-OFF*
  switch (user->phys->state_var) {
  case STATEVAR_CONSERVATIVE:
    problem->ics.qfunction                           = IC_NewtonianWave_Conserv;
    problem->ics.qfunction_loc                       = IC_NewtonianWave_Conserv_loc;
    problem->apply_freestream.qfunction              = Freestream_Conserv;
    problem->apply_freestream.qfunction_loc          = Freestream_Conserv_loc;
    problem->apply_freestream_jacobian.qfunction     = Freestream_Jacobian_Conserv;
    problem->apply_freestream_jacobian.qfunction_loc = Freestream_Jacobian_Conserv_loc;
  case STATEVAR_PRIMITIVE:
    problem->ics.qfunction                           = IC_NewtonianWave_Prim;
    problem->ics.qfunction_loc                       = IC_NewtonianWave_Prim_loc;
    problem->apply_freestream.qfunction              = Freestream_Prim;
    problem->apply_freestream.qfunction_loc          = Freestream_Prim_loc;
    problem->apply_freestream_jacobian.qfunction     = Freestream_Jacobian_Prim;
    problem->apply_freestream_jacobian.qfunction_loc = Freestream_Jacobian_Prim_loc;
  }
  // *INDENT-ON*

  // -- Option Defaults
  CeedScalar U_inf[3]     = {0.};   // m/s
  CeedScalar T_inf        = 288.;   // K
  CeedScalar P_inf        = 1.01e5; // Pa
  CeedScalar epicenter[3] = {0.};   // m
  CeedScalar width        = 0.002;  // m
  CeedScalar amplitude    = 0.1;   // -

  PetscOptionsBegin(comm, NULL, "Options for NEWTONIAN_WAVE problem", NULL);
  PetscInt narray=3;
  PetscCall(PetscOptionsScalarArray("-velocity_infinity",
                                    "Velocity at freestream condition",
                                    NULL, U_inf, &narray, NULL));
  PetscCheck(narray == 3, comm, PETSC_ERR_ARG_SIZ,
             "-velocity_infinity should recieve array of size 3, instead recieved size %"
             PetscInt_FMT".", narray);

  PetscCall(PetscOptionsScalar("-temperature_infinity",
                               "Temperature at freestream condition",
                               NULL, T_inf, &T_inf, NULL));
  PetscCall(PetscOptionsScalar("-pressure_infinity",
                               "Pressure at freestream condition",
                               NULL, P_inf, &P_inf, NULL));

  narray = 3;
  PetscCall(PetscOptionsScalarArray("-epicenter", "Coordinates of center of wave",
                                    NULL, epicenter, &narray, NULL));
  PetscCheck(narray == 3, comm, PETSC_ERR_ARG_SIZ,
             "-epicenter should recieve array of size 3, instead recieved size %"
             PetscInt_FMT".", narray);
  PetscCall(PetscOptionsScalar("-width", "Width parameter for perturbation size",
                               NULL, width, &width, NULL));
  PetscCall(PetscOptionsScalar("-amplitude", "Amplitude of the perturbation",
                               NULL, amplitude, &amplitude, NULL));
  PetscOptionsEnd();

  PetscScalar meter  = user->units->meter;
  PetscScalar second = user->units->second;
  PetscScalar Kelvin = user->units->Kelvin;
  PetscScalar Pascal = user->units->Pascal;

  T_inf *= Kelvin;
  P_inf *= Pascal;
  width *= meter;
  for (int i=0; i<3; i++) {
    U_inf[i]     *= meter / second;
    epicenter[i] *= meter;
  }

  // -- Set newtwave_ctx struct values
  PetscCall(PetscCalloc1(1, &newtwave_ctx));
  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context,
                              CEED_MEM_HOST, &newtonian_ig_ctx);
  State S_infty;
  {
    CeedScalar Y[5] = {P_inf, U_inf[0], U_inf[1], U_inf[2], T_inf};
    CeedScalar x[3] = {0.};
    S_infty = StateFromY(newtonian_ig_ctx, Y, x);
  }

  newtwave_ctx->amplitude = amplitude;
  newtwave_ctx->width     = width;
  newtwave_ctx->S_infty   = S_infty;
  newtwave_ctx->newt_ctx  = *newtonian_ig_ctx;
  PetscCall(PetscArraycpy(newtwave_ctx->epicenter, epicenter, 3));

  // -- Set freestream_ctx struct values
  PetscCall(PetscCalloc1(1, &freestream_ctx));
  freestream_ctx->newtonian_ctx = *newtonian_ig_ctx;
  freestream_ctx->S_infty       = S_infty;

  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context,
                                  &newtonian_ig_ctx);

  CeedQFunctionContextCreate(user->ceed, &newtwave_context);
  CeedQFunctionContextSetData(newtwave_context, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof(*newtwave_ctx), newtwave_ctx);
  CeedQFunctionContextSetDataDestroy(newtwave_context, CEED_MEM_HOST,
                                     FreeContextPetsc);
  CeedQFunctionContextDestroy(&problem->ics.qfunction_context);
  problem->ics.qfunction_context = newtwave_context;

  CeedQFunctionContextCreate(user->ceed, &freestream_context);
  CeedQFunctionContextSetData(freestream_context, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof(*freestream_ctx), freestream_ctx);
  CeedQFunctionContextSetDataDestroy(freestream_context, CEED_MEM_HOST,
                                     FreeContextPetsc);
  problem->apply_freestream.qfunction_context = freestream_context;
  problem->apply_freestream_jacobian.qfunction_context = freestream_context;

  PetscFunctionReturn(0);
}
