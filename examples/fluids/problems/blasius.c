// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Blasius Boundary Layer

#include "../navierstokes.h"
#include "../qfunctions/newtonian.h"
#include "../qfunctions/blasius.h"

#ifndef blasius_context_struct
#define blasius_context_struct
typedef struct BlasiusContext_ *BlasiusContext;
struct BlasiusContext_ {
  bool implicit;
  int stabilization; // See StabilizationType: 0=none, 1=SU, 2=SUPG
  NewtonianIdealGasContext newt_ctx;

};

#endif
PetscErrorCode NS_BLASIUS(ProblemData *problem, DM dm, void *setup_ctx,
                          void *ctx) {

  PetscInt ierr;
  ierr = NS_NEWTONIAN_IG(problem, dm, setup_ctx, ctx); CHKERRQ(ierr);
  SetupContext      setup_context = *(SetupContext *)setup_ctx;
  User              user = *(User *)ctx;
  MPI_Comm          comm = PETSC_COMM_WORLD;
  PetscFunctionBeginUser;

  // ------------------------------------------------------
  //               SET UP Blasius
  // ------------------------------------------------------
  problem->ics                     = ICsBlasius;
  problem->ics_loc                 = ICsBlasius_loc;
  problem->apply_inflow            = Blasius_Inflow;
  problem->apply_inflow_loc        = Blasius_Inflow_loc;
  problem->apply_outflow           = Blasius_Outflow;
  problem->apply_outflow_loc       = Blasius_Outflow_loc;
  problem->setup_ctx               = SetupContext_BLASIUS;
  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  // ierr = PetscOptionsBegin(comm, NULL, "Options for DENSITY_CURRENT problem",
  //                          NULL); CHKERRQ(ierr);
  // ierr = PetscOptionsEnd(); CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

PetscErrorCode SetupContext_BLASIUS(Ceed ceed, CeedData ceed_data,
                                    AppCtx app_ctx, SetupContext setup_ctx, Physics phys) {
  PetscFunctionBeginUser;
  CeedQFunctionContextCreate(ceed, &ceed_data->setup_context);
  CeedQFunctionContextSetData(ceed_data->setup_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*setup_ctx), setup_ctx);
  CeedQFunctionContextCreate(ceed, &ceed_data->newt_ig_context);
  CeedQFunctionContextSetData(ceed_data->newt_ig_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*phys->newtonian_ig_ctx), phys->newtonian_ig_ctx);
  phys->has_neumann = PETSC_TRUE;
  if (ceed_data->qf_ics)
    CeedQFunctionSetContext(ceed_data->qf_ics, ceed_data->newt_ig_context);
  if (ceed_data->qf_rhs_vol)
    CeedQFunctionSetContext(ceed_data->qf_rhs_vol, ceed_data->newt_ig_context);
  if (ceed_data->qf_ifunction_vol)
    CeedQFunctionSetContext(ceed_data->qf_ifunction_vol,
                            ceed_data->newt_ig_context);
  if (ceed_data->qf_apply_inflow)
    CeedQFunctionSetContext(ceed_data->qf_apply_inflow, ceed_data->newt_ig_context);
  if (ceed_data->qf_apply_outflow)
    CeedQFunctionSetContext(ceed_data->qf_apply_outflow,
                            ceed_data->newt_ig_context);
  PetscFunctionReturn(0);
}
