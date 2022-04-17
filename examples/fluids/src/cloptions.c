// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Command line option processing for Navier-Stokes example using PETSc

#include "../navierstokes.h"

// Register problems to be available on the command line
PetscErrorCode RegisterProblems_NS(AppCtx app_ctx) {


  app_ctx->problems = NULL;
  PetscErrorCode   ierr;
  PetscFunctionBeginUser;

  ierr = PetscFunctionListAdd(&app_ctx->problems, "density_current",
                              NS_DENSITY_CURRENT); CHKERRQ(ierr);

  ierr = PetscFunctionListAdd(&app_ctx->problems, "euler_vortex",
                              NS_EULER_VORTEX); CHKERRQ(ierr);

  ierr = PetscFunctionListAdd(&app_ctx->problems, "advection",
                              NS_ADVECTION); CHKERRQ(ierr);

  ierr = PetscFunctionListAdd(&app_ctx->problems, "advection2d",
                              NS_ADVECTION2D); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx,
    SimpleBC bc) {

  PetscBool ceed_flag = PETSC_FALSE;
  PetscBool problem_flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  PetscOptionsBegin(comm, NULL, "Navier-Stokes in PETSc with libCEED",
                    NULL);

  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, app_ctx->ceed_resource, app_ctx->ceed_resource,
                            sizeof(app_ctx->ceed_resource), &ceed_flag); CHKERRQ(ierr);

  app_ctx->test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test", "Run in test mode",
                          NULL, app_ctx->test_mode, &app_ctx->test_mode, NULL); CHKERRQ(ierr);

  app_ctx->test_tol = 1E-11;
  ierr = PetscOptionsScalar("-compare_final_state_atol",
                            "Test absolute tolerance",
                            NULL, app_ctx->test_tol, &app_ctx->test_tol, NULL); CHKERRQ(ierr);

  ierr = PetscOptionsString("-compare_final_state_filename", "Test filename",
                            NULL, app_ctx->file_path, app_ctx->file_path,
                            sizeof(app_ctx->file_path), NULL); CHKERRQ(ierr);

  ierr = PetscOptionsFList("-problem", "Problem to solve", NULL,
                           app_ctx->problems,
                           app_ctx->problem_name, app_ctx->problem_name, sizeof(app_ctx->problem_name),
                           &problem_flag); CHKERRQ(ierr);

  app_ctx->viz_refine = 0;
  ierr = PetscOptionsInt("-viz_refine",
                         "Regular refinement levels for visualization",
                         NULL, app_ctx->viz_refine, &app_ctx->viz_refine, NULL); CHKERRQ(ierr);

  app_ctx->output_freq = 10;
  ierr = PetscOptionsInt("-output_freq",
                         "Frequency of output, in number of steps",
                         NULL, app_ctx->output_freq, &app_ctx->output_freq, NULL); CHKERRQ(ierr);

  app_ctx->cont_steps = 0;
  ierr = PetscOptionsInt("-continue", "Continue from previous solution",
                         NULL, app_ctx->cont_steps, &app_ctx->cont_steps, NULL); CHKERRQ(ierr);

  app_ctx->degree = 1;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of finite elements",
                         NULL, app_ctx->degree, &app_ctx->degree, NULL); CHKERRQ(ierr);

  app_ctx->q_extra = 2;
  ierr = PetscOptionsInt("-q_extra", "Number of extra quadrature points",
                         NULL, app_ctx->q_extra, &app_ctx->q_extra, NULL); CHKERRQ(ierr);

  ierr = PetscStrncpy(app_ctx->output_dir, ".", 2); CHKERRQ(ierr);
  ierr = PetscOptionsString("-output_dir", "Output directory",
                            NULL, app_ctx->output_dir, app_ctx->output_dir,
                            sizeof(app_ctx->output_dir), NULL); CHKERRQ(ierr);

  // Provide default ceed resource if not specified
  if (!ceed_flag) {
    const char *ceed_resource = "/cpu/self";
    strncpy(app_ctx->ceed_resource, ceed_resource, 10);
  }

  // Provide default problem if not specified
  if (!problem_flag) {
    const char *problem_name = "density_current";
    strncpy(app_ctx->problem_name, problem_name, 16);
  }

  // Wall Boundary Conditions
  bc->num_wall = 16;
  PetscBool flg;
  ierr = PetscOptionsIntArray("-bc_wall",
                              "Face IDs to apply wall BC",
                              NULL, bc->walls, &bc->num_wall, NULL); CHKERRQ(ierr);
  bc->num_comps = 5;
  ierr = PetscOptionsIntArray("-wall_comps",
                              "An array of constrained component numbers",
                              NULL, bc->wall_comps, &bc->num_comps, &flg); CHKERRQ(ierr);
  // Slip Boundary Conditions
  for (PetscInt j=0; j<3; j++) {
    bc->num_slip[j] = 16;
    PetscBool flg;
    const char *flags[3] = {"-bc_slip_x", "-bc_slip_y", "-bc_slip_z"};
    ierr = PetscOptionsIntArray(flags[j],
                                "Face IDs to apply slip BC",
                                NULL, bc->slips[j], &bc->num_slip[j], &flg); CHKERRQ(ierr);
    if (flg) bc->user_bc = PETSC_TRUE;
  }

  // Error if wall and slip BCs are set on the same face
  if (bc->user_bc)
    for (PetscInt c = 0; c < 3; c++)
      for (PetscInt s = 0; s < bc->num_slip[c]; s++)
        for (PetscInt w = 0; w < bc->num_wall; w++)
          if (bc->slips[c][s] == bc->walls[w])
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                    "Boundary condition already set on face %" PetscInt_FMT "!\n",
                    bc->walls[w]);

  // Inflow BCs
  bc->num_inflow = 16;
  ierr = PetscOptionsIntArray("-bc_inflow",
                              "Face IDs to apply inflow BC",
                              NULL, bc->inflows, &bc->num_inflow, NULL); CHKERRQ(ierr);
  // Outflow BCs
  bc->num_outflow = 16;
  ierr = PetscOptionsIntArray("-bc_outflow",
                              "Face IDs to apply outflow BC",
                              NULL, bc->outflows, &bc->num_outflow, NULL); CHKERRQ(ierr);

  PetscOptionsEnd();

  PetscFunctionReturn(0);
}
