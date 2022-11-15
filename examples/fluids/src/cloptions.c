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
  PetscFunctionBeginUser;

  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "density_current", NS_DENSITY_CURRENT));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "euler_vortex", NS_EULER_VORTEX));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "shocktube", NS_SHOCKTUBE));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "advection", NS_ADVECTION));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "advection2d", NS_ADVECTION2D));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "blasius", NS_BLASIUS));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "channel", NS_CHANNEL));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "newtonian_wave", NS_NEWTONIAN_WAVE));

  PetscFunctionReturn(0);
}

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx, SimpleBC bc) {
  PetscBool ceed_flag    = PETSC_FALSE;
  PetscBool problem_flag = PETSC_FALSE;
  PetscBool option_set   = PETSC_FALSE;

  PetscFunctionBeginUser;

  PetscOptionsBegin(comm, NULL, "Navier-Stokes in PETSc with libCEED", NULL);

  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, app_ctx->ceed_resource, app_ctx->ceed_resource,
                               sizeof(app_ctx->ceed_resource), &ceed_flag));

  app_ctx->test_mode = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-test", "Run in test mode", NULL, app_ctx->test_mode, &app_ctx->test_mode, NULL));

  app_ctx->test_tol = 1E-11;
  PetscCall(PetscOptionsScalar("-compare_final_state_atol", "Test absolute tolerance", NULL, app_ctx->test_tol, &app_ctx->test_tol, NULL));

  PetscCall(PetscOptionsString("-compare_final_state_filename", "Test filename", NULL, app_ctx->file_path, app_ctx->file_path,
                               sizeof(app_ctx->file_path), NULL));

  PetscCall(PetscOptionsFList("-problem", "Problem to solve", NULL, app_ctx->problems, app_ctx->problem_name, app_ctx->problem_name,
                              sizeof(app_ctx->problem_name), &problem_flag));

  app_ctx->viz_refine = 0;
  PetscCall(PetscOptionsInt("-viz_refine", "Regular refinement levels for visualization", NULL, app_ctx->viz_refine, &app_ctx->viz_refine, NULL));

  app_ctx->output_freq = 10;
  PetscCall(PetscOptionsInt("-output_freq", "Frequency of output, in number of steps", NULL, app_ctx->output_freq, &app_ctx->output_freq, NULL));

  PetscCall(PetscOptionsBool("-output_add_stepnum2bin", "Add step number to the binary outputs", NULL, app_ctx->add_stepnum2bin,
                             &app_ctx->add_stepnum2bin, NULL));

  PetscCall(PetscStrncpy(app_ctx->output_dir, ".", 2));
  PetscCall(PetscOptionsString("-output_dir", "Output directory", NULL, app_ctx->output_dir, app_ctx->output_dir, sizeof(app_ctx->output_dir), NULL));

  app_ctx->cont_steps = 0;
  PetscCall(PetscOptionsInt("-continue", "Continue from previous solution", NULL, app_ctx->cont_steps, &app_ctx->cont_steps, NULL));

  PetscCall(PetscStrcpy(app_ctx->cont_file, "[output_dir]/ns-solution.bin"));
  PetscCall(PetscOptionsString("-continue_filename", "Filename to get initial condition from", NULL, app_ctx->cont_file, app_ctx->cont_file,
                               sizeof(app_ctx->cont_file), &option_set));
  if (!option_set) PetscCall(PetscSNPrintf(app_ctx->cont_file, sizeof app_ctx->cont_file, "%s/ns-solution.bin", app_ctx->output_dir));

  PetscCall(PetscStrcpy(app_ctx->cont_time_file, "[output_dir]/ns-time.bin"));
  PetscCall(PetscOptionsString("-continue_time_filename", "Filename to get initial condition time from", NULL, app_ctx->cont_time_file,
                               app_ctx->cont_time_file, sizeof(app_ctx->cont_time_file), &option_set));
  if (!option_set) PetscCall(PetscSNPrintf(app_ctx->cont_time_file, sizeof app_ctx->cont_time_file, "%s/ns-time.bin", app_ctx->output_dir));

  app_ctx->degree = 1;
  PetscCall(PetscOptionsInt("-degree", "Polynomial degree of finite elements", NULL, app_ctx->degree, &app_ctx->degree, NULL));

  app_ctx->q_extra = 2;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, app_ctx->q_extra, &app_ctx->q_extra, NULL));

  {
    PetscBool option_set;
    char      amat_type[256] = "";
    PetscCall(PetscOptionsFList("-amat_type", "Set the type of Amat distinct from Pmat (-dm_mat_type)", NULL, MatList, amat_type, amat_type,
                                sizeof(amat_type), &option_set));
    if (option_set) PetscCall(PetscStrallocpy(amat_type, (char **)&app_ctx->amat_type));
  }
  PetscCall(PetscOptionsBool("-pmat_pbdiagonal", "Assemble only point-block diagonal for Pmat", NULL, app_ctx->pmat_pbdiagonal,
                             &app_ctx->pmat_pbdiagonal, NULL));

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
  PetscCall(PetscOptionsIntArray("-bc_wall", "Face IDs to apply wall BC", NULL, bc->walls, &bc->num_wall, NULL));
  bc->num_comps = 5;
  PetscCall(PetscOptionsIntArray("-wall_comps", "An array of constrained component numbers", NULL, bc->wall_comps, &bc->num_comps, &flg));
  // Slip Boundary Conditions
  for (PetscInt j = 0; j < 3; j++) {
    bc->num_slip[j] = 16;
    PetscBool   flg;
    const char *flags[3] = {"-bc_slip_x", "-bc_slip_y", "-bc_slip_z"};
    PetscCall(PetscOptionsIntArray(flags[j], "Face IDs to apply slip BC", NULL, bc->slips[j], &bc->num_slip[j], &flg));
    if (flg) bc->user_bc = PETSC_TRUE;
  }

  // Error if wall and slip BCs are set on the same face
  if (bc->user_bc)
    for (PetscInt c = 0; c < 3; c++)
      for (PetscInt s = 0; s < bc->num_slip[c]; s++)
        for (PetscInt w = 0; w < bc->num_wall; w++)
          if (bc->slips[c][s] == bc->walls[w])
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Boundary condition already set on face %" PetscInt_FMT "!\n", bc->walls[w]);

  // Inflow BCs
  bc->num_inflow = 16;
  PetscCall(PetscOptionsIntArray("-bc_inflow", "Face IDs to apply inflow BC", NULL, bc->inflows, &bc->num_inflow, NULL));
  // Outflow BCs
  bc->num_outflow = 16;
  PetscCall(PetscOptionsIntArray("-bc_outflow", "Face IDs to apply outflow BC", NULL, bc->outflows, &bc->num_outflow, NULL));
  // Freestream BCs
  bc->num_freestream = 16;
  PetscCall(PetscOptionsIntArray("-bc_freestream", "Face IDs to apply freestream BC", NULL, bc->freestreams, &bc->num_freestream, NULL));

  PetscOptionsEnd();

  PetscFunctionReturn(0);
}
