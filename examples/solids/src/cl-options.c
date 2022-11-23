// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Command line option processing for solid mechanics example using PETSc

#include "../include/cl-options.h"

// -----------------------------------------------------------------------------
// Process command line options
// -----------------------------------------------------------------------------
// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx) {
  PetscBool ceed_flag = PETSC_FALSE;

  PetscFunctionBeginUser;

  PetscOptionsBegin(comm, NULL, "Elasticity / Hyperelasticity in PETSc with libCEED", NULL);

  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, app_ctx->ceed_resource, app_ctx->ceed_resource,
                               sizeof(app_ctx->ceed_resource), &ceed_flag));

  PetscCall(PetscStrncpy(app_ctx->output_dir, ".", 2));  // Default - current directory
  PetscCall(PetscOptionsString("-output_dir", "Output directory", NULL, app_ctx->output_dir, app_ctx->output_dir, sizeof(app_ctx->output_dir), NULL));

  app_ctx->degree = 3;
  PetscCall(PetscOptionsInt("-degree", "Polynomial degree of tensor product basis", NULL, app_ctx->degree, &app_ctx->degree, NULL));

  app_ctx->q_extra = 0;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, app_ctx->q_extra, &app_ctx->q_extra, NULL));

  PetscCall(PetscOptionsString("-mesh", "Read mesh from file", NULL, app_ctx->mesh_file, app_ctx->mesh_file, sizeof(app_ctx->mesh_file), NULL));

  app_ctx->problem_choice = ELAS_LINEAR;  // Default - Linear Elasticity
  PetscCall(PetscOptionsEnum("-problem", "Solves Elasticity & Hyperelasticity Problems", NULL, problemTypes, (PetscEnum)app_ctx->problem_choice,
                             (PetscEnum *)&app_ctx->problem_choice, NULL));
  app_ctx->name          = problemTypes[app_ctx->problem_choice];
  app_ctx->name_for_disp = problemTypesForDisp[app_ctx->problem_choice];

  app_ctx->num_increments = app_ctx->problem_choice == ELAS_LINEAR ? 1 : 10;
  PetscCall(PetscOptionsInt("-num_steps", "Number of pseudo-time steps", NULL, app_ctx->num_increments, &app_ctx->num_increments, NULL));

  app_ctx->forcing_choice = FORCE_NONE;  // Default - no forcing term
  PetscCall(PetscOptionsEnum("-forcing", "Set forcing function option", NULL, forcing_types, (PetscEnum)app_ctx->forcing_choice,
                             (PetscEnum *)&app_ctx->forcing_choice, NULL));

  PetscInt max_n             = 3;
  app_ctx->forcing_vector[0] = 0;
  app_ctx->forcing_vector[1] = -1;
  app_ctx->forcing_vector[2] = 0;
  PetscCall(PetscOptionsScalarArray("-forcing_vec", "Direction to apply constant force", NULL, app_ctx->forcing_vector, &max_n, NULL));

  if ((app_ctx->problem_choice == ELAS_FSInitial_NH1 || app_ctx->problem_choice == ELAS_FSInitial_NH2 ||
       app_ctx->problem_choice == ELAS_FSCurrent_NH1 || app_ctx->problem_choice == ELAS_FSCurrent_NH2 ||
       app_ctx->problem_choice == ELAS_FSInitial_MR1) &&
      app_ctx->forcing_choice == FORCE_CONST)
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
            "Cannot use constant forcing and finite strain formulation. "
            "Constant forcing in reference frame currently unavaliable.");

  // Dirichlet boundary conditions
  app_ctx->bc_clamp_count = 16;
  PetscCall(
      PetscOptionsIntArray("-bc_clamp", "Face IDs to apply incremental Dirichlet BC", NULL, app_ctx->bc_clamp_faces, &app_ctx->bc_clamp_count, NULL));
  // Set vector for each clamped BC
  for (PetscInt i = 0; i < app_ctx->bc_clamp_count; i++) {
    // Translation vector
    char         option_name[25];
    const size_t nclamp_params = sizeof(app_ctx->bc_clamp_max[0]) / sizeof(app_ctx->bc_clamp_max[0][0]);
    for (PetscInt j = 0; j < nclamp_params; j++) app_ctx->bc_clamp_max[i][j] = 0.;

    snprintf(option_name, sizeof option_name, "-bc_clamp_%" PetscInt_FMT "_translate", app_ctx->bc_clamp_faces[i]);
    max_n = 3;
    PetscCall(PetscOptionsScalarArray(option_name, "Vector to translate clamped end by", NULL, app_ctx->bc_clamp_max[i], &max_n, NULL));

    // Rotation vector
    max_n = 5;
    snprintf(option_name, sizeof option_name, "-bc_clamp_%" PetscInt_FMT "_rotate", app_ctx->bc_clamp_faces[i]);
    PetscCall(PetscOptionsScalarArray(option_name, "Vector with axis of rotation and rotation, in radians", NULL, &app_ctx->bc_clamp_max[i][3],
                                      &max_n, NULL));

    // Normalize
    PetscScalar norm = sqrt(app_ctx->bc_clamp_max[i][3] * app_ctx->bc_clamp_max[i][3] + app_ctx->bc_clamp_max[i][4] * app_ctx->bc_clamp_max[i][4] +
                            app_ctx->bc_clamp_max[i][5] * app_ctx->bc_clamp_max[i][5]);
    if (fabs(norm) < 1e-16) norm = 1;
    for (PetscInt j = 0; j < 3; j++) app_ctx->bc_clamp_max[i][3 + j] /= norm;
  }

  // Neumann boundary conditions
  app_ctx->bc_traction_count = 16;
  PetscCall(PetscOptionsIntArray("-bc_traction", "Face IDs to apply traction (Neumann) BC", NULL, app_ctx->bc_traction_faces,
                                 &app_ctx->bc_traction_count, NULL));
  // Set vector for each traction BC
  for (PetscInt i = 0; i < app_ctx->bc_traction_count; i++) {
    // Translation vector
    char option_name[25];
    for (PetscInt j = 0; j < 3; j++) app_ctx->bc_traction_vector[i][j] = 0.;

    snprintf(option_name, sizeof option_name, "-bc_traction_%" PetscInt_FMT, app_ctx->bc_traction_faces[i]);
    max_n         = 3;
    PetscBool set = false;
    PetscCall(PetscOptionsScalarArray(option_name, "Traction vector for constrained face", NULL, app_ctx->bc_traction_vector[i], &max_n, &set));

    if (!set) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Traction vector must be set for all traction boundary conditions.");
  }

  app_ctx->multigrid_choice = MULTIGRID_LOGARITHMIC;
  PetscCall(PetscOptionsEnum("-multigrid", "Set multigrid type option", NULL, multigrid_types, (PetscEnum)app_ctx->multigrid_choice,
                             (PetscEnum *)&app_ctx->multigrid_choice, NULL));

  app_ctx->test_mode = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, app_ctx->test_mode, &(app_ctx->test_mode), NULL));

  app_ctx->expect_final_strain = -1.;
  PetscCall(PetscOptionsReal("-expect_final_strain_energy", "Expect final strain energy close to this value.", NULL, app_ctx->expect_final_strain,
                             &app_ctx->expect_final_strain, NULL));

  app_ctx->test_tol = 1e-8;
  PetscCall(PetscOptionsReal("-expect_final_state_rtol", "Relative tolerance for final strain energy test", NULL, app_ctx->test_tol,
                             &app_ctx->test_tol, NULL));

  app_ctx->view_soln = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-view_soln", "Write out solution vector for viewing", NULL, app_ctx->view_soln, &(app_ctx->view_soln), NULL));

  app_ctx->view_final_soln = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-view_final_soln", "Write out final solution vector for viewing", NULL, app_ctx->view_final_soln,
                             &(app_ctx->view_final_soln), NULL));

  PetscBool set;
  char      energy_viewer_filename[PETSC_MAX_PATH_LEN] = "";
  PetscCall(PetscOptionsString("-strain_energy_monitor", "Print out current strain energy at every load increment", NULL, energy_viewer_filename,
                               energy_viewer_filename, sizeof(energy_viewer_filename), &set));
  if (set) {
    PetscCall(PetscViewerASCIIOpen(comm, energy_viewer_filename, &app_ctx->energy_viewer));
    PetscCall(PetscViewerASCIIPrintf(app_ctx->energy_viewer, "increment,energy\n"));
    // Initial configuration is base energy state; this may not be true if we extend in the future to
    // initially loaded configurations (because a truly at-rest initial state may not be realizable).
    PetscCall(PetscViewerASCIIPrintf(app_ctx->energy_viewer, "%f,%e\n", 0., 0.));
  }
  PetscOptionsEnd();  // End of setting AppCtx

  // Check for all required values set
  if (app_ctx->test_mode) {
    if (app_ctx->forcing_choice == FORCE_NONE && !app_ctx->bc_clamp_count) app_ctx->forcing_choice = FORCE_MMS;
  }
  if (!app_ctx->bc_clamp_count && app_ctx->forcing_choice != FORCE_MMS) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-boundary options needed");
  }

  // Provide default ceed resource if not specified
  if (!ceed_flag) {
    const char *ceed_resource = "/cpu/self";
    strncpy(app_ctx->ceed_resource, ceed_resource, 10);
  }

  // Determine number of levels
  switch (app_ctx->multigrid_choice) {
    case MULTIGRID_LOGARITHMIC:
      app_ctx->num_levels = ceil(log(app_ctx->degree) / log(2)) + 1;
      break;
    case MULTIGRID_UNIFORM:
      app_ctx->num_levels = app_ctx->degree;
      break;
    case MULTIGRID_NONE:
      app_ctx->num_levels = 1;
      break;
  }

  // Populate array of degrees for each level for multigrid
  PetscCall(PetscMalloc1(app_ctx->num_levels, &(app_ctx->level_degrees)));

  switch (app_ctx->multigrid_choice) {
    case MULTIGRID_LOGARITHMIC:
      for (int i = 0; i < app_ctx->num_levels - 1; i++) app_ctx->level_degrees[i] = pow(2, i);
      app_ctx->level_degrees[app_ctx->num_levels - 1] = app_ctx->degree;
      break;
    case MULTIGRID_UNIFORM:
      for (int i = 0; i < app_ctx->num_levels; i++) app_ctx->level_degrees[i] = i + 1;
      break;
    case MULTIGRID_NONE:
      app_ctx->level_degrees[0] = app_ctx->degree;
      break;
  }

  PetscFunctionReturn(0);
};
