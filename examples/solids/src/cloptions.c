// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Command line option processing for solid mechanics example using PETSc

#include "../elasticity.h"

// -----------------------------------------------------------------------------
// Process command line options
// -----------------------------------------------------------------------------
// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx) {
  PetscErrorCode ierr;
  PetscBool ceed_flag = PETSC_FALSE;

  PetscFunctionBeginUser;

  ierr = PetscOptionsBegin(comm, NULL,
                           "Elasticity / Hyperelasticity in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);

  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, app_ctx->ceed_resource, app_ctx->ceed_resource,
                            sizeof(app_ctx->ceed_resource), &ceed_flag);
  CHKERRQ(ierr);

  ierr = PetscStrncpy(app_ctx->output_dir, ".", 2);
  CHKERRQ(ierr); // Default - current directory
  ierr = PetscOptionsString("-output_dir", "Output directory",
                            NULL, app_ctx->output_dir, app_ctx->output_dir,
                            sizeof(app_ctx->output_dir), NULL); CHKERRQ(ierr);

  app_ctx->degree         = 3;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, app_ctx->degree, &app_ctx->degree, NULL);
  CHKERRQ(ierr);

  app_ctx->q_extra         = 0;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, app_ctx->q_extra, &app_ctx->q_extra, NULL);
  CHKERRQ(ierr);

  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            app_ctx->mesh_file, app_ctx->mesh_file,
                            sizeof(app_ctx->mesh_file), NULL); CHKERRQ(ierr);

  app_ctx->problem_choice  = ELAS_LINEAR;       // Default - Linear Elasticity
  ierr = PetscOptionsEnum("-problem",
                          "Solves Elasticity & Hyperelasticity Problems",
                          NULL, problemTypes, (PetscEnum)app_ctx->problem_choice,
                          (PetscEnum *)&app_ctx->problem_choice, NULL);
  CHKERRQ(ierr);

  app_ctx->num_increments = app_ctx->problem_choice == ELAS_LINEAR ? 1 : 10;
  ierr = PetscOptionsInt("-num_steps", "Number of pseudo-time steps",
                         NULL, app_ctx->num_increments, &app_ctx->num_increments,
                         NULL); CHKERRQ(ierr);

  app_ctx->forcing_choice  = FORCE_NONE;     // Default - no forcing term
  ierr = PetscOptionsEnum("-forcing", "Set forcing function option", NULL,
                          forcing_types, (PetscEnum)app_ctx->forcing_choice,
                          (PetscEnum *)&app_ctx->forcing_choice, NULL);
  CHKERRQ(ierr);

  PetscInt max_n = 3;
  app_ctx->forcing_vector[0] = 0;
  app_ctx->forcing_vector[1] = -1;
  app_ctx->forcing_vector[2] = 0;
  ierr = PetscOptionsScalarArray("-forcing_vec",
                                 "Direction to apply constant force", NULL,
                                 app_ctx->forcing_vector, &max_n, NULL);
  CHKERRQ(ierr);

  if ((app_ctx->problem_choice == ELAS_FSInitial_NH1 ||
       app_ctx->problem_choice == ELAS_FSInitial_NH2 ||
       app_ctx->problem_choice == ELAS_FSCurrent_NH1 ||
       app_ctx->problem_choice == ELAS_FSCurrent_NH2 ||
       app_ctx->problem_choice == ELAS_FSInitial_MR1) &&
       app_ctx->forcing_choice == FORCE_CONST)
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
            "Cannot use constant forcing and finite strain formulation. "
            "Constant forcing in reference frame currently unavaliable.");

  // Dirichlet boundary conditions
  app_ctx->bc_clamp_count = 16;
  ierr = PetscOptionsIntArray("-bc_clamp",
                              "Face IDs to apply incremental Dirichlet BC",
                              NULL, app_ctx->bc_clamp_faces, &app_ctx->bc_clamp_count,
                              NULL); CHKERRQ(ierr);
  // Set vector for each clamped BC
  for (PetscInt i = 0; i < app_ctx->bc_clamp_count; i++) {
    // Translation vector
    char option_name[25];
    const size_t nclamp_params = sizeof(app_ctx->bc_clamp_max[0])/sizeof(
                                   app_ctx->bc_clamp_max[0][0]);
    for (PetscInt j = 0; j < nclamp_params; j++)
      app_ctx->bc_clamp_max[i][j] = 0.;

    snprintf(option_name, sizeof option_name, "-bc_clamp_%d_translate",
             app_ctx->bc_clamp_faces[i]);
    max_n = 3;
    ierr = PetscOptionsScalarArray(option_name,
                                   "Vector to translate clamped end by", NULL,
                                   app_ctx->bc_clamp_max[i], &max_n, NULL);
    CHKERRQ(ierr);

    // Rotation vector
    max_n = 5;
    snprintf(option_name, sizeof option_name, "-bc_clamp_%d_rotate",
             app_ctx->bc_clamp_faces[i]);
    ierr = PetscOptionsScalarArray(option_name,
                                   "Vector with axis of rotation and rotation, in radians",
                                   NULL, &app_ctx->bc_clamp_max[i][3], &max_n, NULL);
    CHKERRQ(ierr);

    // Normalize
    PetscScalar norm = sqrt(app_ctx->bc_clamp_max[i][3]*app_ctx->bc_clamp_max[i][3]
                            + app_ctx->bc_clamp_max[i][4]*app_ctx->bc_clamp_max[i][4]
                            + app_ctx->bc_clamp_max[i][5]*app_ctx->bc_clamp_max[i][5]);
    if (fabs(norm) < 1e-16)
      norm = 1;
    for (PetscInt j = 0; j < 3; j++)
      app_ctx->bc_clamp_max[i][3 + j] /= norm;
  }

  // Neumann boundary conditions
  app_ctx->bc_traction_count = 16;
  ierr = PetscOptionsIntArray("-bc_traction",
                              "Face IDs to apply traction (Neumann) BC",
                              NULL, app_ctx->bc_traction_faces,
                              &app_ctx->bc_traction_count, NULL); CHKERRQ(ierr);
  // Set vector for each traction BC
  for (PetscInt i = 0; i < app_ctx->bc_traction_count; i++) {
    // Translation vector
    char option_name[25];
    for (PetscInt j = 0; j < 3; j++)
      app_ctx->bc_traction_vector[i][j] = 0.;

    snprintf(option_name, sizeof option_name, "-bc_traction_%d",
             app_ctx->bc_traction_faces[i]);
    max_n = 3;
    PetscBool set = false;
    ierr = PetscOptionsScalarArray(option_name,
                                   "Traction vector for constrained face", NULL,
                                   app_ctx->bc_traction_vector[i], &max_n, &set);
    CHKERRQ(ierr);

    if (!set)
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,
              "Traction vector must be set for all traction boundary conditions.");
  }

  app_ctx->multigrid_choice = MULTIGRID_LOGARITHMIC;
  ierr = PetscOptionsEnum("-multigrid", "Set multigrid type option", NULL,
                          multigrid_types, (PetscEnum)app_ctx->multigrid_choice,
                          (PetscEnum *)&app_ctx->multigrid_choice, NULL);
  CHKERRQ(ierr);

  app_ctx->nu_smoother = 0.;
  ierr = PetscOptionsScalar("-nu_smoother", "Poisson's ratio for smoother",
                            NULL, app_ctx->nu_smoother, &app_ctx->nu_smoother, NULL);
  CHKERRQ(ierr);

  app_ctx->test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, app_ctx->test_mode, &(app_ctx->test_mode), NULL);
  CHKERRQ(ierr);

  app_ctx->view_soln = PETSC_FALSE;
  ierr = PetscOptionsBool("-view_soln", "Write out solution vector for viewing",
                          NULL, app_ctx->view_soln, &(app_ctx->view_soln), NULL);
  CHKERRQ(ierr);

  app_ctx->view_final_soln = PETSC_FALSE;
  ierr = PetscOptionsBool("-view_final_soln",
                          "Write out final solution vector for viewing",
                          NULL, app_ctx->view_final_soln, &(app_ctx->view_final_soln),
                          NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr); // End of setting AppCtx

  // Check for all required values set
  if (!app_ctx->test_mode) {
    if (!app_ctx->bc_clamp_count && (app_ctx->forcing_choice != FORCE_MMS)) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-boundary options needed");
    }
  } else {
    app_ctx->forcing_choice = FORCE_MMS;
  }

  // Provide default ceed resource if not specified
  if (!ceed_flag) {
    const char *ceed_resource = "/cpu/self";
    strncpy(app_ctx->ceed_resource, ceed_resource, 10);
  }

  // Determine number of levels
  switch (app_ctx->multigrid_choice) {
  case MULTIGRID_LOGARITHMIC:
    app_ctx->num_levels = ceil(log(app_ctx->degree)/log(2)) + 1;
    break;
  case MULTIGRID_UNIFORM:
    app_ctx->num_levels = app_ctx->degree;
    break;
  case MULTIGRID_NONE:
    app_ctx->num_levels = 1;
    break;
  }

  // Populate array of degrees for each level for multigrid
  ierr = PetscMalloc1(app_ctx->num_levels, &(app_ctx->level_degrees));
  CHKERRQ(ierr);

  switch (app_ctx->multigrid_choice) {
  case MULTIGRID_LOGARITHMIC:
    for (int i = 0; i < app_ctx->num_levels-1; i++)
      app_ctx->level_degrees[i] = pow(2,i);
    app_ctx->level_degrees[app_ctx->num_levels-1] = app_ctx->degree;
    break;
  case MULTIGRID_UNIFORM:
    for (int i = 0; i < app_ctx->num_levels; i++)
      app_ctx->level_degrees[i] = i + 1;
    break;
  case MULTIGRID_NONE:
    app_ctx->level_degrees[0] = app_ctx->degree;
    break;
  }

  PetscFunctionReturn(0);
};

// Process physics options
PetscErrorCode ProcessPhysics(MPI_Comm comm, Physics phys, Units units) {
  PetscErrorCode ierr;
  PetscBool nu_flag = PETSC_FALSE;
  PetscBool Young_flag = PETSC_FALSE;
  phys->nu = 0;
  phys->E = 0;
  units->meter     = 1;        // 1 meter in scaled length units
  units->second    = 1;        // 1 second in scaled time units
  units->kilogram  = 1;        // 1 kilogram in scaled mass units

  PetscFunctionBeginUser;

  ierr = PetscOptionsBegin(comm, NULL,
                           "Elasticity / Hyperelasticity in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-nu", "Poisson's ratio", NULL, phys->nu, &phys->nu,
                            &nu_flag); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-E", "Young's Modulus", NULL, phys->E, &phys->E,
                            &Young_flag); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, units->meter, &units->meter, NULL);
  CHKERRQ(ierr);
  units->meter = fabs(units->meter);

  ierr = PetscOptionsScalar("-units_second", "1 second in scaled time units",
                            NULL, units->second, &units->second, NULL);
  CHKERRQ(ierr);
  units->second = fabs(units->second);

  ierr = PetscOptionsScalar("-units_kilogram", "1 kilogram in scaled mass units",
                            NULL, units->kilogram, &units->kilogram, NULL);
  CHKERRQ(ierr);
  units->kilogram = fabs(units->kilogram);

  ierr = PetscOptionsEnd(); CHKERRQ(ierr); // End of setting Physics

  // Check for all required options to be set
  if (!nu_flag) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-nu option needed");
  }
  if (!Young_flag) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-E option needed");
  }

  // Define derived units
  units->Pascal = units->kilogram / (units->meter * PetscSqr(units->second));

  // Scale E to Pa
  phys->E *= units->Pascal;

  PetscFunctionReturn(0);
};

// Process physics options - Mooney-Rivlin
PetscErrorCode ProcessPhysics_MR(MPI_Comm comm, Physics_MR phys_MR, Units units) { 
  PetscErrorCode ierr;
  PetscBool mu_1_Flag = PETSC_FALSE;
  PetscBool mu_2_Flag = PETSC_FALSE;
  PetscBool k_1_Flag = PETSC_FALSE;
  phys_MR->mu_1 = 0;
  phys_MR->mu_2 = 0;
  phys_MR->k_1 = 0;
  units->meter     = 1;        // 1 meter in scaled length units
  units->second    = 1;        // 1 second in scaled time units
  units->kilogram  = 1;        // 1 kilogram in scaled mass units

  PetscFunctionBeginUser;

  ierr = PetscOptionsBegin(comm, NULL,
                           "Elasticity / Hyperelasticity in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-mu_1", "Material Property mu_1", NULL, phys_MR->mu_1, &phys_MR->mu_1,
                            &mu_1_Flag); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-mu_2", "Material Property mu_2", NULL, phys_MR->mu_2, &phys_MR->mu_2,
                            &mu_2_Flag); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-K", "Material Property K_1", NULL, phys_MR->k_1, &phys_MR->k_1,
                            &k_1_Flag); CHKERRQ(ierr);

  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, units->meter, &units->meter, NULL);
  CHKERRQ(ierr);
  units->meter = fabs(units->meter);

  ierr = PetscOptionsScalar("-units_second", "1 second in scaled time units",
                            NULL, units->second, &units->second, NULL);
  CHKERRQ(ierr);
  units->second = fabs(units->second);

  ierr = PetscOptionsScalar("-units_kilogram", "1 kilogram in scaled mass units",
                            NULL, units->kilogram, &units->kilogram, NULL);
  CHKERRQ(ierr);
  units->kilogram = fabs(units->kilogram);

  ierr = PetscOptionsEnd(); CHKERRQ(ierr); // End of setting Physics

  // Check for all required options to be set
  if (!mu_1_Flag) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-mu_1 option needed");
  }
  if (!mu_2_Flag) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-mu_2 option needed");
  }
  if (!k_1_Flag) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "-K option needed");
  }

  // Define derived units
  units->Pascal = units->kilogram / (units->meter * PetscSqr(units->second));

  // Scale E to Pa
  phys_MR->k_1 *= units->Pascal;

  PetscFunctionReturn(0);
};

PetscErrorCode ProcessPhysics_General(MPI_Comm comm, AppCtx app_ctx, Physics phys, Physics_MR phys_MR, Units units){

  if(app_ctx -> problem_choice != ELAS_FSInitial_MR1)
    {
      PetscErrorCode ProcessPhysics(MPI_Comm comm, Physics phys, Units units);
    }
  else
    {
      PetscErrorCode ProcessPhysics_MR(MPI_Comm comm, Physics_MR phys_MR, Units units);
    }

    PetscFunctionReturn(0);
};
