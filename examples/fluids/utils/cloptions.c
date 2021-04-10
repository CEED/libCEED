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
/// Command line option processing for Navier-Stokes example using PETSc

#include "../navierstokes.h"

// -----------------------------------------------------------------------------
// Process command line options
// -----------------------------------------------------------------------------

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx) {

  PetscErrorCode ierr;
  PetscBool ceed_flag = PETSC_FALSE;
  PetscBool problem_flag = PETSC_FALSE;

  PetscFunctionBeginUser;

  ierr = PetscOptionsBegin(comm, NULL, "Navier-Stokes in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);

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

  app_ctx->degree = 2;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of finite elements",
                         NULL, app_ctx->degree, &app_ctx->degree, NULL); CHKERRQ(ierr);

  app_ctx->q_extra = 2;
  ierr = PetscOptionsInt("-q_extra", "Number of extra quadrature points",
                         NULL, app_ctx->q_extra, &app_ctx->q_extra, NULL); CHKERRQ(ierr);

  app_ctx->q_extra_sur = 2;
  ierr = PetscOptionsInt("-q_extra_boundary",
                         "Number of extra quadrature points on in/outflow faces",
                         NULL, app_ctx->q_extra_sur, &app_ctx->q_extra_sur, NULL); CHKERRQ(ierr);

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

  // todo
  //ierr = PetscOptionsEnum("-memtype",
  //                        "CEED MemType requested", NULL,
  //                        memTypes, (PetscEnum)app_ctx->memtyperequested,
  //                        (PetscEnum *)&app_ctx->memtyperequested, &app_ctx->setmemtyperequest);
  //CHKERRQ(ierr);

  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
