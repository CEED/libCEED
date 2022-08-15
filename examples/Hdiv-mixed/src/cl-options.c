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
/// Command line option processing for H(div) example using PETSc

#include "../include/cl-options.h"

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(AppCtx app_ctx) {

  PetscBool problem_flag = PETSC_FALSE;
  PetscBool ceed_flag = PETSC_FALSE;
  PetscFunctionBeginUser;

  PetscOptionsBegin(app_ctx->comm, NULL, "H(div) examples in PETSc with libCEED",
                    NULL);

  PetscCall( PetscOptionsString("-ceed", "CEED resource specifier",
                                NULL, app_ctx->ceed_resource, app_ctx->ceed_resource,
                                sizeof(app_ctx->ceed_resource), &ceed_flag) );

  // Provide default ceed resource if not specified
  if (!ceed_flag) {
    const char *ceed_resource = "/cpu/self";
    strncpy(app_ctx->ceed_resource, ceed_resource, 10);
  }

  PetscCall( PetscOptionsFList("-problem", "Problem to solve", NULL,
                               app_ctx->problems,
                               app_ctx->problem_name, app_ctx->problem_name, sizeof(app_ctx->problem_name),
                               &problem_flag) );
  // Provide default problem if not specified
  if (!problem_flag) {
    const char *problem_name = "darcy2d";
    strncpy(app_ctx->problem_name, problem_name, 16);
  }
  app_ctx->degree = 1;
  PetscCall( PetscOptionsInt("-degree", "Polynomial degree of finite elements",
                             NULL, app_ctx->degree, &app_ctx->degree, NULL) );

  app_ctx->q_extra = 0;
  PetscCall( PetscOptionsInt("-q_extra", "Number of extra quadrature points",
                             NULL, app_ctx->q_extra, &app_ctx->q_extra, NULL) );
  app_ctx->view_solution = PETSC_FALSE;
  PetscCall( PetscOptionsBool("-view_solution",
                              "View solution in Paraview",
                              NULL, app_ctx->view_solution,
                              &(app_ctx->view_solution), NULL) );
  app_ctx->quartic = PETSC_FALSE;
  PetscCall( PetscOptionsBool("-quartic",
                              "To test PetscViewer",
                              NULL, app_ctx->quartic,
                              &(app_ctx->quartic), NULL) );

  PetscCall( PetscStrncpy(app_ctx->output_dir, ".", 2) );
  PetscCall( PetscOptionsString("-output_dir", "Output directory",
                                NULL, app_ctx->output_dir, app_ctx->output_dir,
                                sizeof(app_ctx->output_dir), NULL) );

  app_ctx->output_freq = 10;
  PetscCall( PetscOptionsInt("-output_freq",
                             "Frequency of output, in number of steps",
                             NULL, app_ctx->output_freq, &app_ctx->output_freq, NULL) );
  app_ctx->bc_pressure_count = 16;
  // we can set one face by: -bc_faces 1 OR multiple faces by :-bc_faces 1,2,3
  PetscCall( PetscOptionsIntArray("-bc_faces",
                                  "Face IDs to apply pressure BC",
                                  NULL, app_ctx->bc_faces, &app_ctx->bc_pressure_count, NULL) );


  PetscOptionsEnd();

  PetscFunctionReturn(0);
}
