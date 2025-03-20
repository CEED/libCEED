// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Command line option processing for Navier-Stokes example using PETSc

#include <petscdevice.h>
#include <petscsys.h>

#include "../navierstokes.h"

// Register problems to be available on the command line
PetscErrorCode RegisterProblems_NS(AppCtx app_ctx) {
  app_ctx->problems = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "density_current", NS_DENSITY_CURRENT));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "euler_vortex", NS_EULER_VORTEX));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "shocktube", NS_SHOCKTUBE));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "advection", NS_ADVECTION));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "blasius", NS_BLASIUS));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "channel", NS_CHANNEL));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "gaussian_wave", NS_GAUSSIAN_WAVE));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "newtonian", NS_NEWTONIAN_IG));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "taylor_green", NS_TAYLOR_GREEN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx, SimpleBC bc) {
  PetscBool ceed_flag    = PETSC_FALSE;
  PetscBool problem_flag = PETSC_FALSE;
  PetscBool option_set   = PETSC_FALSE;

  PetscFunctionBeginUser;
  {
    PetscInt num_options;

    PetscCall(PetscOptionsLeftGet(NULL, &num_options, NULL, NULL));
    PetscCheck(num_options > 0, comm, PETSC_ERR_USER_INPUT,
               "Command line options required."
               " Please consult the documentation to see which options are required.");
    PetscCall(PetscOptionsLeftRestore(NULL, &num_options, NULL, NULL));
  }

  PetscOptionsBegin(comm, NULL, "Navier-Stokes in PETSc with libCEED", NULL);

  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, app_ctx->ceed_resource, app_ctx->ceed_resource,
                               sizeof(app_ctx->ceed_resource), &ceed_flag));

  app_ctx->test_type = TESTTYPE_NONE;
  PetscCall(PetscOptionsEnum("-test_type", "Type of test to run", NULL, TestTypes, (PetscEnum)(app_ctx->test_type), (PetscEnum *)&app_ctx->test_type,
                             NULL));

  app_ctx->test_tol = 1E-11;
  PetscCall(PetscOptionsScalar("-compare_final_state_atol", "Test absolute tolerance", NULL, app_ctx->test_tol, &app_ctx->test_tol, NULL));

  PetscCall(PetscOptionsString("-compare_final_state_filename", "Test filename", NULL, app_ctx->test_file_path, app_ctx->test_file_path,
                               sizeof(app_ctx->test_file_path), NULL));

  PetscCall(PetscOptionsFList("-problem", "Problem to solve", NULL, app_ctx->problems, app_ctx->problem_name, app_ctx->problem_name,
                              sizeof(app_ctx->problem_name), &problem_flag));

  app_ctx->viz_refine = 0;
  PetscCall(PetscOptionsInt("-viz_refine", "Regular refinement levels for visualization", NULL, app_ctx->viz_refine, &app_ctx->viz_refine, NULL));

  app_ctx->checkpoint_interval = 10;
  app_ctx->checkpoint_vtk      = PETSC_FALSE;
  PetscCall(PetscOptionsDeprecated("-output_freq", "-checkpoint_interval", "libCEED 0.11.1", "Use -checkpoint_vtk true to include VTK output"));
  PetscCall(PetscOptionsInt("-output_freq", "Frequency of output, in number of steps", NULL, app_ctx->checkpoint_interval,
                            &app_ctx->checkpoint_interval, &option_set));
  if (option_set) app_ctx->checkpoint_vtk = PETSC_TRUE;
  PetscCall(PetscOptionsInt("-checkpoint_interval", "Frequency of output, in number of steps", NULL, app_ctx->checkpoint_interval,
                            &app_ctx->checkpoint_interval, NULL));
  PetscCall(PetscOptionsBool("-checkpoint_vtk", "Include VTK (*.vtu) output at each binary checkpoint", NULL, app_ctx->checkpoint_vtk,
                             &app_ctx->checkpoint_vtk, NULL));

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
  if (option_set && app_ctx->cont_steps == 0) app_ctx->cont_steps = -1;  // Read time from file

  PetscCall(PetscStrcpy(app_ctx->cont_time_file, "[output_dir]/ns-time.bin"));
  PetscCall(PetscOptionsString("-continue_time_filename", "Filename to get initial condition time from", NULL, app_ctx->cont_time_file,
                               app_ctx->cont_time_file, sizeof(app_ctx->cont_time_file), &option_set));
  if (!option_set) PetscCall(PetscSNPrintf(app_ctx->cont_time_file, sizeof app_ctx->cont_time_file, "%s/ns-time.bin", app_ctx->output_dir));

  app_ctx->degree = 1;
  PetscCall(PetscOptionsInt("-degree", "Polynomial degree of finite elements", NULL, app_ctx->degree, &app_ctx->degree, NULL));

  app_ctx->q_extra = 0;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, app_ctx->q_extra, &app_ctx->q_extra, NULL));

  {
    PetscBool option_set;
    char      amat_type[256] = "";
    PetscCall(PetscOptionsFList("-amat_type", "Set the type of Amat distinct from Pmat (-dm_mat_type)", NULL, MatList, amat_type, amat_type,
                                sizeof(amat_type), &option_set));
    if (option_set) PetscCall(PetscStrallocpy(amat_type, (char **)&app_ctx->amat_type));
  }
  {
    PetscBool option_set;
    PetscCall(PetscOptionsHasName(NULL, NULL, "-pmat_pbdiagonal", &option_set));
    if (option_set) PetscCall(PetscPrintf(comm, "Warning! -pmat_pbdiagonal no longer used. Pmat assembly determined from -pc_type setting\n"));
  }

  // Provide default ceed resource if not specified
  if (!ceed_flag) {
    const char *ceed_resource = "/cpu/self";
    strncpy(app_ctx->ceed_resource, ceed_resource, 10);
  }
  // If we request a GPU, make sure PETSc has initialized its device (which is
  // MPI-aware in case multiple devices are available) before CeedInit so that
  // PETSc and libCEED agree about which device to use.
  if (strncmp(app_ctx->ceed_resource, "/gpu", 4) == 0) PetscCall(PetscDeviceInitialize(PETSC_DEVICE_DEFAULT()));

  // Provide default problem if not specified
  if (!problem_flag) {
    const char *problem_name = "density_current";
    strncpy(app_ctx->problem_name, problem_name, 16);
  }

  // Statistics Options
  app_ctx->turb_spanstats_collect_interval = 1;
  PetscCall(PetscOptionsInt("-ts_monitor_turbulence_spanstats_collect_interval", "Number of timesteps between statistics collection", NULL,
                            app_ctx->turb_spanstats_collect_interval, &app_ctx->turb_spanstats_collect_interval, NULL));

  app_ctx->turb_spanstats_viewer_interval = -1;
  PetscCall(PetscOptionsInt("-ts_monitor_turbulence_spanstats_viewer_interval", "Number of timesteps between statistics viewer writing", NULL,
                            app_ctx->turb_spanstats_viewer_interval, &app_ctx->turb_spanstats_viewer_interval, NULL));

  PetscCall(PetscOptionsViewer("-ts_monitor_turbulence_spanstats_viewer", "Viewer for the statistics", NULL, &app_ctx->turb_spanstats_viewer,
                               &app_ctx->turb_spanstats_viewer_format, &app_ctx->turb_spanstats_enable));

  PetscCall(PetscOptionsViewer("-ts_monitor_wall_force", "Viewer for force on each (no-slip) wall", NULL, &app_ctx->wall_forces.viewer,
                               &app_ctx->wall_forces.viewer_format, NULL));

  PetscCall(PetscOptionsBool("-diff_filter_monitor", "Enable differential filtering TSMonitor", NULL, app_ctx->diff_filter_monitor,
                             &app_ctx->diff_filter_monitor, NULL));

  // Mesh Transformation Options
  app_ctx->mesh_transform_type = MESH_TRANSFORM_NONE;
  PetscCall(PetscOptionsEnum("-mesh_transform", "Mesh transform to perform", NULL, MeshTransformTypes, (PetscEnum)app_ctx->mesh_transform_type,
                             (PetscEnum *)&app_ctx->mesh_transform_type, NULL));

  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}
