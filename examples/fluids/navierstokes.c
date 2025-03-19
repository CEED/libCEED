// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                        libCEED + PETSc Example: Navier-Stokes
//
// This example demonstrates a simple usage of libCEED with PETSc to solve a Navier-Stokes problem.
//
// Build with:
//
//     make [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>] navierstokes
//
// Sample runs:
//
//     ./navierstokes -ceed /cpu/self -options_file gaussianwave.yml
//     ./navierstokes -ceed /gpu/cuda -problem advection -degree 1
//
//TESTARGS(name="Newtonian and Riemann Solver Unit Tests",only="cpu") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/gaussianwave.yaml -compare_final_state_atol 1e100 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-gaussianwave-IDL-entropy.bin -dm_plex_box_faces 5,5,1 -ts_max_steps 0 -newtonian_unit_tests -riemann_solver_unit_tests
//TESTARGS(name="Gaussian Wave, IDL and Entropy variables") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/gaussianwave.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-gaussianwave-IDL-entropy.bin -state_var entropy -dm_plex_box_faces 5,5,1 -ts_max_steps 5 -idl_decay_time 2e-3 -idl_length 0.25 -idl_start 0 -idl_pressure 70
//TESTARGS(name="Gaussian Wave, explicit, supg, IDL") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/gaussianwave.yaml -compare_final_state_atol 1e-8 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-gaussianwave-explicit.bin -dm_plex_box_faces 2,2,1 -ts_max_steps 5 -degree 3 -implicit false -ts_type rk -stab supg -state_var conservative -mass_ksp_type gmres -mass_pc_jacobi_type diagonal -idl_decay_time 2e-3 -idl_length 0.25 -idl_start 0 -idl_pressure 70
//TESTARGS(name="Advection 2D, rotation, explicit, supg, consistent mass") -ceed {ceed_resource} -test_type solver -problem advection -degree 3 -dm_plex_box_faces 2,2 -dm_plex_box_lower 0,0 -dm_plex_box_upper 125,125 -bc_wall 1,2,3,4 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ts_dt 1e-3 -ts_max_steps 10 -stab supg -Ctaus 0.5 -mass_ksp_type gmres -mass_pc_type vpbjacobi -compare_final_state_atol 1e-10 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-rotation-explicit-stab-supg-consistent-mass.bin
//TESTARGS(name="Advection, skew") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/advection.yaml -ts_max_steps 5 -wind_type translation -wind_translation -0.5547002,0.83205029,0 -advection_ic_type skew  -dm_plex_box_faces 2,1,1 -degree 2 -stab supg -stab_tau advdiff_shakib -Ctau_a 4 -ksp_type gmres -diffusion_coeff 5e-4 -compare_final_state_atol 7e-10 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-skew.bin
//TESTARGS(name="Blasius, bc_slip") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/blasius.yaml -ts_max_steps 5 -dm_plex_box_faces 3,20,1 -platemesh_nDelta 10 -platemesh_growth 1.2 -bc_outflow 5 -bc_slip 4 -compare_final_state_atol 2E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius-bc_slip.bin
//TESTARGS(name="Advection, rotation, cosine") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/advection.yaml -ts_max_steps 0 -advection_ic_type cosine_hill -dm_plex_box_faces 2,1,1 -compare_final_state_atol 1e-10 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-rotation-cosine.bin
//TESTARGS(name="Gaussian Wave, using MatShell") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/gaussianwave.yaml -compare_final_state_atol 1e-8 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-gaussianwave-shell.bin -dm_plex_box_faces 2,2,1 -ts_max_steps 5 -degree 3 -amat_type shell -pc_type vpbjacobi -ts_alpha_radius 0.5
//TESTARGS(name="Taylor-Green Vortex IC") -ceed {ceed_resource} -problem taylor_green -test_type solver -dm_plex_dim 3 -dm_plex_box_faces 6,6,6 -ts_max_steps 0 -compare_final_state_atol 1e-12 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-taylor-green-IC.bin
//TESTARGS(name="Blasius, Anisotropic Differential Filter") -ceed {ceed_resource} -test_type diff_filter -options_file examples/fluids/tests-output/blasius_test.yaml -compare_final_state_atol 5e-10 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius_diff_filter_aniso_vandriest.bin -diff_filter_monitor -ts_max_steps 0 -state_var primitive -diff_filter_friction_length 1e-5 -diff_filter_wall_damping_function van_driest -diff_filter_ksp_rtol 1e-8 -diff_filter_grid_based_width -diff_filter_width_scaling 1,0.7,1
//TESTARGS(name="Blasius, Isotropic Differential Filter") -ceed {ceed_resource} -test_type diff_filter -options_file examples/fluids/tests-output/blasius_test.yaml -compare_final_state_atol 2e-12 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius_diff_filter_iso.bin -diff_filter_monitor -ts_max_steps 0 -diff_filter_width_scaling 4.2e-5,4.2e-5,4.2e-5 -diff_filter_ksp_atol 1e-14 -diff_filter_ksp_rtol 1e-16
//TESTARGS(name="Gaussian Wave, with IDL") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/gaussianwave.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-gaussianwave-IDL.bin -dm_plex_box_faces 5,5,1 -ts_max_steps 5 -idl_decay_time 2e-3 -idl_length 0.25 -idl_start 0 -ts_alpha_radius 0.5 -idl_pressure 70
//TESTARGS(name="Spanwise Turbulence Statistics") -ceed {ceed_resource} -test_type turb_spanstats -options_file examples/fluids/tests-output/stats_test.yaml -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-turb-spanstats-stats.bin
//TESTARGS(name="Blasius") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/tests-output/blasius_test.yaml -compare_final_state_atol 2E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius.bin
//TESTARGS(name="Blasius, STG Inflow") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/tests-output/blasius_stgtest.yaml -compare_final_state_atol 2E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius_STG.bin
//TESTARGS(name="Blasius, STG Inflow, Weak Temperature") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/tests-output/blasius_stgtest.yaml -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius_STG_weakT.bin -weakT
//TESTARGS(name="Blasius, Strong STG Inflow") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/tests-output/blasius_stgtest.yaml -compare_final_state_atol 1E-10 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius_STG_strongBC.bin -stg_strong true
//TESTARGS(name="Channel") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/channel.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-channel.bin -dm_plex_box_faces 5,5,1 -ts_max_steps 5
//TESTARGS(name="Channel, Primitive") -ceed {ceed_resource} -test_type solver -options_file examples/fluids/channel.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-channel-prim.bin -dm_plex_box_faces 5,5,1 -ts_max_steps 5 -state_var primitive
//TESTARGS(name="Density Current, explicit") -ceed {ceed_resource} -test_type solver -degree 3 -q_extra 2 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -bc_symmetry_x 5,6 -bc_symmetry_y 3,4 -bc_symmetry_z 1,2 -units_kilogram 1e-9 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -mu 75 -gravity 0,0,-9.81 -ts_dt 1e-3 -units_meter 1e-2 -units_second 1e-2 -ts_max_steps 10 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-dc-explicit.bin
//TESTARGS(name="Density Current, implicit, no stabilization") -ceed {ceed_resource} -test_type solver -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -bc_symmetry_x 5,6 -bc_symmetry_y 3,4 -bc_symmetry_z 1,2 -units_kilogram 1e-9 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -mu 75 -gravity 0,0,-9.81 -units_meter 1e-2 -units_second 1e-2 -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -ts_max_steps 10 -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-dc-implicit-stab-none.bin
//TESTARGS(name="Advection, rotation, implicit, SUPG stabilization") -ceed {ceed_resource} -test_type solver -problem advection -CtauS .3 -stab supg -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -bc_wall 1,2,3,4,5,6 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -dm_mat_preallocate_skip 0 -ts_type alpha -compare_final_state_atol 5E-4 -ts_max_steps 10 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-rotation-implicit-stab-supg.bin
//TESTARGS(name="Advection, translation, implicit, SU stabilization") -ceed {ceed_resource} -test_type solver -problem advection -CtauS .3 -stab su -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -units_kilogram 1e-9 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -dm_mat_preallocate_skip 0 -ts_type alpha -wind_type translation -wind_translation .53,-1.33,-2.65 -bc_inflow 1,2,3,4,5,6 -ts_max_steps 10 -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-translation-implicit-stab-su.bin
//TESTARGS(name="Advection 2D, rotation, explicit, strong form") -ceed {ceed_resource} -test_type solver -problem advection -strong_form 1 -degree 3 -dm_plex_box_faces 2,2 -dm_plex_box_lower 0,0 -dm_plex_box_upper 125,125 -bc_wall 1,2,3,4 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ts_dt 1e-3 -compare_final_state_atol 5E-11 -ts_max_steps 10 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-rotation-explicit-strong.bin
//TESTARGS(name="Advection 2D, rotation, implicit, SUPG stabilization") -ceed {ceed_resource} -test_type solver -problem advection -CtauS .3 -stab supg -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0 -dm_plex_box_upper 125,125 -bc_wall 1,2,3,4 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -dm_mat_preallocate_skip 0 -ts_type alpha -ts_max_steps 10 -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-rotation-implicit-stab-supg.bin
//TESTARGS(name="Euler, implicit") -ceed {ceed_resource} -test_type solver -problem euler_vortex -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -units_meter 1e-4 -units_second 1e-4 -mean_velocity 1.4,-2.,0 -bc_inflow 4,6 -bc_outflow 3,5 -bc_symmetry_z 1,2 -vortex_strength 2 -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -dm_mat_preallocate_skip 0 -ts_type alpha -ts_max_steps 10 -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-euler-implicit.bin
//TESTARGS(name="Euler, explicit") -ceed {ceed_resource} -test_type solver -problem euler_vortex -degree 3 -q_extra 2 -dm_plex_box_faces 2,2,1 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -units_meter 1e-4 -units_second 1e-4 -mean_velocity 1.4,-2.,0 -bc_inflow 4,6 -bc_outflow 3,5 -bc_symmetry_z 1,2 -vortex_strength 2 -ts_dt 1e-7 -ts_rk_type 5bs -ts_rtol 1e-10 -ts_atol 1e-10 -ts_max_steps 10 -compare_final_state_atol 1E-7 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-euler-explicit.bin
//TESTARGS(name="Sod Shocktube, explicit, SU stabilization, y-z-beta shock capturing") -ceed {ceed_resource} -test_type solver -problem shocktube -degree 1 -q_extra 2 -dm_plex_box_faces 50,1,1 -units_meter 1e-2 units_second 1e-2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1000,20,20 -dm_plex_dim 3 -bc_symmetry_x 5,6 -bc_symmetry_y 3,4 -bc_symmetry_z 1,2 -yzb -stab su -ts_max_steps 10 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-shocktube-explicit-su-yzb.bin

/// @file
/// Navier-Stokes example using PETSc

const char help[] = "Solve Navier-Stokes using PETSc and libCEED\n";

#include "navierstokes.h"
#include <petscdevice.h>

#include <ceed.h>
#include <petscdmplex.h>
#include <petscts.h>

int main(int argc, char **argv) {
  // ---------------------------------------------------------------------------
  // Initialize PETSc
  // ---------------------------------------------------------------------------
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // ---------------------------------------------------------------------------
  // Create structs
  // ---------------------------------------------------------------------------
  AppCtx app_ctx;
  PetscCall(PetscCalloc1(1, &app_ctx));

  ProblemData problem;
  PetscCall(PetscCalloc1(1, &problem));

  User user;
  PetscCall(PetscCalloc1(1, &user));

  CeedData ceed_data;
  PetscCall(PetscCalloc1(1, &ceed_data));

  SimpleBC bc;
  PetscCall(PetscCalloc1(1, &bc));

  Physics phys_ctx;
  PetscCall(PetscCalloc1(1, &phys_ctx));

  Units units;
  PetscCall(PetscCalloc1(1, &units));

  user->app_ctx            = app_ctx;
  user->units              = units;
  user->phys               = phys_ctx;
  problem->set_bc_from_ics = PETSC_TRUE;

  PetscCall(RegisterLogEvents());

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  // -- Register problems to be available on the command line
  PetscCall(RegisterProblems_NS(app_ctx));

  // -- Process general command line options
  MPI_Comm comm = PETSC_COMM_WORLD;
  user->comm    = comm;
  PetscCall(ProcessCommandLineOptions(comm, app_ctx, bc));
  PetscCall(BoundaryConditionSetUp(user, problem, app_ctx, bc));

  // ---------------------------------------------------------------------------
  // Initialize libCEED
  // ---------------------------------------------------------------------------
  // -- Initialize backend
  Ceed ceed;
  PetscCheck(CeedInit(app_ctx->ceed_resource, &ceed) == CEED_ERROR_SUCCESS, comm, PETSC_ERR_LIB, "Ceed initialization failed");
  user->ceed = ceed;

  PetscCheck(CeedSetErrorHandler(ceed, CeedErrorStore) == CEED_ERROR_SUCCESS, comm, PETSC_ERR_LIB, "Setting libCEED error handler failed");

  // -- Check preferred MemType
  CeedMemType mem_type_backend;
  PetscCallCeed(ceed, CeedGetPreferredMemType(ceed, &mem_type_backend));

  {
    const char *resource;
    PetscCallCeed(ceed, CeedGetResource(ceed, &resource));
    if (strstr(resource, "/gpu/sycl")) {
      PetscDeviceContext dctx;
      PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
      void *stream_handle;
      PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream_handle));
      PetscCallCeed(ceed, CeedSetStream(ceed, stream_handle));
    }
  }

  // ---------------------------------------------------------------------------
  // Set up global mesh
  // ---------------------------------------------------------------------------
  // -- Create DM
  DM      dm;
  VecType vec_type = NULL;
  MatType mat_type = NULL;
  switch (mem_type_backend) {
    case CEED_MEM_HOST:
      vec_type = VECSTANDARD;
      break;
    case CEED_MEM_DEVICE: {
      const char *resolved;
      PetscCallCeed(ceed, CeedGetResource(ceed, &resolved));
      if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
      else if (strstr(resolved, "/gpu/hip")) vec_type = VECKOKKOS;
      else if (strstr(resolved, "/gpu/sycl")) vec_type = VECKOKKOS;
      else vec_type = VECSTANDARD;
    }
  }
  if (strstr(vec_type, VECCUDA)) mat_type = MATAIJCUSPARSE;
  else if (strstr(vec_type, VECKOKKOS)) mat_type = MATAIJKOKKOS;
  else mat_type = MATAIJ;
  PetscCall(CreateDM(comm, problem, mat_type, vec_type, &dm));
  user->dm = dm;
  PetscCall(DMSetApplicationContext(dm, user));

  // ---------------------------------------------------------------------------
  // Choose the problem from the list of registered problems
  // ---------------------------------------------------------------------------
  {
    PetscErrorCode (*p)(ProblemData, DM, void *, SimpleBC);
    PetscCall(PetscFunctionListFind(app_ctx->problems, app_ctx->problem_name, &p));
    PetscCheck(p, PETSC_COMM_SELF, 1, "Problem '%s' not found", app_ctx->problem_name);
    PetscCall((*p)(problem, dm, &user, bc));
  }

  // -- Set up DM
  PetscCall(SetUpDM(dm, problem, app_ctx->degree, app_ctx->q_extra, bc, phys_ctx));

  // -- Refine DM for high-order viz
  if (app_ctx->viz_refine) PetscCall(VizRefineDM(dm, user, problem, bc, phys_ctx));

  // ---------------------------------------------------------------------------
  // Create solution vectors
  // ---------------------------------------------------------------------------
  // -- Set up global state vector Q
  Vec Q;
  PetscCall(DMCreateGlobalVector(dm, &Q));
  PetscCall(VecZeroEntries(Q));

  // -- Set up local state vectors Q_loc, Q_dot_loc
  PetscCall(DMCreateLocalVector(dm, &user->Q_loc));
  PetscCall(DMCreateLocalVector(dm, &user->Q_dot_loc));
  PetscCall(VecZeroEntries(user->Q_dot_loc));

  // ---------------------------------------------------------------------------
  // Set up libCEED
  // ---------------------------------------------------------------------------
  // -- Set up libCEED objects
  PetscCall(SetupLibceed(ceed, ceed_data, dm, user, app_ctx, problem, bc));

  // ---------------------------------------------------------------------------
  // Set up ICs
  // ---------------------------------------------------------------------------
  // -- Fix multiplicity for ICs
  PetscCall(ICs_FixMultiplicity(dm, ceed_data, user, user->Q_loc, Q, 0.0));

  // ---------------------------------------------------------------------------
  // Record boundary values from initial condition
  // ---------------------------------------------------------------------------
  // -- This overrides DMPlexInsertBoundaryValues().
  //    We use this for the main simulation DM because the reference DMPlexInsertBoundaryValues() is very slow on the GPU due to extra device-to-host
  //    communication. If we disable this, we should still get the same results due to the problem->bc function, but with potentially much slower
  //    execution.
  if (problem->set_bc_from_ics) {
    PetscCall(SetBCsFromICs(dm, Q, user->Q_loc));
  }

  // ---------------------------------------------------------------------------
  // Create output directory
  // ---------------------------------------------------------------------------
  PetscMPIInt rank;
  MPI_Comm_rank(comm, &rank);
  if (!rank) {
    PetscCall(PetscMkdir(app_ctx->output_dir));
  }

  // ---------------------------------------------------------------------------
  // Gather initial Q values in case of continuation of simulation
  // ---------------------------------------------------------------------------
  // -- Set up initial values from binary file
  if (app_ctx->cont_steps) {
    PetscCall(SetupICsFromBinary(comm, app_ctx, Q));
  }

  // -- Zero Q_loc
  PetscCall(VecZeroEntries(user->Q_loc));

  // ---------------------------------------------------------------------------
  // TS: Create, setup, and solve
  // ---------------------------------------------------------------------------
  TS          ts;
  PetscScalar final_time;
  PetscCall(TSSolve_NS(dm, user, app_ctx, phys_ctx, problem, &Q, &final_time, &ts));

  // ---------------------------------------------------------------------------
  // Post-processing
  // ---------------------------------------------------------------------------
  PetscCall(PostProcess(ts, ceed_data, dm, problem, user, Q, final_time));

  // ---------------------------------------------------------------------------
  // Destroy libCEED objects
  // ---------------------------------------------------------------------------

  PetscCall(TurbulenceStatisticsDestroy(user, ceed_data));
  PetscCall(NodalProjectionDataDestroy(user->grad_velo_proj));
  PetscCall(DifferentialFilterDataDestroy(user->diff_filter));

  // -- Vectors
  PetscCallCeed(ceed, CeedVectorDestroy(&ceed_data->x_coord));
  PetscCallCeed(ceed, CeedVectorDestroy(&ceed_data->q_data));
  PetscCallCeed(ceed, CeedVectorDestroy(&user->q_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&user->q_dot_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&user->g_ceed));

  // -- Bases
  PetscCallCeed(ceed, CeedBasisDestroy(&ceed_data->basis_q));
  PetscCallCeed(ceed, CeedBasisDestroy(&ceed_data->basis_x));

  // -- Restrictions
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&ceed_data->elem_restr_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&ceed_data->elem_restr_x));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&ceed_data->elem_restr_qd_i));

  // Destroy QFunction contexts after using
  // ToDo: Simplify tracked libCEED objects, smaller struct
  {
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_inflow.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_inflow_jacobian.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_outflow.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_outflow_jacobian.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_freestream.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_freestream_jacobian.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_slip.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_slip_jacobian.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->ics.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_vol_rhs.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_vol_ifunction.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->apply_vol_ijacobian.qfunction_context));
  }

  // -- Operators
  PetscCall(OperatorApplyContextDestroy(ceed_data->op_ics_ctx));
  PetscCall(OperatorApplyContextDestroy(user->op_rhs_ctx));
  PetscCall(OperatorApplyContextDestroy(user->op_strong_bc_ctx));
  PetscCallCeed(ceed, CeedOperatorDestroy(&user->op_ifunction));

  // -- Ceed
  PetscCheck(CeedDestroy(&ceed) == CEED_ERROR_SUCCESS, comm, PETSC_ERR_LIB, "Destroying Ceed object failed");

  if (app_ctx->test_type != TESTTYPE_NONE) {
    PetscInt num_options_left = 0;
    PetscCall(PetscOptionsLeftGet(NULL, &num_options_left, NULL, NULL));
    PetscCheck(num_options_left == 0, PETSC_COMM_WORLD, -1,
               "There are unused options. This is not allowed. See error message for the unused options (or use -options_left directly)");
  }

  // ---------------------------------------------------------------------------
  // Clean up PETSc
  // ---------------------------------------------------------------------------
  // -- Vectors
  PetscCall(VecDestroy(&Q));
  PetscCall(VecDestroy(&user->Q_loc));
  PetscCall(VecDestroy(&user->Q_dot_loc));

  PetscCall(KSPDestroy(&user->mass_ksp));

  // -- Matrices
  PetscCall(MatDestroy(&user->interp_viz));
  PetscCall(MatDestroy(&user->mat_ijacobian));

  // -- DM
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&user->dm_viz));

  // -- TS
  PetscCall(TSDestroy(&ts));

  // -- Function list
  PetscCall(PetscFunctionListDestroy(&app_ctx->problems));

  PetscCall(PetscFree(app_ctx->amat_type));
  PetscCall(PetscFree(app_ctx->wall_forces.walls));
  PetscCall(PetscViewerDestroy(&app_ctx->wall_forces.viewer));
  PetscCall(PetscViewerDestroy(&app_ctx->turb_spanstats_viewer));

  // -- Structs
  for (PetscInt i = 0; i < problem->num_bc_defs; i++) {
    PetscCall(BCDefinitionDestroy(&problem->bc_defs[i]));
  }
  PetscCall(PetscFree(problem->bc_defs));
  PetscCall(PetscFree(units));
  PetscCall(PetscFree(user));
  PetscCall(PetscFree(problem));
  PetscCall(PetscFree(bc));
  PetscCall(PetscFree(phys_ctx));
  PetscCall(PetscFree(app_ctx));
  PetscCall(PetscFree(ceed_data));
  PetscCall(PetscFree(problem));

  return PetscFinalize();
}
