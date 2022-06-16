// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                        libCEED + PETSc Example: Navier-Stokes
//
// This example demonstrates a simple usage of libCEED with PETSc to solve a
// Navier-Stokes problem.
//
// The code is intentionally "raw", using only low-level communication
// primitives.
//
// Build with:
//
//     make [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>] navierstokes
//
// Sample runs:
//
//     ./navierstokes -ceed /cpu/self -problem density_current -degree 1
//     ./navierstokes -ceed /gpu/cuda -problem advection -degree 1
//
//TESTARGS(name="channel") -ceed {ceed_resource} -test -options_file examples/fluids/channel.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-channel.bin -snes_fd_color
//TESTARGS(name="dc_explicit") -ceed {ceed_resource} -test -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -bc_slip_x 5,6 -bc_slip_y 3,4 -bc_Slip_z 1,2 -units_kilogram 1e-9 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -mu 75 -ts_dt 1e-3 -units_meter 1e-2 -units_second 1e-2 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-dc-explicit.bin
//TESTARGS(name="dc_implicit_stab_none") -ceed {ceed_resource} -test -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -bc_slip_x 5,6 -bc_slip_y 3,4 -bc_Slip_z 1,2 -units_kilogram 1e-9 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -mu 75 -units_meter 1e-2 -units_second 1e-2 -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-dc-implicit-stab-none.bin -snes_fd_color
//TESTARGS(name="adv_rotation_explicit_strong") -ceed {ceed_resource} -test -problem advection -strong_form 1 -degree 3 -dm_plex_box_faces 2,2,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -bc_wall 1,2,3,4,5,6 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ts_dt 1e-3 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-rotation-explicit-strong.bin
//TESTARGS(name="adv_rotation_implicit_sharp_cylinder") -ceed {ceed_resource} -test -problem advection -bubble_type cylinder -bubble_continuity back_sharp -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -bc_Slip_z 1,2 -bc_wall 3,4,5,6 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-rotation-implicit-sharp-cylinder.bin
//TESTARGS(name="adv_rotation_implicit_stab_supg") -ceed {ceed_resource} -test -problem advection -CtauS .3 -stab supg -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -bc_wall 1,2,3,4,5,6 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-rotation-implicit-stab-supg.bin
//TESTARGS(name="adv_translation_implicit_stab_su") -ceed {ceed_resource} -test -problem advection -CtauS .3 -stab su -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -units_kilogram 1e-9 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -wind_type translation -wind_translation .53,-1.33,-2.65 -bc_inflow 1,2,3,4,5,6 -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-translation-implicit-stab-su.bin
//TESTARGS(name="adv2d_rotation_explicit_strong") -ceed {ceed_resource} -test -problem advection2d -strong_form 1 -degree 3 -dm_plex_box_faces 2,2 -dm_plex_box_lower 0,0 -dm_plex_box_upper 125,125 -bc_wall 1,2,3,4 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ts_dt 1e-3 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-rotation-explicit-strong.bin
//TESTARGS(name="adv2d_rotation_implicit_stab_supg") -ceed {ceed_resource} -test -problem advection2d -CtauS .3 -stab supg -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0 -dm_plex_box_upper 125,125 -bc_wall 1,2,3,4 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-rotation-implicit-stab-supg.bin
//TESTARGS(name="adv2d_translation_implicit_stab_su") -ceed {ceed_resource} -test -problem advection2d -CtauS .3 -stab su -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0 -dm_plex_box_upper 125,125 -units_kilogram 1e-9 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -wind_type translation -wind_translation .53,-1.33,0 -bc_inflow 1,2,3,4 -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-translation-implicit-stab-su.bin
//TESTARGS(name="euler_implicit") -ceed {ceed_resource} -test -problem euler_vortex -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -units_meter 1e-4 -units_second 1e-4 -mean_velocity 1.4,-2.,0 -bc_inflow 4,6 -bc_outflow 3,5 -bc_slip_z 1,2 -vortex_strength 2 -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-euler-implicit.bin
//TESTARGS(name="euler_explicit") -ceed {ceed_resource} -test -problem euler_vortex -degree 3 -dm_plex_box_faces 2,2,1 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -units_meter 1e-4 -units_second 1e-4 -mean_velocity 1.4,-2.,0 -bc_inflow 4,6 -bc_outflow 3,5 -bc_slip_z 1,2 -vortex_strength 2 -ts_dt 1e-7 -ts_rk_type 5bs -ts_rtol 1e-10 -ts_atol 1e-10 -compare_final_state_atol 1E-7 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-euler-explicit.bin
//TESTARGS(name="shocktube_explicit_su_yzb") -ceed {ceed_resource} -test -problem shocktube -degree 1 -dm_plex_box_faces 50,1,1 -units_meter 1e-2 units_second 1e-2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1000,20,20 -dm_plex_dim 3 -bc_slip_x 5,6 -bc_slip_y 3,4 -bc_Slip_z 1,2 -yzb -stab su -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-shocktube-explicit-su-yzb.bin
//TESTARGS(name="blasius_STG") -ceed {ceed_resource} -test -options_file examples/fluids/tests-output/blasius_stgtest.yaml -compare_final_state_atol 2E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius_STG.bin -snes_fd_color
//TESTARGS(name="blasius_STG_weakT") -ceed {ceed_resource} -test -options_file examples/fluids/tests-output/blasius_stgtest.yaml -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius_STG_weakT.bin -weakT -snes_fd_color
//TESTARGS(name="blasius_STG_strongBC") -ceed {ceed_resource} -test -options_file examples/fluids/tests-output/blasius_stgtest.yaml -compare_final_state_atol 1E-10 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius_STG_strongBC.bin -stg_strong true

/// @file
/// Navier-Stokes example using PETSc

const char help[] = "Solve Navier-Stokes using PETSc and libCEED\n";

#include "navierstokes.h"

int main(int argc, char **argv) {
  // ---------------------------------------------------------------------------
  // Initialize PETSc
  // ---------------------------------------------------------------------------
  PetscInt ierr;
  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  // ---------------------------------------------------------------------------
  // Create structs
  // ---------------------------------------------------------------------------
  AppCtx app_ctx;
  ierr = PetscCalloc1(1, &app_ctx); CHKERRQ(ierr);

  ProblemData *problem = NULL;
  ierr = PetscCalloc1(1, &problem); CHKERRQ(ierr);

  User user;
  ierr = PetscCalloc1(1, &user); CHKERRQ(ierr);

  CeedData ceed_data;
  ierr = PetscCalloc1(1, &ceed_data); CHKERRQ(ierr);

  SimpleBC bc;
  ierr = PetscCalloc1(1, &bc); CHKERRQ(ierr);

  Physics phys_ctx;
  ierr = PetscCalloc1(1, &phys_ctx); CHKERRQ(ierr);

  Units units;
  ierr = PetscCalloc1(1, &units); CHKERRQ(ierr);

  user->app_ctx = app_ctx;
  user->units   = units;
  user->phys    = phys_ctx;
  problem->bc_from_ics = PETSC_TRUE;

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  // -- Register problems to be available on the command line
  ierr = RegisterProblems_NS(app_ctx); CHKERRQ(ierr);

  // -- Process general command line options
  MPI_Comm comm = PETSC_COMM_WORLD;
  user->comm = comm;
  ierr = ProcessCommandLineOptions(comm, app_ctx, bc); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Initialize libCEED
  // ---------------------------------------------------------------------------
  // -- Initialize backend
  Ceed ceed;
  CeedInit(app_ctx->ceed_resource, &ceed);
  user->ceed = ceed;

  // -- Check preferred MemType
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  // ---------------------------------------------------------------------------
  // Set up global mesh
  // ---------------------------------------------------------------------------
  // -- Create DM
  DM dm;
  VecType vec_type = NULL;
  MatType mat_type = NULL;
  switch (mem_type_backend) {
  case CEED_MEM_HOST: vec_type = VECSTANDARD; break;
  case CEED_MEM_DEVICE: {
    const char *resolved;
    CeedGetResource(ceed, &resolved);
    if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
    else if (strstr(resolved, "/gpu/hip")) vec_type = VECKOKKOS;
    else vec_type = VECSTANDARD;
  }
  }
  if (strstr(vec_type, VECCUDA)) mat_type = MATAIJCUSPARSE;
  else if (strstr(vec_type, VECKOKKOS)) mat_type = MATAIJKOKKOS;
  else mat_type = MATAIJ;
  ierr = CreateDM(comm, problem, mat_type, vec_type, &dm); CHKERRQ(ierr);
  user->dm = dm;

  // ---------------------------------------------------------------------------
  // Choose the problem from the list of registered problems
  // ---------------------------------------------------------------------------
  {
    PetscErrorCode (*p)(ProblemData *, DM, void *);
    ierr = PetscFunctionListFind(app_ctx->problems, app_ctx->problem_name, &p);
    CHKERRQ(ierr);
    if (!p) SETERRQ(PETSC_COMM_SELF, 1, "Problem '%s' not found",
                      app_ctx->problem_name);
    ierr = (*p)(problem, dm, &user); CHKERRQ(ierr);
  }

  // -- Set up DM
  ierr = SetUpDM(dm, problem, app_ctx->degree, bc, phys_ctx);
  CHKERRQ(ierr);

  // -- Refine DM for high-order viz
  if (app_ctx->viz_refine) {
    ierr = VizRefineDM(dm, user, problem, bc, phys_ctx);
    CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Set up libCEED
  // ---------------------------------------------------------------------------
  // -- Set up libCEED objects
  ierr = SetupLibceed(ceed, ceed_data, dm, user, app_ctx, problem, bc);
  CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Set up ICs
  // ---------------------------------------------------------------------------
  // -- Set up global state vector Q
  Vec Q;
  ierr = DMCreateGlobalVector(dm, &Q); CHKERRQ(ierr);
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);

  // -- Set up local state vectors Q_loc, Q_dot_loc
  ierr = DMCreateLocalVector(dm, &user->Q_loc); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &user->Q_dot_loc); CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Q_dot_loc); CHKERRQ(ierr);

  // -- Fix multiplicity for ICs
  ierr = ICs_FixMultiplicity(dm, ceed_data, user, user->Q_loc, Q, 0.0);
  CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Set up lumped mass matrix
  // ---------------------------------------------------------------------------
  // -- Set up global mass vector
  ierr = VecDuplicate(Q, &user->M); CHKERRQ(ierr);

  // -- Compute lumped mass matrix
  ierr = ComputeLumpedMassMatrix(ceed, dm, ceed_data, user->M); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Record boundary values from initial condition
  // ---------------------------------------------------------------------------
  // -- This overrides DMPlexInsertBoundaryValues().
  //    We use this for the main simulation DM because the reference
  //    DMPlexInsertBoundaryValues() is very slow. If we disable this, we should
  //    still get the same results due to the problem->bc function, but with
  //    potentially much slower execution.
  if (problem->bc_from_ics) {
    ierr = SetBCsFromICs_NS(dm, Q, user->Q_loc); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Create output directory
  // ---------------------------------------------------------------------------
  PetscMPIInt rank;
  MPI_Comm_rank(comm, &rank);
  if (!rank) {ierr = PetscMkdir(app_ctx->output_dir); CHKERRQ(ierr);}

  // ---------------------------------------------------------------------------
  // Gather initial Q values in case of continuation of simulation
  // ---------------------------------------------------------------------------
  // -- Set up initial values from binary file
  if (app_ctx->cont_steps) {
    ierr = SetupICsFromBinary(comm, app_ctx, Q); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Print problem summary
  // ---------------------------------------------------------------------------
  if (!app_ctx->test_mode) {
    // Header and rank
    char host_name[PETSC_MAX_PATH_LEN];
    int  comm_size;
    ierr = PetscGetHostName(host_name, sizeof host_name); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &comm_size); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "\n-- Navier-Stokes solver - libCEED + PETSc --\n"
                       "  MPI:\n"
                       "    Host Name                          : %s\n"
                       "    Total ranks                        : %d\n",
                       host_name, comm_size); CHKERRQ(ierr);

    // Problem specific info
    ierr = problem->print_info(problem, app_ctx); CHKERRQ(ierr);

    // libCEED
    const char *used_resource;
    CeedGetResource(ceed, &used_resource);
    ierr = PetscPrintf(comm,
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n",
                       used_resource, CeedMemTypes[mem_type_backend]); CHKERRQ(ierr);
    // PETSc
    char box_faces_str[PETSC_MAX_PATH_LEN] = "3,3,3";
    if (problem->dim == 2) box_faces_str[3] = '\0';
    ierr = PetscOptionsGetString(NULL, NULL, "-dm_plex_box_faces", box_faces_str,
                                 sizeof(box_faces_str), NULL); CHKERRQ(ierr);
    MatType mat_type;
    VecType vec_type;
    ierr = DMGetMatType(dm, &mat_type); CHKERRQ(ierr);
    ierr = DMGetVecType(dm, &vec_type); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "  PETSc:\n"
                       "    Box Faces                          : %s\n"
                       "    DM MatType                         : %s\n"
                       "    DM VecType                         : %s\n"
                       "    Time Stepping Scheme               : %s\n",
                       box_faces_str, mat_type, vec_type,
                       phys_ctx->implicit ? "implicit" : "explicit"); CHKERRQ(ierr);
    // Mesh
    const PetscInt num_comp_q = 5;
    CeedInt        glob_dofs, owned_dofs;
    PetscInt       glob_nodes, owned_nodes;
    const CeedInt  num_P = app_ctx->degree + 1,
                   num_Q = num_P + app_ctx->q_extra;
    // -- Get global size
    ierr = VecGetSize(Q, &glob_dofs); CHKERRQ(ierr);
    ierr = VecGetLocalSize(Q, &owned_dofs); CHKERRQ(ierr);
    glob_nodes = glob_dofs/num_comp_q;
    // -- Get local size
    ierr = VecGetSize(user->Q_loc, &owned_nodes); CHKERRQ(ierr);
    owned_nodes /= num_comp_q;
    ierr = PetscPrintf(comm,
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (P)       : %d\n"
                       "    Number of 1D Quadrature Points (Q) : %d\n"
                       "    Global DoFs                        : %" PetscInt_FMT "\n"
                       "    Owned DoFs                         : %" PetscInt_FMT "\n"
                       "    DoFs per node                      : %" PetscInt_FMT "\n"
                       "    Global nodes                       : %" PetscInt_FMT "\n"
                       "    Owned nodes                        : %" PetscInt_FMT "\n",
                       num_P, num_Q, glob_dofs, owned_dofs, num_comp_q,
                       glob_nodes, owned_nodes); CHKERRQ(ierr);
  }
  // -- Zero Q_loc
  ierr = VecZeroEntries(user->Q_loc); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // TS: Create, setup, and solve
  // ---------------------------------------------------------------------------
  TS ts;
  PetscScalar final_time;
  ierr = TSSolve_NS(dm, user, app_ctx, phys_ctx, &Q, &final_time, &ts);
  CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Post-processing
  // ---------------------------------------------------------------------------
  ierr = PostProcess_NS(ts, ceed_data, dm, problem, user, Q, final_time);
  CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Destroy libCEED objects
  // ---------------------------------------------------------------------------
  // -- Vectors
  CeedVectorDestroy(&ceed_data->x_coord);
  CeedVectorDestroy(&ceed_data->q_data);
  CeedVectorDestroy(&user->q_ceed);
  CeedVectorDestroy(&user->q_dot_ceed);
  CeedVectorDestroy(&user->g_ceed);
  CeedVectorDestroy(&user->coo_values);

  // -- QFunctions
  CeedQFunctionDestroy(&ceed_data->qf_setup_vol);
  CeedQFunctionDestroy(&ceed_data->qf_ics);
  CeedQFunctionDestroy(&ceed_data->qf_rhs_vol);
  CeedQFunctionDestroy(&ceed_data->qf_ifunction_vol);
  CeedQFunctionDestroy(&ceed_data->qf_setup_sur);
  CeedQFunctionDestroy(&ceed_data->qf_apply_inflow);
  CeedQFunctionDestroy(&ceed_data->qf_apply_outflow);

  // -- Bases
  CeedBasisDestroy(&ceed_data->basis_q);
  CeedBasisDestroy(&ceed_data->basis_x);
  CeedBasisDestroy(&ceed_data->basis_xc);
  CeedBasisDestroy(&ceed_data->basis_q_sur);
  CeedBasisDestroy(&ceed_data->basis_x_sur);

  // -- Restrictions
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_q);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_x);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_qd_i);

  // -- Operators
  CeedOperatorDestroy(&ceed_data->op_setup_vol);
  CeedOperatorDestroy(&ceed_data->op_ics);
  CeedOperatorDestroy(&user->op_rhs_vol);
  CeedOperatorDestroy(&user->op_ifunction_vol);
  CeedOperatorDestroy(&user->op_rhs);
  CeedOperatorDestroy(&user->op_ifunction);
  CeedOperatorDestroy(&user->op_ijacobian);

  // -- Ceed
  CeedDestroy(&ceed);

  // ---------------------------------------------------------------------------
  // Clean up PETSc
  // ---------------------------------------------------------------------------
  // -- Vectors
  ierr = VecDestroy(&Q); CHKERRQ(ierr);
  ierr = VecDestroy(&user->M); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Q_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Q_dot_loc); CHKERRQ(ierr);

  // -- Matrices
  ierr = MatDestroy(&user->interp_viz); CHKERRQ(ierr);

  // -- DM
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = DMDestroy(&user->dm_viz); CHKERRQ(ierr);

  // -- TS
  ierr = TSDestroy(&ts); CHKERRQ(ierr);

  // -- Function list
  ierr = PetscFunctionListDestroy(&app_ctx->problems); CHKERRQ(ierr);

  ierr = PetscFree(problem->bc_ctx); CHKERRQ(ierr);

  // -- Structs
  ierr = PetscFree(units); CHKERRQ(ierr);
  ierr = PetscFree(user); CHKERRQ(ierr);
  ierr = PetscFree(problem); CHKERRQ(ierr);
  ierr = PetscFree(bc); CHKERRQ(ierr);
  ierr = PetscFree(phys_ctx); CHKERRQ(ierr);
  ierr = PetscFree(app_ctx); CHKERRQ(ierr);
  ierr = PetscFree(ceed_data); CHKERRQ(ierr);

  return PetscFinalize();
}
