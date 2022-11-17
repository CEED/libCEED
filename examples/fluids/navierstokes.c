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
//TESTARGS(name="blasius") -ceed {ceed_resource} -test -options_file examples/fluids/tests-output/blasius_test.yaml -compare_final_state_atol 2E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius.bin
//TESTARGS(name="blasius_STG") -ceed {ceed_resource} -test -options_file examples/fluids/tests-output/blasius_stgtest.yaml -compare_final_state_atol 2E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius_STG.bin
//TESTARGS(name="blasius_STG_weakT") -ceed {ceed_resource} -test -options_file examples/fluids/tests-output/blasius_stgtest.yaml -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius_STG_weakT.bin -weakT
//TESTARGS(name="blasius_STG_strongBC") -ceed {ceed_resource} -test -options_file examples/fluids/tests-output/blasius_stgtest.yaml -compare_final_state_atol 1E-10 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-blasius_STG_strongBC.bin -stg_strong true
//TESTARGS(name="channel") -ceed {ceed_resource} -test -options_file examples/fluids/channel.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-channel.bin -dm_plex_box_faces 5,5,1 -ts_max_steps 5
//TESTARGS(name="channel-primitive") -ceed {ceed_resource} -test -options_file examples/fluids/channel.yaml -compare_final_state_atol 2e-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-channel-prim.bin -dm_plex_box_faces 5,5,1 -ts_max_steps 5 -state_var primitive
//TESTARGS(name="dc_explicit") -ceed {ceed_resource} -test -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -bc_slip_x 5,6 -bc_slip_y 3,4 -bc_Slip_z 1,2 -units_kilogram 1e-9 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -mu 75 -ts_dt 1e-3 -units_meter 1e-2 -units_second 1e-2 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-dc-explicit.bin
//TESTARGS(name="dc_implicit_stab_none") -ceed {ceed_resource} -test -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -bc_slip_x 5,6 -bc_slip_y 3,4 -bc_Slip_z 1,2 -units_kilogram 1e-9 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -mu 75 -units_meter 1e-2 -units_second 1e-2 -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-dc-implicit-stab-none.bin
//TESTARGS(name="adv_rotation_implicit_stab_supg") -ceed {ceed_resource} -test -problem advection -CtauS .3 -stab supg -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -bc_wall 1,2,3,4,5,6 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -dm_mat_preallocate_skip 0 -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-rotation-implicit-stab-supg.bin
//TESTARGS(name="adv_translation_implicit_stab_su") -ceed {ceed_resource} -test -problem advection -CtauS .3 -stab su -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -units_kilogram 1e-9 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -dm_mat_preallocate_skip 0 -ts_type alpha -wind_type translation -wind_translation .53,-1.33,-2.65 -bc_inflow 1,2,3,4,5,6 -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-translation-implicit-stab-su.bin
//TESTARGS(name="adv2d_rotation_explicit_strong") -ceed {ceed_resource} -test -problem advection2d -strong_form 1 -degree 3 -dm_plex_box_faces 2,2 -dm_plex_box_lower 0,0 -dm_plex_box_upper 125,125 -bc_wall 1,2,3,4 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ts_dt 1e-3 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-rotation-explicit-strong.bin
//TESTARGS(name="adv2d_rotation_implicit_stab_supg") -ceed {ceed_resource} -test -problem advection2d -CtauS .3 -stab supg -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0 -dm_plex_box_upper 125,125 -bc_wall 1,2,3,4 -wall_comps 4 -units_kilogram 1e-9 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -dm_mat_preallocate_skip 0 -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-rotation-implicit-stab-supg.bin
//TESTARGS(name="euler_implicit") -ceed {ceed_resource} -test -problem euler_vortex -degree 3 -dm_plex_box_faces 1,1,2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -units_meter 1e-4 -units_second 1e-4 -mean_velocity 1.4,-2.,0 -bc_inflow 4,6 -bc_outflow 3,5 -bc_slip_z 1,2 -vortex_strength 2 -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -dm_mat_preallocate_skip 0 -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-euler-implicit.bin
//TESTARGS(name="euler_explicit") -ceed {ceed_resource} -test -problem euler_vortex -degree 3 -dm_plex_box_faces 2,2,1 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 125,125,250 -dm_plex_dim 3 -units_meter 1e-4 -units_second 1e-4 -mean_velocity 1.4,-2.,0 -bc_inflow 4,6 -bc_outflow 3,5 -bc_slip_z 1,2 -vortex_strength 2 -ts_dt 1e-7 -ts_rk_type 5bs -ts_rtol 1e-10 -ts_atol 1e-10 -compare_final_state_atol 1E-7 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-euler-explicit.bin
//TESTARGS(name="shocktube_explicit_su_yzb") -ceed {ceed_resource} -test -problem shocktube -degree 1 -dm_plex_box_faces 50,1,1 -units_meter 1e-2 units_second 1e-2 -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1000,20,20 -dm_plex_dim 3 -bc_slip_x 5,6 -bc_slip_y 3,4 -bc_Slip_z 1,2 -yzb -stab su -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-shocktube-explicit-su-yzb.bin

/// @file
/// Navier-Stokes example using PETSc

const char help[] = "Solve Navier-Stokes using PETSc and libCEED\n";

#include "navierstokes.h"

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

  ProblemData *problem = NULL;
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

  user->app_ctx        = app_ctx;
  user->units          = units;
  user->phys           = phys_ctx;
  problem->bc_from_ics = PETSC_TRUE;

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  // -- Register problems to be available on the command line
  PetscCall(RegisterProblems_NS(app_ctx));

  // -- Process general command line options
  MPI_Comm comm = PETSC_COMM_WORLD;
  user->comm    = comm;
  PetscCall(ProcessCommandLineOptions(comm, app_ctx, bc));

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
  DM      dm;
  VecType vec_type = NULL;
  MatType mat_type = NULL;
  switch (mem_type_backend) {
    case CEED_MEM_HOST:
      vec_type = VECSTANDARD;
      break;
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
  PetscCall(CreateDM(comm, problem, mat_type, vec_type, &dm));
  user->dm = dm;
  PetscCall(DMSetApplicationContext(dm, user));

  // ---------------------------------------------------------------------------
  // Choose the problem from the list of registered problems
  // ---------------------------------------------------------------------------
  {
    PetscErrorCode (*p)(ProblemData *, DM, void *);
    PetscCall(PetscFunctionListFind(app_ctx->problems, app_ctx->problem_name, &p));
    if (!p) SETERRQ(PETSC_COMM_SELF, 1, "Problem '%s' not found", app_ctx->problem_name);
    PetscCall((*p)(problem, dm, &user));
  }

  // -- Set up DM
  PetscCall(SetUpDM(dm, problem, app_ctx->degree, bc, phys_ctx));

  // -- Refine DM for high-order viz
  if (app_ctx->viz_refine) {
    PetscCall(VizRefineDM(dm, user, problem, bc, phys_ctx));
  }

  // ---------------------------------------------------------------------------
  // Set up libCEED
  // ---------------------------------------------------------------------------
  // -- Set up libCEED objects
  PetscCall(SetupLibceed(ceed, ceed_data, dm, user, app_ctx, problem, bc));

  // ---------------------------------------------------------------------------
  // Set up ICs
  // ---------------------------------------------------------------------------
  // -- Set up global state vector Q
  Vec Q;
  PetscCall(DMCreateGlobalVector(dm, &Q));
  PetscCall(VecZeroEntries(Q));

  // -- Set up local state vectors Q_loc, Q_dot_loc
  PetscCall(DMCreateLocalVector(dm, &user->Q_loc));
  PetscCall(DMCreateLocalVector(dm, &user->Q_dot_loc));
  PetscCall(VecZeroEntries(user->Q_dot_loc));

  // -- Fix multiplicity for ICs
  PetscCall(ICs_FixMultiplicity(dm, ceed_data, user, user->Q_loc, Q, 0.0));

  // ---------------------------------------------------------------------------
  // Set up lumped mass matrix
  // ---------------------------------------------------------------------------
  // -- Set up global mass vector
  PetscCall(VecDuplicate(Q, &user->M));

  // -- Compute lumped mass matrix
  PetscCall(ComputeLumpedMassMatrix(ceed, dm, ceed_data, user->M));

  // ---------------------------------------------------------------------------
  // Record boundary values from initial condition
  // ---------------------------------------------------------------------------
  // -- This overrides DMPlexInsertBoundaryValues().
  //    We use this for the main simulation DM because the reference
  //    DMPlexInsertBoundaryValues() is very slow. If we disable this, we should
  //    still get the same results due to the problem->bc function, but with
  //    potentially much slower execution.
  if (problem->bc_from_ics) {
    PetscCall(SetBCsFromICs_NS(dm, Q, user->Q_loc));
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

  // ---------------------------------------------------------------------------
  // Print problem summary
  // ---------------------------------------------------------------------------
  if (!app_ctx->test_mode) {
    // Header and rank
    char host_name[PETSC_MAX_PATH_LEN];
    int  comm_size;
    PetscCall(PetscGetHostName(host_name, sizeof host_name));
    PetscCall(MPI_Comm_size(comm, &comm_size));
    PetscCall(PetscPrintf(comm,
                          "\n-- Navier-Stokes solver - libCEED + PETSc --\n"
                          "  MPI:\n"
                          "    Host Name                          : %s\n"
                          "    Total ranks                        : %d\n",
                          host_name, comm_size));

    // Problem specific info
    PetscCall(problem->print_info(problem, app_ctx));

    // libCEED
    const char *used_resource;
    CeedGetResource(ceed, &used_resource);
    PetscCall(PetscPrintf(comm,
                          "  libCEED:\n"
                          "    libCEED Backend                    : %s\n"
                          "    libCEED Backend MemType            : %s\n",
                          used_resource, CeedMemTypes[mem_type_backend]));
    // PETSc
    char box_faces_str[PETSC_MAX_PATH_LEN] = "3,3,3";
    if (problem->dim == 2) box_faces_str[3] = '\0';
    PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_plex_box_faces", box_faces_str, sizeof(box_faces_str), NULL));
    MatType mat_type;
    VecType vec_type;
    PetscCall(DMGetMatType(dm, &mat_type));
    PetscCall(DMGetVecType(dm, &vec_type));
    PetscCall(PetscPrintf(comm,
                          "  PETSc:\n"
                          "    Box Faces                          : %s\n"
                          "    DM MatType                         : %s\n"
                          "    DM VecType                         : %s\n"
                          "    Time Stepping Scheme               : %s\n",
                          box_faces_str, mat_type, vec_type, phys_ctx->implicit ? "implicit" : "explicit"));
    // Mesh
    const PetscInt num_comp_q = 5;
    CeedInt        glob_dofs, owned_dofs;
    PetscInt       glob_nodes, owned_nodes;
    const CeedInt  num_P = app_ctx->degree + 1, num_Q = num_P + app_ctx->q_extra;
    // -- Get global size
    PetscCall(VecGetSize(Q, &glob_dofs));
    PetscCall(VecGetLocalSize(Q, &owned_dofs));
    glob_nodes = glob_dofs / num_comp_q;
    // -- Get local size
    PetscCall(VecGetSize(user->Q_loc, &owned_nodes));
    owned_nodes /= num_comp_q;
    PetscCall(PetscPrintf(comm,
                          "  Mesh:\n"
                          "    Number of 1D Basis Nodes (P)       : %" CeedInt_FMT "\n"
                          "    Number of 1D Quadrature Points (Q) : %" CeedInt_FMT "\n"
                          "    Global DoFs                        : %" PetscInt_FMT "\n"
                          "    Owned DoFs                         : %" PetscInt_FMT "\n"
                          "    DoFs per node                      : %" PetscInt_FMT "\n"
                          "    Global nodes                       : %" PetscInt_FMT "\n"
                          "    Owned nodes                        : %" PetscInt_FMT "\n",
                          num_P, num_Q, glob_dofs, owned_dofs, num_comp_q, glob_nodes, owned_nodes));
  }
  // -- Zero Q_loc
  PetscCall(VecZeroEntries(user->Q_loc));

  // ---------------------------------------------------------------------------
  // TS: Create, setup, and solve
  // ---------------------------------------------------------------------------
  TS          ts;
  PetscScalar final_time;
  PetscCall(TSSolve_NS(dm, user, app_ctx, phys_ctx, &Q, &final_time, &ts));

  // ---------------------------------------------------------------------------
  // Post-processing
  // ---------------------------------------------------------------------------
  PetscCall(PostProcess_NS(ts, ceed_data, dm, problem, user, Q, final_time));

  // ---------------------------------------------------------------------------
  // Destroy libCEED objects
  // ---------------------------------------------------------------------------
  // -- Vectors
  CeedVectorDestroy(&ceed_data->x_coord);
  CeedVectorDestroy(&ceed_data->q_data);
  CeedVectorDestroy(&user->q_ceed);
  CeedVectorDestroy(&user->q_dot_ceed);
  CeedVectorDestroy(&user->g_ceed);
  CeedVectorDestroy(&user->coo_values_amat);
  CeedVectorDestroy(&user->coo_values_pmat);

  // -- QFunctions
  CeedQFunctionDestroy(&ceed_data->qf_setup_vol);
  CeedQFunctionDestroy(&ceed_data->qf_ics);
  CeedQFunctionDestroy(&ceed_data->qf_rhs_vol);
  CeedQFunctionDestroy(&ceed_data->qf_ifunction_vol);
  CeedQFunctionDestroy(&ceed_data->qf_setup_sur);
  CeedQFunctionDestroy(&ceed_data->qf_apply_inflow);
  CeedQFunctionDestroy(&ceed_data->qf_apply_inflow_jacobian);
  CeedQFunctionDestroy(&ceed_data->qf_apply_outflow);
  CeedQFunctionDestroy(&ceed_data->qf_apply_outflow_jacobian);

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
  PetscCall(VecDestroy(&Q));
  PetscCall(VecDestroy(&user->M));
  PetscCall(VecDestroy(&user->Q_loc));
  PetscCall(VecDestroy(&user->Q_dot_loc));

  // -- Matrices
  PetscCall(MatDestroy(&user->interp_viz));

  // -- DM
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&user->dm_viz));

  // -- TS
  PetscCall(TSDestroy(&ts));

  // -- Function list
  PetscCall(PetscFunctionListDestroy(&app_ctx->problems));

  PetscCall(PetscFree(app_ctx->amat_type));

  // -- Structs
  PetscCall(PetscFree(units));
  PetscCall(PetscFree(user));
  PetscCall(PetscFree(problem));
  PetscCall(PetscFree(bc));
  PetscCall(PetscFree(phys_ctx));
  PetscCall(PetscFree(app_ctx));
  PetscCall(PetscFree(ceed_data));

  return PetscFinalize();
}
