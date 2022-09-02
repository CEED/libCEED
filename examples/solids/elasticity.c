// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                        libCEED + PETSc Example: Elasticity
//
// This example demonstrates a simple usage of libCEED with PETSc to solve
//   elasticity problems.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make elasticity [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     ./elasticity -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -problem Linear -forcing mms
//     ./elasticity -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -bc_clamp 998,999 -bc_clamp_998_translate 0.1,0.2,0.3 -problem SS-NH -forcing none -ceed
//     /cpu/self
//     ./elasticity -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -bc_clamp 998,999 -bc_clamp_998_rotate 1,0,0,0.2 -problem FSInitial-NH1 -forcing none
//     -ceed /gpu/cuda
//
// Sample meshes can be found at https://github.com/jeremylt/ceedSampleMeshes
//
//TESTARGS(name="solids-Linear-MMS") -ceed {ceed_resource} -test -degree 3 -nu 0.3 -E 1 -dm_plex_box_faces 3,3,3
//TESTARGS(name="solids-NH1-1") -ceed {ceed_resource} -test -problem FSInitial-NH1 -E 2.8 -nu 0.4 -degree 2 -dm_plex_box_faces 2,2,2 -num_steps 1 -bc_clamp 6 -bc_traction 5 -bc_traction_5 0,0,-.5 -expect_final_strain_energy 2.124627916174e-01
//TESTARGS(name="solids-MR1-1") -ceed {ceed_resource} -test -problem FSInitial-MR1 -mu_1 .5 -mu_2 .5 -nu 0.4 -degree 2 -dm_plex_box_faces 2,2,2 -num_steps 1 -bc_clamp 6 -bc_traction 5 -bc_traction_5 0,0,-.5 -expect_final_strain_energy 2.339138880207e-01

/// @file
/// CEED elasticity example using PETSc with DMPlex

const char help[] = "Solve solid Problems with CEED and PETSc DMPlex\n";

#include "elasticity.h"

int main(int argc, char **argv) {
  MPI_Comm comm;
  // Context structs
  AppCtx           app_ctx;            // Contains problem options
  ProblemFunctions problem_functions;  // Setup functions for each problem
  Units            units;              // Contains units scaling
  // PETSc objects
  PetscLogStage stage_dm_setup, stage_libceed_setup, stage_snes_setup, stage_snes_solve;
  DM            dm_orig;                   // Distributed DM to clone
  DM            dm_energy, dm_diagnostic;  // DMs for postprocessing
  DM           *level_dms;
  Vec           U, *U_g, *U_loc;     // U: solution, R: residual, F: forcing
  Vec           R, R_loc, F, F_loc;  // g: global, loc: local
  Vec           neumann_bcs = NULL, bcs_loc = NULL;
  SNES          snes;
  Mat          *jacob_mat, jacob_mat_coarse, *prolong_restr_mat;
  // PETSc data
  UserMult              res_ctx, jacob_coarse_ctx = NULL, *jacob_ctx;
  FormJacobCtx          form_jacob_ctx;
  UserMultProlongRestr *prolong_restr_ctx;
  PCMGCycleType         pcmg_cycle_type = PC_MG_CYCLE_V;
  // libCEED objects
  Ceed                 ceed;
  CeedData            *ceed_data;
  CeedQFunctionContext ctx_phys, ctx_phys_smoother = NULL;
  // Parameters
  PetscInt  num_comp_u = 3;                  // 3 DoFs in 3D
  PetscInt  num_comp_e = 1, num_comp_d = 5;  // 1 energy output, 5 diagnostic
  PetscInt  num_levels = 1, fine_level = 0;
  PetscInt *U_g_size, *U_l_size, *U_loc_size;
  PetscInt  snes_its = 0, ksp_its = 0;
  double    start_time, elapsed_time, min_time, max_time;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  comm = PETSC_COMM_WORLD;

  // -- Set mesh file, polynomial degree, problem type
  PetscCall(PetscCalloc1(1, &app_ctx));
  PetscCall(ProcessCommandLineOptions(comm, app_ctx));
  PetscCall(PetscCalloc1(1, &problem_functions));
  PetscCall(RegisterProblems(problem_functions));
  num_levels = app_ctx->num_levels;
  fine_level = num_levels - 1;

  // ---------------------------------------------------------------------------
  // Initialize libCEED
  // ---------------------------------------------------------------------------
  // Initialize backend
  CeedInit(app_ctx->ceed_resource, &ceed);

  // Check preferred MemType
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);
  // Setup physics context and wrap in libCEED object
  {
    PetscErrorCode (*SetupPhysics)(MPI_Comm, Ceed, Units *, CeedQFunctionContext *);
    PetscCall(PetscFunctionListFind(problem_functions->setupPhysics, app_ctx->name, &SetupPhysics));
    if (!SetupPhysics) SETERRQ(PETSC_COMM_SELF, 1, "Physics setup for '%s' not found", app_ctx->name);
    PetscCall((*SetupPhysics)(comm, ceed, &units, &ctx_phys));
    PetscErrorCode (*SetupSmootherPhysics)(MPI_Comm, Ceed, CeedQFunctionContext, CeedQFunctionContext *);
    PetscCall(PetscFunctionListFind(problem_functions->setupSmootherPhysics, app_ctx->name, &SetupSmootherPhysics));
    if (!SetupSmootherPhysics) SETERRQ(PETSC_COMM_SELF, 1, "Smoother physics setup for '%s' not found", app_ctx->name);
    PetscCall((*SetupSmootherPhysics)(comm, ceed, ctx_phys, &ctx_phys_smoother));
  }

  // ---------------------------------------------------------------------------
  // Setup DM
  // ---------------------------------------------------------------------------
  // Performance logging
  PetscCall(PetscLogStageRegister("DM and Vector Setup Stage", &stage_dm_setup));
  PetscCall(PetscLogStagePush(stage_dm_setup));

  // -- Create distributed DM from mesh file
  PetscCall(CreateDistributedDM(comm, app_ctx, &dm_orig));
  VecType vectype;
  switch (mem_type_backend) {
    case CEED_MEM_HOST:
      vectype = VECSTANDARD;
      break;
    case CEED_MEM_DEVICE: {
      const char *resolved;
      CeedGetResource(ceed, &resolved);
      if (strstr(resolved, "/gpu/cuda")) vectype = VECCUDA;
      else if (strstr(resolved, "/gpu/hip")) vectype = VECHIP;
      else vectype = VECSTANDARD;
    }
  }
  PetscCall(DMSetVecType(dm_orig, vectype));
  PetscCall(DMPlexDistributeSetDefault(dm_orig, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dm_orig));

  // -- Setup DM by polynomial degree
  PetscCall(PetscMalloc1(num_levels, &level_dms));
  for (PetscInt level = 0; level < num_levels; level++) {
    PetscCall(DMClone(dm_orig, &level_dms[level]));
    PetscCall(DMGetVecType(dm_orig, &vectype));
    PetscCall(DMSetVecType(level_dms[level], vectype));
    PetscCall(SetupDMByDegree(level_dms[level], app_ctx, app_ctx->level_degrees[level], PETSC_TRUE, num_comp_u));
    // -- Label field components for viewing
    // Empty name for conserved field (because there is only one field)
    PetscSection section;
    PetscCall(DMGetLocalSection(level_dms[level], &section));
    PetscCall(PetscSectionSetFieldName(section, 0, "Displacement"));
    PetscCall(PetscSectionSetComponentName(section, 0, 0, "DisplacementX"));
    PetscCall(PetscSectionSetComponentName(section, 0, 1, "DisplacementY"));
    PetscCall(PetscSectionSetComponentName(section, 0, 2, "DisplacementZ"));
  }

  // -- Setup postprocessing DMs
  PetscCall(DMClone(dm_orig, &dm_energy));
  PetscCall(SetupDMByDegree(dm_energy, app_ctx, app_ctx->level_degrees[fine_level], PETSC_FALSE, num_comp_e));
  PetscCall(DMClone(dm_orig, &dm_diagnostic));
  PetscCall(SetupDMByDegree(dm_diagnostic, app_ctx, app_ctx->level_degrees[fine_level], PETSC_FALSE, num_comp_u + num_comp_d));
  PetscCall(DMSetVecType(dm_energy, vectype));
  PetscCall(DMSetVecType(dm_diagnostic, vectype));
  {
    // -- Label field components for viewing
    // Empty name for conserved field (because there is only one field)
    PetscSection section;
    PetscCall(DMGetLocalSection(dm_diagnostic, &section));
    PetscCall(PetscSectionSetFieldName(section, 0, "Diagnostics"));
    PetscCall(PetscSectionSetComponentName(section, 0, 0, "DisplacementX"));
    PetscCall(PetscSectionSetComponentName(section, 0, 1, "DisplacementY"));
    PetscCall(PetscSectionSetComponentName(section, 0, 2, "DisplacementZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, 3, "Pressure"));
    PetscCall(PetscSectionSetComponentName(section, 0, 4, "VolumentricStrain"));
    PetscCall(PetscSectionSetComponentName(section, 0, 5, "TraceE2"));
    PetscCall(PetscSectionSetComponentName(section, 0, 6, "detJ"));
    PetscCall(PetscSectionSetComponentName(section, 0, 7, "StrainEnergyDensity"));
  }

  // ---------------------------------------------------------------------------
  // Setup solution and work vectors
  // ---------------------------------------------------------------------------
  // Allocate arrays
  PetscCall(PetscMalloc1(num_levels, &U_g));
  PetscCall(PetscMalloc1(num_levels, &U_loc));
  PetscCall(PetscMalloc1(num_levels, &U_g_size));
  PetscCall(PetscMalloc1(num_levels, &U_l_size));
  PetscCall(PetscMalloc1(num_levels, &U_loc_size));

  // -- Setup solution vectors for each level
  for (PetscInt level = 0; level < num_levels; level++) {
    // -- Create global unknown vector U
    PetscCall(DMCreateGlobalVector(level_dms[level], &U_g[level]));
    PetscCall(VecGetSize(U_g[level], &U_g_size[level]));
    // Note: Local size for matShell
    PetscCall(VecGetLocalSize(U_g[level], &U_l_size[level]));

    // -- Create local unknown vector U_loc
    PetscCall(DMCreateLocalVector(level_dms[level], &U_loc[level]));
    // Note: local size for libCEED
    PetscCall(VecGetSize(U_loc[level], &U_loc_size[level]));
  }

  // -- Create residual and forcing vectors
  PetscCall(VecDuplicate(U_g[fine_level], &U));
  PetscCall(VecDuplicate(U_g[fine_level], &R));
  PetscCall(VecDuplicate(U_g[fine_level], &F));
  PetscCall(VecDuplicate(U_loc[fine_level], &R_loc));
  PetscCall(VecDuplicate(U_loc[fine_level], &F_loc));

  // Performance logging
  PetscCall(PetscLogStagePop());

  // ---------------------------------------------------------------------------
  // Set up libCEED
  // ---------------------------------------------------------------------------
  // Performance logging
  PetscCall(PetscLogStageRegister("libCEED Setup Stage", &stage_libceed_setup));
  PetscCall(PetscLogStagePush(stage_libceed_setup));

  // -- Create libCEED local forcing vector
  CeedVector   force_ceed;
  CeedScalar  *f;
  PetscMemType force_mem_type;
  if (app_ctx->forcing_choice != FORCE_NONE) {
    PetscCall(VecGetArrayAndMemType(F_loc, &f, &force_mem_type));
    CeedVectorCreate(ceed, U_loc_size[fine_level], &force_ceed);
    CeedVectorSetArray(force_ceed, MemTypeP2C(force_mem_type), CEED_USE_POINTER, f);
  }

  // -- Create libCEED local Neumann BCs vector
  CeedVector   neumann_ceed;
  CeedScalar  *n;
  PetscMemType nummann_mem_type;
  if (app_ctx->bc_traction_count > 0) {
    PetscCall(VecDuplicate(U, &neumann_bcs));
    PetscCall(VecDuplicate(U_loc[fine_level], &bcs_loc));
    PetscCall(VecGetArrayAndMemType(bcs_loc, &n, &nummann_mem_type));
    CeedVectorCreate(ceed, U_loc_size[fine_level], &neumann_ceed);
    CeedVectorSetArray(neumann_ceed, MemTypeP2C(nummann_mem_type), CEED_USE_POINTER, n);
  }

  // -- Setup libCEED objects
  PetscCall(PetscMalloc1(num_levels, &ceed_data));
  // ---- Setup residual, Jacobian evaluator and geometric information
  PetscCall(PetscCalloc1(1, &ceed_data[fine_level]));
  {
    PetscErrorCode (*SetupLibceedFineLevel)(DM, DM, DM, Ceed, AppCtx, CeedQFunctionContext, PetscInt, PetscInt, PetscInt, PetscInt, CeedVector,
                                            CeedVector, CeedData *);
    PetscCall(PetscFunctionListFind(problem_functions->setupLibceedFineLevel, app_ctx->name, &SetupLibceedFineLevel));
    if (!SetupLibceedFineLevel) SETERRQ(PETSC_COMM_SELF, 1, "Fine grid setup for '%s' not found", app_ctx->name);
    PetscCall((*SetupLibceedFineLevel)(level_dms[fine_level], dm_energy, dm_diagnostic, ceed, app_ctx, ctx_phys, fine_level, num_comp_u,
                                       U_g_size[fine_level], U_loc_size[fine_level], force_ceed, neumann_ceed, ceed_data));
  }
  // ---- Setup coarse Jacobian evaluator and prolongation/restriction
  for (PetscInt level = num_levels - 2; level >= 0; level--) {
    PetscCall(PetscCalloc1(1, &ceed_data[level]));

    // Get global communication restriction
    PetscCall(VecZeroEntries(U_g[level + 1]));
    PetscCall(VecSet(U_loc[level + 1], 1.0));
    PetscCall(DMLocalToGlobal(level_dms[level + 1], U_loc[level + 1], ADD_VALUES, U_g[level + 1]));
    PetscCall(DMGlobalToLocal(level_dms[level + 1], U_g[level + 1], INSERT_VALUES, U_loc[level + 1]));

    // Place in libCEED array
    const PetscScalar *m;
    PetscMemType       m_mem_type;
    PetscCall(VecGetArrayReadAndMemType(U_loc[level + 1], &m, &m_mem_type));
    CeedVectorSetArray(ceed_data[level + 1]->x_ceed, MemTypeP2C(m_mem_type), CEED_USE_POINTER, (CeedScalar *)m);

    // Note: use high order ceed, if specified and degree > 4
    PetscErrorCode (*SetupLibceedLevel)(DM, Ceed, AppCtx, PetscInt, PetscInt, PetscInt, PetscInt, CeedVector, CeedData *);
    PetscCall(PetscFunctionListFind(problem_functions->setupLibceedLevel, app_ctx->name, &SetupLibceedLevel));
    if (!SetupLibceedLevel) SETERRQ(PETSC_COMM_SELF, 1, "Coarse grid setup for '%s' not found", app_ctx->name);
    PetscCall((*SetupLibceedLevel)(level_dms[level], ceed, app_ctx, level, num_comp_u, U_g_size[level], U_loc_size[level],
                                   ceed_data[level + 1]->x_ceed, ceed_data));

    // Restore PETSc vector
    CeedVectorTakeArray(ceed_data[level + 1]->x_ceed, MemTypeP2C(m_mem_type), (CeedScalar **)&m);
    PetscCall(VecRestoreArrayReadAndMemType(U_loc[level + 1], &m));
    PetscCall(VecZeroEntries(U_g[level + 1]));
    PetscCall(VecZeroEntries(U_loc[level + 1]));
  }

  // Performance logging
  PetscCall(PetscLogStagePop());

  // ---------------------------------------------------------------------------
  // Setup global forcing and Neumann BC vectors
  // ---------------------------------------------------------------------------
  PetscCall(VecZeroEntries(F));

  if (app_ctx->forcing_choice != FORCE_NONE) {
    CeedVectorTakeArray(force_ceed, MemTypeP2C(force_mem_type), NULL);
    PetscCall(VecRestoreArrayAndMemType(F_loc, &f));
    PetscCall(DMLocalToGlobal(level_dms[fine_level], F_loc, ADD_VALUES, F));
    CeedVectorDestroy(&force_ceed);
  }

  if (app_ctx->bc_traction_count > 0) {
    PetscCall(VecZeroEntries(neumann_bcs));
    CeedVectorTakeArray(neumann_ceed, MemTypeP2C(nummann_mem_type), NULL);
    PetscCall(VecRestoreArrayAndMemType(bcs_loc, &n));
    PetscCall(DMLocalToGlobal(level_dms[fine_level], bcs_loc, ADD_VALUES, neumann_bcs));
    CeedVectorDestroy(&neumann_ceed);
  }

  // ---------------------------------------------------------------------------
  // Print problem summary
  // ---------------------------------------------------------------------------
  if (!app_ctx->test_mode) {
    const char *usedresource;
    CeedGetResource(ceed, &usedresource);
    char hostname[PETSC_MAX_PATH_LEN];
    PetscCall(PetscGetHostName(hostname, sizeof hostname));
    PetscInt comm_size;
    PetscCall(MPI_Comm_size(comm, &comm_size));

    PetscCall(PetscPrintf(comm,
                          "\n-- Elasticity Example - libCEED + PETSc --\n"
                          "  MPI:\n"
                          "    Hostname                           : %s\n"
                          "    Total ranks                        : %d\n"
                          "  libCEED:\n"
                          "    libCEED Backend                    : %s\n"
                          "    libCEED Backend MemType            : %s\n",
                          hostname, comm_size, usedresource, CeedMemTypes[mem_type_backend]));

    VecType vecType;
    PetscCall(VecGetType(U, &vecType));
    PetscCall(PetscPrintf(comm,
                          "  PETSc:\n"
                          "    PETSc Vec Type                     : %s\n",
                          vecType));

    PetscCall(PetscPrintf(comm,
                          "  Problem:\n"
                          "    Problem Name                       : %s\n"
                          "    Forcing Function                   : %s\n"
                          "  Mesh:\n"
                          "    File                               : %s\n"
                          "    Number of 1D Basis Nodes (p)       : %" CeedInt_FMT "\n"
                          "    Number of 1D Quadrature Points (q) : %" CeedInt_FMT "\n"
                          "    Global nodes                       : %" PetscInt_FMT "\n"
                          "    Owned nodes                        : %" PetscInt_FMT "\n"
                          "    DoF per node                       : %" PetscInt_FMT "\n"
                          "  Multigrid:\n"
                          "    Type                               : %s\n"
                          "    Number of Levels                   : %" CeedInt_FMT "\n",
                          app_ctx->name_for_disp, forcing_types_for_disp[app_ctx->forcing_choice],
                          app_ctx->mesh_file[0] ? app_ctx->mesh_file : "Box Mesh", app_ctx->degree + 1, app_ctx->degree + 1,
                          U_g_size[fine_level] / num_comp_u, U_l_size[fine_level] / num_comp_u, num_comp_u,
                          (app_ctx->degree == 1 && app_ctx->multigrid_choice != MULTIGRID_NONE) ? "Algebraic multigrid"
                                                                                                : multigrid_types_for_disp[app_ctx->multigrid_choice],
                          (app_ctx->degree == 1 || app_ctx->multigrid_choice == MULTIGRID_NONE) ? 0 : num_levels));

    if (app_ctx->multigrid_choice != MULTIGRID_NONE) {
      for (PetscInt i = 0; i < 2; i++) {
        CeedInt level = i ? fine_level : 0;
        PetscCall(PetscPrintf(comm,
                              "    Level %" PetscInt_FMT " (%s):\n"
                              "      Number of 1D Basis Nodes (p)     : %" CeedInt_FMT "\n"
                              "      Global Nodes                     : %" PetscInt_FMT "\n"
                              "      Owned Nodes                      : %" PetscInt_FMT "\n",
                              level, i ? "fine" : "coarse", app_ctx->level_degrees[level] + 1, U_g_size[level] / num_comp_u,
                              U_l_size[level] / num_comp_u));
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Setup SNES
  // ---------------------------------------------------------------------------
  // Performance logging
  PetscCall(PetscLogStageRegister("SNES Setup Stage", &stage_snes_setup));
  PetscCall(PetscLogStagePush(stage_snes_setup));

  // Create SNES
  PetscCall(SNESCreate(comm, &snes));
  PetscCall(SNESSetDM(snes, level_dms[fine_level]));

  // -- Jacobian evaluators
  PetscCall(PetscMalloc1(num_levels, &jacob_ctx));
  PetscCall(PetscMalloc1(num_levels, &jacob_mat));
  for (PetscInt level = 0; level < num_levels; level++) {
    // -- Jacobian context for level
    PetscCall(PetscMalloc1(1, &jacob_ctx[level]));
    PetscCall(SetupJacobianCtx(comm, app_ctx, level_dms[level], U_g[level], U_loc[level], ceed_data[level], ceed, ctx_phys, ctx_phys_smoother,
                               jacob_ctx[level]));

    // -- Form Action of Jacobian on delta_u
    PetscCall(MatCreateShell(comm, U_l_size[level], U_l_size[level], U_g_size[level], U_g_size[level], jacob_ctx[level], &jacob_mat[level]));
    PetscCall(MatShellSetOperation(jacob_mat[level], MATOP_MULT, (void (*)(void))ApplyJacobian_Ceed));
    PetscCall(MatShellSetOperation(jacob_mat[level], MATOP_GET_DIAGONAL, (void (*)(void))GetDiag_Ceed));
    PetscCall(MatShellSetVecType(jacob_mat[level], vectype));
  }
  // Note: FormJacobian updates Jacobian matrices on each level
  //   and assembles the Jpre matrix, if needed
  PetscCall(PetscMalloc1(1, &form_jacob_ctx));
  form_jacob_ctx->jacob_ctx  = jacob_ctx;
  form_jacob_ctx->num_levels = num_levels;
  form_jacob_ctx->jacob_mat  = jacob_mat;

  // -- Residual evaluation function
  PetscCall(PetscCalloc1(1, &res_ctx));
  PetscCall(PetscMemcpy(res_ctx, jacob_ctx[fine_level], sizeof(*jacob_ctx[fine_level])));
  res_ctx->op = ceed_data[fine_level]->op_residual;
  res_ctx->qf = ceed_data[fine_level]->qf_residual;
  if (app_ctx->bc_traction_count > 0) res_ctx->neumann_bcs = neumann_bcs;
  else res_ctx->neumann_bcs = NULL;
  PetscCall(SNESSetFunction(snes, R, FormResidual_Ceed, res_ctx));

  // -- Prolongation/Restriction evaluation
  PetscCall(PetscMalloc1(num_levels, &prolong_restr_ctx));
  PetscCall(PetscMalloc1(num_levels, &prolong_restr_mat));
  for (PetscInt level = 1; level < num_levels; level++) {
    // ---- Prolongation/restriction context for level
    PetscCall(PetscMalloc1(1, &prolong_restr_ctx[level]));
    PetscCall(SetupProlongRestrictCtx(comm, app_ctx, level_dms[level - 1], level_dms[level], U_g[level], U_loc[level - 1], U_loc[level],
                                      ceed_data[level - 1], ceed_data[level], ceed, prolong_restr_ctx[level]));

    // ---- Form Action of Jacobian on delta_u
    PetscCall(MatCreateShell(comm, U_l_size[level], U_l_size[level - 1], U_g_size[level], U_g_size[level - 1], prolong_restr_ctx[level],
                             &prolong_restr_mat[level]));
    // Note: In PCMG, restriction is the transpose of prolongation
    PetscCall(MatShellSetOperation(prolong_restr_mat[level], MATOP_MULT, (void (*)(void))Prolong_Ceed));
    PetscCall(MatShellSetOperation(prolong_restr_mat[level], MATOP_MULT_TRANSPOSE, (void (*)(void))Restrict_Ceed));
    PetscCall(MatShellSetVecType(prolong_restr_mat[level], vectype));
  }

  // ---------------------------------------------------------------------------
  // Setup for AMG coarse solve
  // ---------------------------------------------------------------------------
  if (app_ctx->multigrid_choice != MULTIGRID_NONE) {
    // -- Jacobian Matrix
    PetscCall(DMCreateMatrix(level_dms[0], &jacob_mat_coarse));

    if (app_ctx->degree > 1) {
      // -- Assemble sparsity pattern
      PetscCount num_entries;
      CeedInt   *rows, *cols;
      CeedVector coo_values;
      CeedOperatorLinearAssembleSymbolic(ceed_data[0]->op_jacobian, &num_entries, &rows, &cols);
      ISLocalToGlobalMapping ltog_row, ltog_col;
      PetscCall(MatGetLocalToGlobalMapping(jacob_mat_coarse, &ltog_row, &ltog_col));
      PetscCall(ISLocalToGlobalMappingApply(ltog_row, num_entries, rows, rows));
      PetscCall(ISLocalToGlobalMappingApply(ltog_col, num_entries, cols, cols));
      PetscCall(MatSetPreallocationCOO(jacob_mat_coarse, num_entries, rows, cols));
      free(rows);
      free(cols);
      CeedVectorCreate(ceed, num_entries, &coo_values);

      // -- Update form_jacob_ctx
      form_jacob_ctx->coo_values       = coo_values;
      form_jacob_ctx->op_coarse        = ceed_data[0]->op_jacobian;
      form_jacob_ctx->jacob_mat_coarse = jacob_mat_coarse;
    }
  }

  // Set Jacobian function
  if (app_ctx->degree > 1) {
    PetscCall(SNESSetJacobian(snes, jacob_mat[fine_level], jacob_mat[fine_level], FormJacobian, form_jacob_ctx));
  } else {
    PetscCall(SNESSetJacobian(snes, jacob_mat[0], jacob_mat_coarse, SNESComputeJacobianDefaultColor, NULL));
  }

  // ---------------------------------------------------------------------------
  // Setup KSP
  // ---------------------------------------------------------------------------
  {
    PC  pc;
    KSP ksp;

    // -- KSP
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(KSPSetOptionsPrefix(ksp, "outer_"));

    // -- Preconditioning
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetDM(pc, level_dms[fine_level]));
    PetscCall(PCSetOptionsPrefix(pc, "outer_"));

    if (app_ctx->multigrid_choice == MULTIGRID_NONE) {
      // ---- No Multigrid
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_DIAGONAL));
    } else if (app_ctx->degree == 1) {
      // ---- AMG for degree 1
      PetscCall(PCSetType(pc, PCGAMG));
    } else {
      // ---- PCMG
      PetscCall(PCSetType(pc, PCMG));

      // ------ PCMG levels
      PetscCall(PCMGSetLevels(pc, num_levels, NULL));
      for (PetscInt level = 0; level < num_levels; level++) {
        // -------- Smoother
        KSP ksp_smoother, ksp_est;
        PC  pc_smoother;

        // ---------- Smoother KSP
        PetscCall(PCMGGetSmoother(pc, level, &ksp_smoother));
        PetscCall(KSPSetDM(ksp_smoother, level_dms[level]));
        PetscCall(KSPSetDMActive(ksp_smoother, PETSC_FALSE));

        // ---------- Chebyshev options
        PetscCall(KSPSetType(ksp_smoother, KSPCHEBYSHEV));
        PetscCall(KSPChebyshevEstEigSet(ksp_smoother, 0, 0.1, 0, 1.1));
        PetscCall(KSPChebyshevEstEigGetKSP(ksp_smoother, &ksp_est));
        PetscCall(KSPSetType(ksp_est, KSPCG));
        PetscCall(KSPChebyshevEstEigSetUseNoisy(ksp_smoother, PETSC_TRUE));
        PetscCall(KSPSetOperators(ksp_smoother, jacob_mat[level], jacob_mat[level]));

        // ---------- Smoother preconditioner
        PetscCall(KSPGetPC(ksp_smoother, &pc_smoother));
        PetscCall(PCSetType(pc_smoother, PCJACOBI));
        PetscCall(PCJacobiSetType(pc_smoother, PC_JACOBI_DIAGONAL));

        // -------- Work vector
        if (level != fine_level) {
          PetscCall(PCMGSetX(pc, level, U_g[level]));
        }

        // -------- Level prolongation/restriction operator
        if (level > 0) {
          PetscCall(PCMGSetInterpolation(pc, level, prolong_restr_mat[level]));
          PetscCall(PCMGSetRestriction(pc, level, prolong_restr_mat[level]));
        }
      }

      // ------ PCMG coarse solve
      KSP ksp_coarse;
      PC  pc_coarse;

      // -------- Coarse KSP
      PetscCall(PCMGGetCoarseSolve(pc, &ksp_coarse));
      PetscCall(KSPSetType(ksp_coarse, KSPPREONLY));
      PetscCall(KSPSetOperators(ksp_coarse, jacob_mat_coarse, jacob_mat_coarse));
      PetscCall(KSPSetOptionsPrefix(ksp_coarse, "coarse_"));

      // -------- Coarse preconditioner
      PetscCall(KSPGetPC(ksp_coarse, &pc_coarse));
      PetscCall(PCSetType(pc_coarse, PCGAMG));
      PetscCall(PCSetOptionsPrefix(pc_coarse, "coarse_"));

      PetscCall(KSPSetFromOptions(ksp_coarse));
      PetscCall(PCSetFromOptions(pc_coarse));

      // ------ PCMG options
      PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
      PetscCall(PCMGSetNumberSmooth(pc, 3));
      PetscCall(PCMGSetCycleType(pc, pcmg_cycle_type));
    }
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(PCSetFromOptions(pc));
  }
  {
    // Default to critical-point (CP) line search (related to Wolfe's curvature condition)
    SNESLineSearch line_search;

    PetscCall(SNESGetLineSearch(snes, &line_search));
    PetscCall(SNESLineSearchSetType(line_search, SNESLINESEARCHCP));
  }

  PetscCall(SNESSetFromOptions(snes));

  // Performance logging
  PetscCall(PetscLogStagePop());

  // ---------------------------------------------------------------------------
  // Set initial guess
  // ---------------------------------------------------------------------------
  PetscCall(PetscObjectSetName((PetscObject)U, ""));
  PetscCall(VecSet(U, 0.0));

  // View solution
  if (app_ctx->view_soln) {
    PetscCall(ViewSolution(comm, app_ctx, U, 0, 0.0));
  }

  // ---------------------------------------------------------------------------
  // Solve SNES
  // ---------------------------------------------------------------------------
  PetscBool snes_monitor = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-snes_monitor", &snes_monitor));

  // Performance logging
  PetscCall(PetscLogStageRegister("SNES Solve Stage", &stage_snes_solve));
  PetscCall(PetscLogStagePush(stage_snes_solve));

  // Timing
  PetscCall(PetscBarrier((PetscObject)snes));
  start_time = MPI_Wtime();

  // Solve for each load increment
  PetscInt increment;
  for (increment = 1; increment <= app_ctx->num_increments; increment++) {
    // -- Log increment count
    if (snes_monitor) {
      PetscCall(PetscPrintf(comm, "%" PetscInt_FMT " Load Increment\n", increment - 1));
    }

    // -- Scale the problem
    PetscScalar load_increment = 1.0 * increment / app_ctx->num_increments,
                scalingFactor  = load_increment / (increment == 1 ? 1 : res_ctx->load_increment);
    res_ctx->load_increment    = load_increment;
    if (app_ctx->num_increments > 1 && app_ctx->forcing_choice != FORCE_NONE) {
      PetscCall(VecScale(F, scalingFactor));
    }

    // -- Solve
    PetscCall(SNESSolve(snes, F, U));

    // -- View solution
    if (app_ctx->view_soln) {
      PetscCall(ViewSolution(comm, app_ctx, U, increment, load_increment));
    }

    // -- Update SNES iteration count
    PetscInt its;
    PetscCall(SNESGetIterationNumber(snes, &its));
    snes_its += its;
    PetscCall(SNESGetLinearSolveIterations(snes, &its));
    ksp_its += its;

    // -- Check for divergence
    SNESConvergedReason reason;
    PetscCall(SNESGetConvergedReason(snes, &reason));
    if (reason < 0) break;
    if (app_ctx->energy_viewer) {
      // -- Log strain energy for current load increment
      CeedScalar energy;
      PetscCall(ComputeStrainEnergy(dm_energy, res_ctx, ceed_data[fine_level]->op_energy, U, &energy));

      if (!app_ctx->test_mode) {
        // -- Output
        PetscCall(PetscPrintf(comm, "    Strain Energy                      : %.12e\n", energy));
      }
      PetscCall(PetscViewerASCIIPrintf(app_ctx->energy_viewer, "%f,%e\n", load_increment, energy));
    }
  }

  // Timing
  elapsed_time = MPI_Wtime() - start_time;

  // Performance logging
  PetscCall(PetscLogStagePop());

  // ---------------------------------------------------------------------------
  // Output summary
  // ---------------------------------------------------------------------------
  if (!app_ctx->test_mode) {
    // -- SNES
    SNESType            snes_type;
    SNESConvergedReason reason;
    PetscReal           rnorm;
    PetscCall(SNESGetType(snes, &snes_type));
    PetscCall(SNESGetConvergedReason(snes, &reason));
    PetscCall(SNESGetFunctionNorm(snes, &rnorm));
    PetscCall(PetscPrintf(comm,
                          "  SNES:\n"
                          "    SNES Type                          : %s\n"
                          "    SNES Convergence                   : %s\n"
                          "    Number of Load Increments          : %" PetscInt_FMT "\n"
                          "    Completed Load Increments          : %" PetscInt_FMT "\n"
                          "    Total SNES Iterations              : %" PetscInt_FMT "\n"
                          "    Final rnorm                        : %e\n",
                          snes_type, SNESConvergedReasons[reason], app_ctx->num_increments, increment - 1, snes_its, (double)rnorm));

    // -- KSP
    KSP     ksp;
    KSPType ksp_type;
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetType(ksp, &ksp_type));
    PetscCall(PetscPrintf(comm,
                          "  Linear Solver:\n"
                          "    KSP Type                           : %s\n"
                          "    Total KSP Iterations               : %" PetscInt_FMT "\n",
                          ksp_type, ksp_its));

    // -- PC
    PC     pc;
    PCType pc_type;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCGetType(pc, &pc_type));
    PetscCall(PetscPrintf(comm, "    PC Type                            : %s\n", pc_type));

    if (!strcmp(pc_type, PCMG)) {
      PCMGType pcmg_type;
      PetscCall(PCMGGetType(pc, &pcmg_type));
      PetscCall(PetscPrintf(comm,
                            "  P-Multigrid:\n"
                            "    PCMG Type                          : %s\n"
                            "    PCMG Cycle Type                    : %s\n",
                            PCMGTypes[pcmg_type], PCMGCycleTypes[pcmg_cycle_type]));

      // -- Coarse Solve
      KSP    ksp_coarse;
      PC     pc_coarse;
      PCType pc_type;

      PetscCall(PCMGGetCoarseSolve(pc, &ksp_coarse));
      PetscCall(KSPGetType(ksp_coarse, &ksp_type));
      PetscCall(KSPGetPC(ksp_coarse, &pc_coarse));
      PetscCall(PCGetType(pc_coarse, &pc_type));
      PetscCall(PetscPrintf(comm,
                            "    Coarse Solve:\n"
                            "      KSP Type                         : %s\n"
                            "      PC Type                          : %s\n",
                            ksp_type, pc_type));
    }
  }

  // ---------------------------------------------------------------------------
  // Compute solve time
  // ---------------------------------------------------------------------------
  if (!app_ctx->test_mode) {
    PetscCall(MPI_Allreduce(&elapsed_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, comm));
    PetscCall(MPI_Allreduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, comm));
    PetscCall(PetscPrintf(comm,
                          "  Performance:\n"
                          "    SNES Solve Time                    : %g (%g) sec\n"
                          "    DoFs/Sec in SNES                   : %g (%g) million\n",
                          max_time, min_time, 1e-6 * U_g_size[fine_level] * ksp_its / max_time, 1e-6 * U_g_size[fine_level] * ksp_its / min_time));
  }

  // ---------------------------------------------------------------------------
  // Compute error
  // ---------------------------------------------------------------------------
  if (app_ctx->forcing_choice == FORCE_MMS) {
    CeedScalar        l2_error = 1., l2_U_norm = 1.;
    const CeedScalar *true_array;
    Vec               error_vec, true_vec;

    // -- Work vectors
    PetscCall(VecDuplicate(U, &error_vec));
    PetscCall(VecSet(error_vec, 0.0));
    PetscCall(VecDuplicate(U, &true_vec));
    PetscCall(VecSet(true_vec, 0.0));

    // -- Assemble global true solution vector
    CeedVectorGetArrayRead(ceed_data[fine_level]->true_soln, CEED_MEM_HOST, &true_array);
    PetscCall(VecPlaceArray(res_ctx->Y_loc, (PetscScalar *)true_array));
    PetscCall(DMLocalToGlobal(res_ctx->dm, res_ctx->Y_loc, INSERT_VALUES, true_vec));
    PetscCall(VecResetArray(res_ctx->Y_loc));
    CeedVectorRestoreArrayRead(ceed_data[fine_level]->true_soln, &true_array);

    // -- Compute L2 error
    PetscCall(VecWAXPY(error_vec, -1.0, U, true_vec));
    PetscCall(VecNorm(error_vec, NORM_2, &l2_error));
    PetscCall(VecNorm(U, NORM_2, &l2_U_norm));
    l2_error /= l2_U_norm;

    // -- Output
    if (!app_ctx->test_mode || l2_error > 0.05) {
      PetscCall(PetscPrintf(comm, "    L2 Error                           : %e\n", l2_error));
    }

    // -- Cleanup
    PetscCall(VecDestroy(&error_vec));
    PetscCall(VecDestroy(&true_vec));
  }

  // ---------------------------------------------------------------------------
  // Compute energy
  // ---------------------------------------------------------------------------
  PetscReal energy;
  PetscCall(ComputeStrainEnergy(dm_energy, res_ctx, ceed_data[fine_level]->op_energy, U, &energy));
  if (!app_ctx->test_mode) {
    // -- Output
    PetscCall(PetscPrintf(comm, "    Strain Energy                      : %.12e\n", energy));
  }
  PetscCall(RegressionTests_solids(app_ctx, energy));

  // ---------------------------------------------------------------------------
  // Output diagnostic quantities
  // ---------------------------------------------------------------------------
  if (app_ctx->view_soln || app_ctx->view_final_soln) {
    // -- Setup context
    UserMult diagnostic_ctx;
    PetscCall(PetscMalloc1(1, &diagnostic_ctx));
    PetscCall(PetscMemcpy(diagnostic_ctx, res_ctx, sizeof(*res_ctx)));
    diagnostic_ctx->dm = dm_diagnostic;
    diagnostic_ctx->op = ceed_data[fine_level]->op_diagnostic;

    // -- Compute and output
    PetscCall(ViewDiagnosticQuantities(comm, level_dms[fine_level], diagnostic_ctx, app_ctx, U, ceed_data[fine_level]->elem_restr_diagnostic));

    // -- Cleanup
    PetscCall(PetscFree(diagnostic_ctx));
  }

  // ---------------------------------------------------------------------------
  // Free objects
  // ---------------------------------------------------------------------------
  // Data in arrays per level
  for (PetscInt level = 0; level < num_levels; level++) {
    // Vectors
    PetscCall(VecDestroy(&U_g[level]));
    PetscCall(VecDestroy(&U_loc[level]));

    // Jacobian matrix and data
    PetscCall(VecDestroy(&jacob_ctx[level]->Y_loc));
    PetscCall(MatDestroy(&jacob_mat[level]));
    PetscCall(PetscFree(jacob_ctx[level]));

    // Prolongation/Restriction matrix and data
    if (level > 0) {
      PetscCall(PetscFree(prolong_restr_ctx[level]));
      PetscCall(MatDestroy(&prolong_restr_mat[level]));
    }

    // DM
    PetscCall(DMDestroy(&level_dms[level]));

    // libCEED objects
    PetscCall(CeedDataDestroy(level, ceed_data[level]));
  }

  PetscCall(PetscViewerDestroy(&app_ctx->energy_viewer));

  // Arrays
  PetscCall(PetscFree(U_g));
  PetscCall(PetscFree(U_loc));
  PetscCall(PetscFree(U_g_size));
  PetscCall(PetscFree(U_l_size));
  PetscCall(PetscFree(U_loc_size));
  PetscCall(PetscFree(jacob_ctx));
  PetscCall(PetscFree(jacob_mat));
  PetscCall(PetscFree(prolong_restr_ctx));
  PetscCall(PetscFree(prolong_restr_mat));
  PetscCall(PetscFree(app_ctx->level_degrees));
  PetscCall(PetscFree(ceed_data));

  // libCEED objects
  CeedVectorDestroy(&form_jacob_ctx->coo_values);
  CeedQFunctionContextDestroy(&ctx_phys);
  CeedQFunctionContextDestroy(&ctx_phys_smoother);
  CeedDestroy(&ceed);

  // PETSc objects
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&R));
  PetscCall(VecDestroy(&R_loc));
  PetscCall(VecDestroy(&F));
  PetscCall(VecDestroy(&F_loc));
  PetscCall(VecDestroy(&neumann_bcs));
  PetscCall(VecDestroy(&bcs_loc));
  PetscCall(MatDestroy(&jacob_mat_coarse));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm_orig));
  PetscCall(DMDestroy(&dm_energy));
  PetscCall(DMDestroy(&dm_diagnostic));
  PetscCall(PetscFree(level_dms));

  // -- Function list
  PetscCall(PetscFunctionListDestroy(&problem_functions->setupPhysics));
  PetscCall(PetscFunctionListDestroy(&problem_functions->setupSmootherPhysics));
  PetscCall(PetscFunctionListDestroy(&problem_functions->setupLibceedFineLevel));
  PetscCall(PetscFunctionListDestroy(&problem_functions->setupLibceedLevel));

  // Structs
  PetscCall(PetscFree(res_ctx));
  PetscCall(PetscFree(form_jacob_ctx));
  PetscCall(PetscFree(jacob_coarse_ctx));
  PetscCall(PetscFree(app_ctx));
  PetscCall(PetscFree(problem_functions));
  PetscCall(PetscFree(units));

  return PetscFinalize();
}
