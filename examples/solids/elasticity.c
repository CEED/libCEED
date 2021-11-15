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
//     ./elasticity -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -bc_clamp 998,999 -bc_clamp_998_translate 0.1,0.2,0.3 -problem SS-NH -forcing none -ceed /cpu/self
//     ./elasticity -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -bc_clamp 998,999 -bc_clamp_998_rotate 1,0,0,0.2 -problem FSInitial-NH1 -forcing none -ceed /gpu/cuda
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
  PetscInt       ierr;
  MPI_Comm       comm;
  // Context structs
  AppCtx         app_ctx;                  // Contains problem options
  ProblemFunctions problem_functions;      // Setup functions for each problem
  Units          units;                    // Contains units scaling
  // PETSc objects
  PetscLogStage  stage_dm_setup, stage_libceed_setup,
                 stage_snes_setup, stage_snes_solve;
  DM             dm_orig;                  // Distributed DM to clone
  DM             dm_energy, dm_diagnostic; // DMs for postprocessing
  DM             *level_dms;
  Vec            U, *U_g, *U_loc;          // U: solution, R: residual, F: forcing
  Vec            R, R_loc, F, F_loc;       // g: global, loc: local
  Vec            neumann_bcs = NULL, bcs_loc = NULL;
  SNES           snes;
  Mat            *jacob_mat, jacob_mat_coarse, *prolong_restr_mat;
  // PETSc data
  UserMult       res_ctx, jacob_coarse_ctx = NULL, *jacob_ctx;
  FormJacobCtx   form_jacob_ctx;
  UserMultProlongRestr *prolong_restr_ctx;
  PCMGCycleType  pcmg_cycle_type = PC_MG_CYCLE_V;
  // libCEED objects
  Ceed           ceed;
  CeedData       *ceed_data;
  CeedQFunctionContext ctx_phys, ctx_phys_smoother = NULL;
  // Parameters
  PetscInt       num_comp_u = 3;                 // 3 DoFs in 3D
  PetscInt       num_comp_e = 1, num_comp_d = 5; // 1 energy output, 5 diagnostic
  PetscInt       num_levels = 1, fine_level = 0;
  PetscInt       *U_g_size, *U_l_size, *U_loc_size;
  PetscInt       snes_its = 0, ksp_its = 0;
  double         start_time, elapsed_time, min_time, max_time;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  comm = PETSC_COMM_WORLD;

  // -- Set mesh file, polynomial degree, problem type
  ierr = PetscCalloc1(1, &app_ctx); CHKERRQ(ierr);
  ierr = ProcessCommandLineOptions(comm, app_ctx); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &problem_functions); CHKERRQ(ierr);
  ierr = RegisterProblems(problem_functions); CHKERRQ(ierr);
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
    ierr = PetscFunctionListFind(problem_functions->setupPhysics, app_ctx->name,
                                 &SetupPhysics); CHKERRQ(ierr);
    if (!SetupPhysics)
      SETERRQ(PETSC_COMM_SELF, 1, "Physics setup for '%s' not found",
              app_ctx->name);
    ierr = (*SetupPhysics)(comm, ceed, &units, &ctx_phys); CHKERRQ(ierr);
    PetscErrorCode (*SetupSmootherPhysics)(MPI_Comm, Ceed, CeedQFunctionContext,
                                           CeedQFunctionContext *);
    ierr = PetscFunctionListFind(problem_functions->setupSmootherPhysics,
                                 app_ctx->name, &SetupSmootherPhysics);
    CHKERRQ(ierr);
    if (!SetupSmootherPhysics)
      SETERRQ(PETSC_COMM_SELF, 1, "Smoother physics setup for '%s' not found",
              app_ctx->name);
    ierr = (*SetupSmootherPhysics)(comm, ceed, ctx_phys, &ctx_phys_smoother);
    CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Setup DM
  // ---------------------------------------------------------------------------
  // Performance logging
  ierr = PetscLogStageRegister("DM and Vector Setup Stage", &stage_dm_setup);
  CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage_dm_setup); CHKERRQ(ierr);

  // -- Create distributed DM from mesh file
  ierr = CreateDistributedDM(comm, app_ctx, &dm_orig); CHKERRQ(ierr);
  VecType vectype;
  switch (mem_type_backend) {
  case CEED_MEM_HOST: vectype = VECSTANDARD; break;
  case CEED_MEM_DEVICE: {
    const char *resolved;
    CeedGetResource(ceed, &resolved);
    if (strstr(resolved, "/gpu/cuda")) vectype = VECCUDA;
    else if (strstr(resolved, "/gpu/hip")) vectype = VECHIP;
    else vectype = VECSTANDARD;
  }
  }
  ierr = DMSetVecType(dm_orig, vectype); CHKERRQ(ierr);
  ierr = DMPlexDistributeSetDefault(dm_orig, PETSC_FALSE); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm_orig); CHKERRQ(ierr);

  // -- Setup DM by polynomial degree
  ierr = PetscMalloc1(num_levels, &level_dms); CHKERRQ(ierr);
  for (PetscInt level = 0; level < num_levels; level++) {
    ierr = DMClone(dm_orig, &level_dms[level]); CHKERRQ(ierr);
    ierr = DMGetVecType(dm_orig, &vectype); CHKERRQ(ierr);
    ierr = DMSetVecType(level_dms[level], vectype); CHKERRQ(ierr);
    ierr = SetupDMByDegree(level_dms[level], app_ctx, app_ctx->level_degrees[level],
                           PETSC_TRUE, num_comp_u); CHKERRQ(ierr);
    // -- Label field components for viewing
    // Empty name for conserved field (because there is only one field)
    PetscSection section;
    ierr = DMGetLocalSection(level_dms[level], &section); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, 0, "Displacement"); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 0, "DisplacementX");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 1, "DisplacementY");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 2, "DisplacementZ");
    CHKERRQ(ierr);
  }

  // -- Setup postprocessing DMs
  ierr = DMClone(dm_orig, &dm_energy); CHKERRQ(ierr);
  ierr = SetupDMByDegree(dm_energy, app_ctx, app_ctx->level_degrees[fine_level],
                         PETSC_FALSE, num_comp_e); CHKERRQ(ierr);
  ierr = DMClone(dm_orig, &dm_diagnostic); CHKERRQ(ierr);
  ierr = SetupDMByDegree(dm_diagnostic, app_ctx,
                         app_ctx->level_degrees[fine_level],
                         PETSC_FALSE, num_comp_u + num_comp_d); CHKERRQ(ierr);
  ierr = DMSetVecType(dm_energy, vectype); CHKERRQ(ierr);
  ierr = DMSetVecType(dm_diagnostic, vectype); CHKERRQ(ierr);
  {
    // -- Label field components for viewing
    // Empty name for conserved field (because there is only one field)
    PetscSection section;
    ierr = DMGetLocalSection(dm_diagnostic, &section); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, 0, "Diagnostics"); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 0, "DisplacementX");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 1, "DisplacementY");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 2, "DisplacementZ");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 3, "Pressure");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 4, "VolumentricStrain");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 5, "TraceE2");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 6, "detJ");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 7, "StrainEnergyDensity");
    CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Setup solution and work vectors
  // ---------------------------------------------------------------------------
  // Allocate arrays
  ierr = PetscMalloc1(num_levels, &U_g); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &U_loc); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &U_g_size); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &U_l_size); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &U_loc_size); CHKERRQ(ierr);

  // -- Setup solution vectors for each level
  for (PetscInt level = 0; level < num_levels; level++) {
    // -- Create global unknown vector U
    ierr = DMCreateGlobalVector(level_dms[level], &U_g[level]); CHKERRQ(ierr);
    ierr = VecGetSize(U_g[level], &U_g_size[level]); CHKERRQ(ierr);
    // Note: Local size for matShell
    ierr = VecGetLocalSize(U_g[level], &U_l_size[level]); CHKERRQ(ierr);

    // -- Create local unknown vector U_loc
    ierr = DMCreateLocalVector(level_dms[level], &U_loc[level]); CHKERRQ(ierr);
    // Note: local size for libCEED
    ierr = VecGetSize(U_loc[level], &U_loc_size[level]); CHKERRQ(ierr);
  }

  // -- Create residual and forcing vectors
  ierr = VecDuplicate(U_g[fine_level], &U); CHKERRQ(ierr);
  ierr = VecDuplicate(U_g[fine_level], &R); CHKERRQ(ierr);
  ierr = VecDuplicate(U_g[fine_level], &F); CHKERRQ(ierr);
  ierr = VecDuplicate(U_loc[fine_level], &R_loc); CHKERRQ(ierr);
  ierr = VecDuplicate(U_loc[fine_level], &F_loc); CHKERRQ(ierr);

  // Performance logging
  ierr = PetscLogStagePop();

  // ---------------------------------------------------------------------------
  // Set up libCEED
  // ---------------------------------------------------------------------------
  // Performance logging
  ierr = PetscLogStageRegister("libCEED Setup Stage", &stage_libceed_setup);
  CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage_libceed_setup); CHKERRQ(ierr);

  // -- Create libCEED local forcing vector
  CeedVector force_ceed;
  CeedScalar *f;
  PetscMemType force_mem_type;
  if (app_ctx->forcing_choice != FORCE_NONE) {
    ierr = VecGetArrayAndMemType(F_loc, &f, &force_mem_type); CHKERRQ(ierr);
    CeedVectorCreate(ceed, U_loc_size[fine_level], &force_ceed);
    CeedVectorSetArray(force_ceed, MemTypeP2C(force_mem_type), CEED_USE_POINTER, f);
  }

  // -- Create libCEED local Neumann BCs vector
  CeedVector neumann_ceed;
  CeedScalar *n;
  PetscMemType nummann_mem_type;
  if (app_ctx->bc_traction_count > 0) {
    ierr = VecDuplicate(U, &neumann_bcs); CHKERRQ(ierr);
    ierr = VecDuplicate(U_loc[fine_level], &bcs_loc); CHKERRQ(ierr);
    ierr = VecGetArrayAndMemType(bcs_loc, &n, &nummann_mem_type); CHKERRQ(ierr);
    CeedVectorCreate(ceed, U_loc_size[fine_level], &neumann_ceed);
    CeedVectorSetArray(neumann_ceed, MemTypeP2C(nummann_mem_type),
                       CEED_USE_POINTER, n);
  }

  // -- Setup libCEED objects
  ierr = PetscMalloc1(num_levels, &ceed_data); CHKERRQ(ierr);
  // ---- Setup residual, Jacobian evaluator and geometric information
  ierr = PetscCalloc1(1, &ceed_data[fine_level]); CHKERRQ(ierr);
  {
    PetscErrorCode (*SetupLibceedFineLevel)(DM, DM, DM, Ceed, AppCtx,
                                            CeedQFunctionContext, PetscInt,
                                            PetscInt, PetscInt, PetscInt,
                                            CeedVector, CeedVector, CeedData *);
    ierr = PetscFunctionListFind(problem_functions->setupLibceedFineLevel,
                                 app_ctx->name, &SetupLibceedFineLevel);
    CHKERRQ(ierr);
    if (!SetupLibceedFineLevel)
      SETERRQ(PETSC_COMM_SELF, 1, "Fine grid setup for '%s' not found",
              app_ctx->name);
    ierr = (*SetupLibceedFineLevel)(level_dms[fine_level], dm_energy, dm_diagnostic,
                                    ceed, app_ctx, ctx_phys, fine_level,
                                    num_comp_u, U_g_size[fine_level],
                                    U_loc_size[fine_level],
                                    force_ceed, neumann_ceed, ceed_data);
    CHKERRQ(ierr);
  }
  // ---- Setup coarse Jacobian evaluator and prolongation/restriction
  for (PetscInt level = num_levels - 2; level >= 0; level--) {
    ierr = PetscCalloc1(1, &ceed_data[level]); CHKERRQ(ierr);

    // Get global communication restriction
    ierr = VecZeroEntries(U_g[level+1]); CHKERRQ(ierr);
    ierr = VecSet(U_loc[level+1], 1.0); CHKERRQ(ierr);
    ierr = DMLocalToGlobal(level_dms[level+1], U_loc[level+1], ADD_VALUES,
                           U_g[level+1]); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(level_dms[level+1], U_g[level+1], INSERT_VALUES,
                           U_loc[level+1]); CHKERRQ(ierr);

    // Place in libCEED array
    const PetscScalar *m;
    PetscMemType m_mem_type;
    ierr = VecGetArrayReadAndMemType(U_loc[level+1], &m, &m_mem_type);
    CHKERRQ(ierr);
    CeedVectorSetArray(ceed_data[level+1]->x_ceed, MemTypeP2C(m_mem_type),
                       CEED_USE_POINTER, (CeedScalar *)m);

    // Note: use high order ceed, if specified and degree > 4
    PetscErrorCode (*SetupLibceedLevel)(DM, Ceed, AppCtx, PetscInt,
                                        PetscInt, PetscInt, PetscInt, CeedVector, CeedData *);
    ierr = PetscFunctionListFind(problem_functions->setupLibceedLevel,
                                 app_ctx->name, &SetupLibceedLevel);
    CHKERRQ(ierr);
    if (!SetupLibceedLevel)
      SETERRQ(PETSC_COMM_SELF, 1, "Coarse grid setup for '%s' not found",
              app_ctx->name);
    ierr = (*SetupLibceedLevel)(level_dms[level], ceed, app_ctx,
                                level, num_comp_u, U_g_size[level],
                                U_loc_size[level], ceed_data[level+1]->x_ceed,
                                ceed_data);
    CHKERRQ(ierr);

    // Restore PETSc vector
    CeedVectorTakeArray(ceed_data[level+1]->x_ceed, MemTypeP2C(m_mem_type),
                        (CeedScalar **)&m);
    ierr = VecRestoreArrayReadAndMemType(U_loc[level+1], &m); CHKERRQ(ierr);
    ierr = VecZeroEntries(U_g[level+1]); CHKERRQ(ierr);
    ierr = VecZeroEntries(U_loc[level+1]); CHKERRQ(ierr);
  }

  // Performance logging
  ierr = PetscLogStagePop();

  // ---------------------------------------------------------------------------
  // Setup global forcing and Neumann BC vectors
  // ---------------------------------------------------------------------------
  ierr = VecZeroEntries(F); CHKERRQ(ierr);

  if (app_ctx->forcing_choice != FORCE_NONE) {
    CeedVectorTakeArray(force_ceed, MemTypeP2C(force_mem_type), NULL);
    ierr = VecRestoreArrayAndMemType(F_loc, &f); CHKERRQ(ierr);
    ierr = DMLocalToGlobal(level_dms[fine_level], F_loc, ADD_VALUES, F);
    CHKERRQ(ierr);
    CeedVectorDestroy(&force_ceed);
  }

  if (app_ctx->bc_traction_count > 0) {
    ierr = VecZeroEntries(neumann_bcs); CHKERRQ(ierr);
    CeedVectorTakeArray(neumann_ceed, MemTypeP2C(nummann_mem_type), NULL);
    ierr = VecRestoreArrayAndMemType(bcs_loc, &n); CHKERRQ(ierr);
    ierr = DMLocalToGlobal(level_dms[fine_level], bcs_loc, ADD_VALUES, neumann_bcs);
    CHKERRQ(ierr);
    CeedVectorDestroy(&neumann_ceed);
  }

  // ---------------------------------------------------------------------------
  // Print problem summary
  // ---------------------------------------------------------------------------
  if (!app_ctx->test_mode) {
    const char *usedresource;
    CeedGetResource(ceed, &usedresource);
    char hostname[PETSC_MAX_PATH_LEN];
    ierr = PetscGetHostName(hostname, sizeof hostname); CHKERRQ(ierr);
    PetscInt comm_size;
    ierr = MPI_Comm_size(comm, &comm_size); CHKERRQ(ierr);

    ierr = PetscPrintf(comm,
                       "\n-- Elasticity Example - libCEED + PETSc --\n"
                       "  MPI:\n"
                       "    Hostname                           : %s\n"
                       "    Total ranks                        : %d\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n",
                       hostname, comm_size, usedresource, CeedMemTypes[mem_type_backend]);
    CHKERRQ(ierr);

    VecType vecType;
    ierr = VecGetType(U, &vecType); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "  PETSc:\n"
                       "    PETSc Vec Type                     : %s\n",
                       vecType); CHKERRQ(ierr);

    ierr = PetscPrintf(comm,
                       "  Problem:\n"
                       "    Problem Name                       : %s\n"
                       "    Forcing Function                   : %s\n"
                       "  Mesh:\n"
                       "    File                               : %s\n"
                       "    Number of 1D Basis Nodes (p)       : %d\n"
                       "    Number of 1D Quadrature Points (q) : %d\n"
                       "    Global nodes                       : %" PetscInt_FMT "\n"
                       "    Owned nodes                        : %" PetscInt_FMT "\n"
                       "    DoF per node                       : %" PetscInt_FMT "\n"
                       "  Multigrid:\n"
                       "    Type                               : %s\n"
                       "    Number of Levels                   : %d\n",
                       app_ctx->name_for_disp,
                       forcing_types_for_disp[app_ctx->forcing_choice],
                       app_ctx->mesh_file[0] ? app_ctx->mesh_file : "Box Mesh",
                       app_ctx->degree + 1, app_ctx->degree + 1,
                       U_g_size[fine_level]/num_comp_u, U_l_size[fine_level]/num_comp_u,
                       num_comp_u,
                       (app_ctx->degree == 1 &&
                        app_ctx->multigrid_choice != MULTIGRID_NONE) ?
                       "Algebraic multigrid" :
                       multigrid_types_for_disp[app_ctx->multigrid_choice],
                       (app_ctx->degree == 1 ||
                        app_ctx->multigrid_choice == MULTIGRID_NONE) ?
                       0 : num_levels); CHKERRQ(ierr);

    if (app_ctx->multigrid_choice != MULTIGRID_NONE) {
      for (PetscInt i = 0; i < 2; i++) {
        CeedInt level = i ? fine_level : 0;
        ierr = PetscPrintf(comm,
                           "    Level %" PetscInt_FMT " (%s):\n"
                           "      Number of 1D Basis Nodes (p)     : %d\n"
                           "      Global Nodes                     : %" PetscInt_FMT "\n"
                           "      Owned Nodes                      : %" PetscInt_FMT "\n",
                           level, i ? "fine" : "coarse",
                           app_ctx->level_degrees[level] + 1,
                           U_g_size[level]/num_comp_u, U_l_size[level]/num_comp_u);
        CHKERRQ(ierr);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Setup SNES
  // ---------------------------------------------------------------------------
  // Performance logging
  ierr = PetscLogStageRegister("SNES Setup Stage", &stage_snes_setup);
  CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage_snes_setup); CHKERRQ(ierr);

  // Create SNES
  ierr = SNESCreate(comm, &snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes, level_dms[fine_level]); CHKERRQ(ierr);

  // -- Jacobian evaluators
  ierr = PetscMalloc1(num_levels, &jacob_ctx); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &jacob_mat); CHKERRQ(ierr);
  for (PetscInt level = 0; level < num_levels; level++) {
    // -- Jacobian context for level
    ierr = PetscMalloc1(1, &jacob_ctx[level]); CHKERRQ(ierr);
    ierr = SetupJacobianCtx(comm, app_ctx, level_dms[level], U_g[level],
                            U_loc[level], ceed_data[level], ceed, ctx_phys,
                            ctx_phys_smoother, jacob_ctx[level]); CHKERRQ(ierr);

    // -- Form Action of Jacobian on delta_u
    ierr = MatCreateShell(comm, U_l_size[level], U_l_size[level], U_g_size[level],
                          U_g_size[level], jacob_ctx[level], &jacob_mat[level]);
    CHKERRQ(ierr);
    ierr = MatShellSetOperation(jacob_mat[level], MATOP_MULT,
                                (void (*)(void))ApplyJacobian_Ceed);
    CHKERRQ(ierr);
    ierr = MatShellSetOperation(jacob_mat[level], MATOP_GET_DIAGONAL,
                                (void(*)(void))GetDiag_Ceed);
    ierr = MatShellSetVecType(jacob_mat[level], vectype); CHKERRQ(ierr);
  }
  // Note: FormJacobian updates Jacobian matrices on each level
  //   and assembles the Jpre matrix, if needed
  ierr = PetscMalloc1(1, &form_jacob_ctx); CHKERRQ(ierr);
  form_jacob_ctx->jacob_ctx = jacob_ctx;
  form_jacob_ctx->num_levels = num_levels;
  form_jacob_ctx->jacob_mat = jacob_mat;

  // -- Residual evaluation function
  ierr = PetscCalloc1(1, &res_ctx); CHKERRQ(ierr);
  ierr = PetscMemcpy(res_ctx, jacob_ctx[fine_level],
                     sizeof(*jacob_ctx[fine_level])); CHKERRQ(ierr);
  res_ctx->op = ceed_data[fine_level]->op_residual;
  res_ctx->qf = ceed_data[fine_level]->qf_residual;
  if (app_ctx->bc_traction_count > 0)
    res_ctx->neumann_bcs = neumann_bcs;
  else
    res_ctx->neumann_bcs = NULL;
  ierr = SNESSetFunction(snes, R, FormResidual_Ceed, res_ctx); CHKERRQ(ierr);

  // -- Prolongation/Restriction evaluation
  ierr = PetscMalloc1(num_levels, &prolong_restr_ctx); CHKERRQ(ierr);
  ierr = PetscMalloc1(num_levels, &prolong_restr_mat); CHKERRQ(ierr);
  for (PetscInt level = 1; level < num_levels; level++) {
    // ---- Prolongation/restriction context for level
    ierr = PetscMalloc1(1, &prolong_restr_ctx[level]); CHKERRQ(ierr);
    ierr = SetupProlongRestrictCtx(comm, app_ctx, level_dms[level-1],
                                   level_dms[level], U_g[level], U_loc[level-1],
                                   U_loc[level], ceed_data[level-1],
                                   ceed_data[level], ceed,
                                   prolong_restr_ctx[level]); CHKERRQ(ierr);

    // ---- Form Action of Jacobian on delta_u
    ierr = MatCreateShell(comm, U_l_size[level], U_l_size[level-1], U_g_size[level],
                          U_g_size[level-1], prolong_restr_ctx[level],
                          &prolong_restr_mat[level]); CHKERRQ(ierr);
    // Note: In PCMG, restriction is the transpose of prolongation
    ierr = MatShellSetOperation(prolong_restr_mat[level], MATOP_MULT,
                                (void (*)(void))Prolong_Ceed);
    ierr = MatShellSetOperation(prolong_restr_mat[level], MATOP_MULT_TRANSPOSE,
                                (void (*)(void))Restrict_Ceed);
    CHKERRQ(ierr);
    ierr = MatShellSetVecType(prolong_restr_mat[level], vectype); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Setup for AMG coarse solve
  // ---------------------------------------------------------------------------
  if (app_ctx->multigrid_choice != MULTIGRID_NONE) {
    // -- Jacobian Matrix
    ierr = DMCreateMatrix(level_dms[0], &jacob_mat_coarse); CHKERRQ(ierr);

    if (app_ctx->degree > 1) {
      // -- Assemble sparsity pattern
      PetscCount num_entries;
      CeedInt *rows, *cols;
      CeedVector coo_values;
      CeedOperatorLinearAssembleSymbolic(ceed_data[0]->op_jacobian, &num_entries,
                                         &rows, &cols);
      ISLocalToGlobalMapping ltog_row, ltog_col;
      ierr = MatGetLocalToGlobalMapping(jacob_mat_coarse, &ltog_row, &ltog_col);
      CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApply(ltog_row, num_entries, rows, rows);
      CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingApply(ltog_col, num_entries, cols, cols);
      CHKERRQ(ierr);
      ierr = MatSetPreallocationCOO(jacob_mat_coarse, num_entries, rows, cols);
      CHKERRQ(ierr);
      free(rows);
      free(cols);
      CeedVectorCreate(ceed, num_entries, &coo_values);

      // -- Update form_jacob_ctx
      form_jacob_ctx->coo_values = coo_values;
      form_jacob_ctx->op_coarse = ceed_data[0]->op_jacobian;
      form_jacob_ctx->jacob_mat_coarse = jacob_mat_coarse;
    }
  }

  // Set Jacobian function
  if (app_ctx->degree > 1) {
    ierr = SNESSetJacobian(snes, jacob_mat[fine_level], jacob_mat[fine_level],
                           FormJacobian, form_jacob_ctx); CHKERRQ(ierr);
  } else {
    ierr = SNESSetJacobian(snes, jacob_mat[0], jacob_mat_coarse,
                           SNESComputeJacobianDefaultColor, NULL);
    CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Setup KSP
  // ---------------------------------------------------------------------------
  {
    PC pc;
    KSP ksp;

    // -- KSP
    ierr = SNESGetKSP(snes, &ksp); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ksp, "outer_"); CHKERRQ(ierr);

    // -- Preconditioning
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetDM(pc, level_dms[fine_level]); CHKERRQ(ierr);
    ierr = PCSetOptionsPrefix(pc, "outer_"); CHKERRQ(ierr);

    if (app_ctx->multigrid_choice == MULTIGRID_NONE) {
      // ---- No Multigrid
      ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
      ierr = PCJacobiSetType(pc, PC_JACOBI_DIAGONAL); CHKERRQ(ierr);
    } else if (app_ctx->degree == 1) {
      // ---- AMG for degree 1
      ierr = PCSetType(pc, PCGAMG); CHKERRQ(ierr);
    } else {
      // ---- PCMG
      ierr = PCSetType(pc, PCMG); CHKERRQ(ierr);

      // ------ PCMG levels
      ierr = PCMGSetLevels(pc, num_levels, NULL); CHKERRQ(ierr);
      for (PetscInt level = 0; level < num_levels; level++) {
        // -------- Smoother
        KSP ksp_smoother, ksp_est;
        PC pc_smoother;

        // ---------- Smoother KSP
        ierr = PCMGGetSmoother(pc, level, &ksp_smoother); CHKERRQ(ierr);
        ierr = KSPSetDM(ksp_smoother, level_dms[level]); CHKERRQ(ierr);
        ierr = KSPSetDMActive(ksp_smoother, PETSC_FALSE); CHKERRQ(ierr);

        // ---------- Chebyshev options
        ierr = KSPSetType(ksp_smoother, KSPCHEBYSHEV); CHKERRQ(ierr);
        ierr = KSPChebyshevEstEigSet(ksp_smoother, 0, 0.1, 0, 1.1);
        CHKERRQ(ierr);
        ierr = KSPChebyshevEstEigGetKSP(ksp_smoother, &ksp_est); CHKERRQ(ierr);
        ierr = KSPSetType(ksp_est, KSPCG); CHKERRQ(ierr);
        ierr = KSPChebyshevEstEigSetUseNoisy(ksp_smoother, PETSC_TRUE);
        CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp_smoother, jacob_mat[level], jacob_mat[level]);
        CHKERRQ(ierr);

        // ---------- Smoother preconditioner
        ierr = KSPGetPC(ksp_smoother, &pc_smoother); CHKERRQ(ierr);
        ierr = PCSetType(pc_smoother, PCJACOBI); CHKERRQ(ierr);
        ierr = PCJacobiSetType(pc_smoother, PC_JACOBI_DIAGONAL); CHKERRQ(ierr);

        // -------- Work vector
        if (level != fine_level) {
          ierr = PCMGSetX(pc, level, U_g[level]); CHKERRQ(ierr);
        }

        // -------- Level prolongation/restriction operator
        if (level > 0) {
          ierr = PCMGSetInterpolation(pc, level, prolong_restr_mat[level]);
          CHKERRQ(ierr);
          ierr = PCMGSetRestriction(pc, level, prolong_restr_mat[level]);
          CHKERRQ(ierr);
        }
      }

      // ------ PCMG coarse solve
      KSP ksp_coarse;
      PC pc_coarse;

      // -------- Coarse KSP
      ierr = PCMGGetCoarseSolve(pc, &ksp_coarse); CHKERRQ(ierr);
      ierr = KSPSetType(ksp_coarse, KSPPREONLY); CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp_coarse, jacob_mat_coarse, jacob_mat_coarse);
      CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp_coarse, "coarse_"); CHKERRQ(ierr);

      // -------- Coarse preconditioner
      ierr = KSPGetPC(ksp_coarse, &pc_coarse); CHKERRQ(ierr);
      ierr = PCSetType(pc_coarse, PCGAMG); CHKERRQ(ierr);
      ierr = PCSetOptionsPrefix(pc_coarse, "coarse_"); CHKERRQ(ierr);

      ierr = KSPSetFromOptions(ksp_coarse); CHKERRQ(ierr);
      ierr = PCSetFromOptions(pc_coarse); CHKERRQ(ierr);

      // ------ PCMG options
      ierr = PCMGSetType(pc, PC_MG_MULTIPLICATIVE); CHKERRQ(ierr);
      ierr = PCMGSetNumberSmooth(pc, 3); CHKERRQ(ierr);
      ierr = PCMGSetCycleType(pc, pcmg_cycle_type); CHKERRQ(ierr);
    }
    ierr = KSPSetFromOptions(ksp);
    ierr = PCSetFromOptions(pc);
  }
  {
    // Default to critical-point (CP) line search (related to Wolfe's curvature condition)
    SNESLineSearch line_search;

    ierr = SNESGetLineSearch(snes, &line_search); CHKERRQ(ierr);
    ierr = SNESLineSearchSetType(line_search, SNESLINESEARCHCP); CHKERRQ(ierr);
  }

  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  // Performance logging
  ierr = PetscLogStagePop();

  // ---------------------------------------------------------------------------
  // Set initial guess
  // ---------------------------------------------------------------------------
  ierr = PetscObjectSetName((PetscObject)U, ""); CHKERRQ(ierr);
  ierr = VecSet(U, 0.0); CHKERRQ(ierr);

  // View solution
  if (app_ctx->view_soln) {
    ierr = ViewSolution(comm, app_ctx, U, 0, 0.0); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Solve SNES
  // ---------------------------------------------------------------------------
  PetscBool snes_monitor = PETSC_FALSE;
  ierr = PetscOptionsHasName(NULL, NULL, "-snes_monitor", &snes_monitor);
  CHKERRQ(ierr);

  // Performance logging
  ierr = PetscLogStageRegister("SNES Solve Stage", &stage_snes_solve);
  CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage_snes_solve); CHKERRQ(ierr);

  // Timing
  ierr = PetscBarrier((PetscObject)snes); CHKERRQ(ierr);
  start_time = MPI_Wtime();

  // Solve for each load increment
  PetscInt increment;
  for (increment = 1; increment <= app_ctx->num_increments; increment++) {
    // -- Log increment count
    if (snes_monitor) {
      ierr = PetscPrintf(comm, "%d Load Increment\n", increment - 1);
      CHKERRQ(ierr);
    }

    // -- Scale the problem
    PetscScalar load_increment = 1.0*increment / app_ctx->num_increments,
                scalingFactor = load_increment /
                                (increment == 1 ? 1 : res_ctx->load_increment);
    res_ctx->load_increment = load_increment;
    if (app_ctx->num_increments > 1 && app_ctx->forcing_choice != FORCE_NONE) {
      ierr = VecScale(F, scalingFactor); CHKERRQ(ierr);
    }

    // -- Solve
    ierr = SNESSolve(snes, F, U); CHKERRQ(ierr);

    // -- View solution
    if (app_ctx->view_soln) {
      ierr = ViewSolution(comm, app_ctx, U, increment, load_increment); CHKERRQ(ierr);
    }

    // -- Update SNES iteration count
    PetscInt its;
    ierr = SNESGetIterationNumber(snes, &its); CHKERRQ(ierr);
    snes_its += its;
    ierr = SNESGetLinearSolveIterations(snes, &its); CHKERRQ(ierr);
    ksp_its += its;

    // -- Check for divergence
    SNESConvergedReason reason;
    ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);
    if (reason < 0)
      break;
    if (app_ctx->energy_viewer) {
      // -- Log strain energy for current load increment
      CeedScalar energy;
      ierr = ComputeStrainEnergy(dm_energy, res_ctx, ceed_data[fine_level]->op_energy,
                                 U, &energy); CHKERRQ(ierr);

      if (!app_ctx->test_mode) {
        // -- Output
        ierr = PetscPrintf(comm,
                           "    Strain Energy                      : %.12e\n",
                           energy); CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(app_ctx->energy_viewer, "%f,%e\n", load_increment,
                                    energy); CHKERRQ(ierr);
    }
  }

  // Timing
  elapsed_time = MPI_Wtime() - start_time;

  // Performance logging
  ierr = PetscLogStagePop();

  // ---------------------------------------------------------------------------
  // Output summary
  // ---------------------------------------------------------------------------
  if (!app_ctx->test_mode) {
    // -- SNES
    SNESType snes_type;
    SNESConvergedReason reason;
    PetscReal rnorm;
    ierr = SNESGetType(snes, &snes_type); CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);
    ierr = SNESGetFunctionNorm(snes, &rnorm); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "  SNES:\n"
                       "    SNES Type                          : %s\n"
                       "    SNES Convergence                   : %s\n"
                       "    Number of Load Increments          : %d\n"
                       "    Completed Load Increments          : %d\n"
                       "    Total SNES Iterations              : %" PetscInt_FMT "\n"
                       "    Final rnorm                        : %e\n",
                       snes_type, SNESConvergedReasons[reason],
                       app_ctx->num_increments, increment - 1,
                       snes_its, (double)rnorm); CHKERRQ(ierr);

    // -- KSP
    KSP ksp;
    KSPType ksp_type;
    ierr = SNESGetKSP(snes, &ksp); CHKERRQ(ierr);
    ierr = KSPGetType(ksp, &ksp_type); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "  Linear Solver:\n"
                       "    KSP Type                           : %s\n"
                       "    Total KSP Iterations               : %" PetscInt_FMT "\n",
                       ksp_type, ksp_its); CHKERRQ(ierr);

    // -- PC
    PC pc;
    PCType pc_type;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCGetType(pc, &pc_type); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "    PC Type                            : %s\n",
                       pc_type); CHKERRQ(ierr);

    if (!strcmp(pc_type, PCMG)) {
      PCMGType pcmg_type;
      ierr = PCMGGetType(pc, &pcmg_type); CHKERRQ(ierr);
      ierr = PetscPrintf(comm,
                         "  P-Multigrid:\n"
                         "    PCMG Type                          : %s\n"
                         "    PCMG Cycle Type                    : %s\n",
                         PCMGTypes[pcmg_type],
                         PCMGCycleTypes[pcmg_cycle_type]); CHKERRQ(ierr);

      // -- Coarse Solve
      KSP ksp_coarse;
      PC pc_coarse;
      PCType pc_type;

      ierr = PCMGGetCoarseSolve(pc, &ksp_coarse); CHKERRQ(ierr);
      ierr = KSPGetType(ksp_coarse, &ksp_type); CHKERRQ(ierr);
      ierr = KSPGetPC(ksp_coarse, &pc_coarse); CHKERRQ(ierr);
      ierr = PCGetType(pc_coarse, &pc_type); CHKERRQ(ierr);
      ierr = PetscPrintf(comm,
                         "    Coarse Solve:\n"
                         "      KSP Type                         : %s\n"
                         "      PC Type                          : %s\n",
                         ksp_type, pc_type); CHKERRQ(ierr);
    }
  }

  // ---------------------------------------------------------------------------
  // Compute solve time
  // ---------------------------------------------------------------------------
  if (!app_ctx->test_mode) {
    ierr = MPI_Allreduce(&elapsed_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, comm);
    CHKERRQ(ierr);
    ierr = MPI_Allreduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, comm);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "  Performance:\n"
                       "    SNES Solve Time                    : %g (%g) sec\n"
                       "    DoFs/Sec in SNES                   : %g (%g) million\n",
                       max_time, min_time, 1e-6*U_g_size[fine_level]*ksp_its/max_time,
                       1e-6*U_g_size[fine_level]*ksp_its/min_time); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Compute error
  // ---------------------------------------------------------------------------
  if (app_ctx->forcing_choice == FORCE_MMS) {
    CeedScalar l2_error = 1., l2_U_norm = 1.;
    const CeedScalar *true_array;
    Vec error_vec, true_vec;

    // -- Work vectors
    ierr = VecDuplicate(U, &error_vec); CHKERRQ(ierr);
    ierr = VecSet(error_vec, 0.0); CHKERRQ(ierr);
    ierr = VecDuplicate(U, &true_vec); CHKERRQ(ierr);
    ierr = VecSet(true_vec, 0.0); CHKERRQ(ierr);

    // -- Assemble global true solution vector
    CeedVectorGetArrayRead(ceed_data[fine_level]->true_soln,
                           CEED_MEM_HOST, &true_array);
    ierr = VecPlaceArray(res_ctx->Y_loc, (PetscScalar *)true_array);
    CHKERRQ(ierr);
    ierr = DMLocalToGlobal(res_ctx->dm, res_ctx->Y_loc, INSERT_VALUES, true_vec);
    CHKERRQ(ierr);
    ierr = VecResetArray(res_ctx->Y_loc); CHKERRQ(ierr);
    CeedVectorRestoreArrayRead(ceed_data[fine_level]->true_soln, &true_array);

    // -- Compute L2 error
    ierr = VecWAXPY(error_vec, -1.0, U, true_vec); CHKERRQ(ierr);
    ierr = VecNorm(error_vec, NORM_2, &l2_error); CHKERRQ(ierr);
    ierr = VecNorm(U, NORM_2, &l2_U_norm); CHKERRQ(ierr);
    l2_error /= l2_U_norm;

    // -- Output
    if (!app_ctx->test_mode || l2_error > 0.05) {
      ierr = PetscPrintf(comm,
                         "    L2 Error                           : %e\n",
                         l2_error); CHKERRQ(ierr);
    }

    // -- Cleanup
    ierr = VecDestroy(&error_vec); CHKERRQ(ierr);
    ierr = VecDestroy(&true_vec); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Compute energy
  // ---------------------------------------------------------------------------
  PetscReal energy;
  ierr = ComputeStrainEnergy(dm_energy, res_ctx, ceed_data[fine_level]->op_energy,
                             U, &energy); CHKERRQ(ierr);
  if (!app_ctx->test_mode) {
    // -- Output
    ierr = PetscPrintf(comm,
                       "    Strain Energy                      : %.12e\n",
                       energy); CHKERRQ(ierr);
  }
  ierr = RegressionTests_solids(app_ctx, energy); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Output diagnostic quantities
  // ---------------------------------------------------------------------------
  if (app_ctx->view_soln || app_ctx->view_final_soln) {
    // -- Setup context
    UserMult diagnostic_ctx;
    ierr = PetscMalloc1(1, &diagnostic_ctx); CHKERRQ(ierr);
    ierr = PetscMemcpy(diagnostic_ctx, res_ctx, sizeof(*res_ctx)); CHKERRQ(ierr);
    diagnostic_ctx->dm = dm_diagnostic;
    diagnostic_ctx->op = ceed_data[fine_level]->op_diagnostic;

    // -- Compute and output
    ierr = ViewDiagnosticQuantities(comm, level_dms[fine_level], diagnostic_ctx,
                                    app_ctx, U,
                                    ceed_data[fine_level]->elem_restr_diagnostic);
    CHKERRQ(ierr);

    // -- Cleanup
    ierr = PetscFree(diagnostic_ctx); CHKERRQ(ierr);
  }

  if(ceed_data[fine_level]->qf_tape) {
    DM dm_tape;
    PetscInt num_comp_t = 1;
    ierr = DMClone(dm_orig, &dm_tape); CHKERRQ(ierr);
    ierr = SetupDMByDegree(dm_tape, app_ctx, app_ctx->level_degrees[fine_level],
                           PETSC_FALSE, num_comp_t); CHKERRQ(ierr);
    ierr = DMSetVecType(dm_tape, vectype); CHKERRQ(ierr);
    ierr = FreeTapeMemory(dm_tape, res_ctx, ceed_data[fine_level]->op_tape, U);
    CHKERRQ(ierr);
    ierr = DMDestroy(&dm_tape); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Free objects
  // ---------------------------------------------------------------------------
  // Data in arrays per level
  for (PetscInt level = 0; level < num_levels; level++) {
    // Vectors
    ierr = VecDestroy(&U_g[level]); CHKERRQ(ierr);
    ierr = VecDestroy(&U_loc[level]); CHKERRQ(ierr);

    // Jacobian matrix and data
    ierr = VecDestroy(&jacob_ctx[level]->Y_loc); CHKERRQ(ierr);
    ierr = MatDestroy(&jacob_mat[level]); CHKERRQ(ierr);
    ierr = PetscFree(jacob_ctx[level]); CHKERRQ(ierr);

    // Prolongation/Restriction matrix and data
    if (level > 0) {
      ierr = PetscFree(prolong_restr_ctx[level]); CHKERRQ(ierr);
      ierr = MatDestroy(&prolong_restr_mat[level]); CHKERRQ(ierr);
    }

    // DM
    ierr = DMDestroy(&level_dms[level]); CHKERRQ(ierr);

    // libCEED objects
    ierr = CeedDataDestroy(level, ceed_data[level]); CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&app_ctx->energy_viewer); CHKERRQ(ierr);

  // Arrays
  ierr = PetscFree(U_g); CHKERRQ(ierr);
  ierr = PetscFree(U_loc); CHKERRQ(ierr);
  ierr = PetscFree(U_g_size); CHKERRQ(ierr);
  ierr = PetscFree(U_l_size); CHKERRQ(ierr);
  ierr = PetscFree(U_loc_size); CHKERRQ(ierr);
  ierr = PetscFree(jacob_ctx); CHKERRQ(ierr);
  ierr = PetscFree(jacob_mat); CHKERRQ(ierr);
  ierr = PetscFree(prolong_restr_ctx); CHKERRQ(ierr);
  ierr = PetscFree(prolong_restr_mat); CHKERRQ(ierr);
  ierr = PetscFree(app_ctx->level_degrees); CHKERRQ(ierr);
  ierr = PetscFree(ceed_data); CHKERRQ(ierr);

  // libCEED objects
  CeedVectorDestroy(&form_jacob_ctx->coo_values);
  CeedQFunctionContextDestroy(&ctx_phys);
  CeedQFunctionContextDestroy(&ctx_phys_smoother);
  CeedDestroy(&ceed);

  // PETSc objects
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = VecDestroy(&R); CHKERRQ(ierr);
  ierr = VecDestroy(&R_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = VecDestroy(&F_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&neumann_bcs); CHKERRQ(ierr);
  ierr = VecDestroy(&bcs_loc); CHKERRQ(ierr);
  ierr = MatDestroy(&jacob_mat_coarse); CHKERRQ(ierr);
  ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  ierr = DMDestroy(&dm_orig); CHKERRQ(ierr);
  ierr = DMDestroy(&dm_energy); CHKERRQ(ierr);
  ierr = DMDestroy(&dm_diagnostic); CHKERRQ(ierr);
  ierr = PetscFree(level_dms); CHKERRQ(ierr);

  // -- Function list
  ierr = PetscFunctionListDestroy(&problem_functions->setupPhysics);
  CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&problem_functions->setupLibceedFineLevel);
  CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&problem_functions->setupLibceedLevel);
  CHKERRQ(ierr);

  // Structs
  ierr = PetscFree(res_ctx); CHKERRQ(ierr);
  ierr = PetscFree(form_jacob_ctx); CHKERRQ(ierr);
  ierr = PetscFree(jacob_coarse_ctx); CHKERRQ(ierr);
  ierr = PetscFree(app_ctx); CHKERRQ(ierr);
  ierr = PetscFree(problem_functions); CHKERRQ(ierr);
  ierr = PetscFree(units); CHKERRQ(ierr);

  return PetscFinalize();
}
