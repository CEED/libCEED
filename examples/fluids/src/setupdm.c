// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Setup DM for Navier-Stokes example using PETSc

#include <ceed.h>
#include <petscdmplex.h>
#include <petscds.h>

#include "../navierstokes.h"
#include "../problems/stg_shur14.h"

// Create mesh
PetscErrorCode CreateDM(MPI_Comm comm, ProblemData problem, MatType mat_type, VecType vec_type, DM *dm) {
  PetscFunctionBeginUser;
  // Create DMPLEX
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  {
    PetscBool skip = PETSC_TRUE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_mat_preallocate_skip", &skip, NULL));
    PetscCall(DMSetMatrixPreallocateSkip(*dm, skip));
  }
  PetscCall(DMSetMatType(*dm, mat_type));
  PetscCall(DMSetVecType(*dm, vec_type));

  // Set Tensor elements
//  PetscCall(PetscOptionsSetValue(NULL, "-dm_plex_simplex", "1"));
  PetscCall(PetscOptionsSetValue(NULL, "-dm_sparse_localize", "0"));
  PetscCall(PetscOptionsSetValue(NULL, "-dm_localize", "0"));  // Localization done in DMSetupByOrderEnd_FEM
  PetscCall(PetscOptionsSetValue(NULL, "-dm_blocking_type", "field_node"));

  // Set CL options
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "After DMSetFromOptions in CreateDM in setupdm.c : \n"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void evaluate_solution(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
                              const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
                              const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal X[], PetscInt numConstants,
                              const PetscScalar constants[], PetscScalar new_u[]) {
  for (PetscInt i = 0; i < 5; i++) new_u[i] = u[i];
}

// Setup DM
PetscErrorCode SetUpDM(DM *dm, ProblemData *problem, PetscInt degree, PetscInt q_extra, SimpleBC bc, Physics phys) {
  PetscInt  num_comp_q     = 5;
  DM        old_dm         = NULL;
  PetscBool SkipProjection = PETSC_FALSE;

  PetscFunctionBeginUser;
  //  Restore a NL vector if requested (same flag used in Distribute)
  char vecName[PETSC_MAX_PATH_LEN] = "";
  PetscOptionsBegin(PetscObjectComm((PetscObject)dm), NULL, "Option Named Vector", NULL);
  PetscCall(PetscOptionsString("-named_local_vector_migrate", "Name of NamedLocalVector to migrate", NULL, vecName, vecName, sizeof(vecName), NULL));
  PetscCall(PetscStrncmp(vecName, "SkipProjection", 14, &SkipProjection));
  PetscOptionsEnd();

  PetscBool has_NL_vector, has_NL_vectord;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Just inside of SetUpDM in setupdm.c : \n"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  PetscCall(DMHasNamedLocalVector(*dm, vecName, &has_NL_vector));
  if (has_NL_vector) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "inside has_NL_vector condtional src/setupdm.c : \n"));
    if (SkipProjection) {
      PetscCall(DMClearFields(*dm));
      PetscCall(DMSetLocalSection(*dm, NULL));
      PetscCall(DMSetSectionSF(*dm, NULL));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "inside SkipProjection condtional src/setupdm.c : \n"));
    } else {
      char vecNamed[PETSC_MAX_PATH_LEN] = "";
      PetscStrcpy(vecNamed, vecName);
      PetscStrlcat(vecNamed, "d", PETSC_MAX_PATH_LEN);
      PetscCall(DMHasNamedLocalVector(*dm, vecNamed, &has_NL_vectord));
      if (has_NL_vectord) PetscStrcpy(vecName, vecNamed);  // distributed correction is the vector we should use
      DM new_dm = NULL;

      PetscScalar valc;
      PetscInt stepNumC;
      PetscCall(DMGetOutputSequenceNumber(*dm, &stepNumC, &valc));

      PetscCall(DMClone(*dm, &new_dm));
      if(stepNumC >= 0) PetscCall(DMSetOutputSequenceNumber(new_dm, stepNumC, valc));

      PetscSF face_sf;
      PetscInt nispf;
      PetscCall(DMPlexGetIsoperiodicFaceSF(*dm, &nispf, &face_sf));
      PetscCall(DMPlexSetIsoperiodicFaceSF(new_dm, nispf, face_sf));
      old_dm = *dm;
      *dm    = new_dm;
    }
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Before DMSetupByOrderBegin in src/setupdm.c : \n"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscBool setupFace = PETSC_TRUE;
  PetscBool setupCoord = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_setupFace", &setupFace, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_setupCoord", &setupCoord, NULL));

// FAILED BOTH  PetscCall(DMSetupByOrderBegin_FEM(PETSC_FALSE, PETSC_FALSE, degree, PETSC_DECIDE, q_extra, 1, &num_comp_q, *dm));
  PetscCall(DMSetupByOrderBegin_FEM(setupFace, setupCoord, degree, PETSC_DECIDE, q_extra, 1, &num_comp_q, *dm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Before bcwork in src/setupdm.c : \n"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  {  // Add strong boundary conditions to DM
    DMLabel label;
    PetscCall(DMGetLabel(*dm, "Face Sets", &label));
    PetscCall(DMPlexLabelComplete(*dm, label));
    // Set wall BCs
    if (bc->num_wall > 0) {
      PetscCall(DMAddBoundary(*dm, DM_BC_ESSENTIAL, "wall", label, bc->num_wall, bc->walls, 0, bc->num_comps, bc->wall_comps, NULL, NULL, NULL, NULL));
    }
    // Set symmetry BCs in the x direction
    if (bc->num_symmetry[0] > 0) {
      PetscInt comps[1] = {1};
      PetscCall(DMAddBoundary(*dm, DM_BC_ESSENTIAL, "symmetry_x", label, bc->num_symmetry[0], bc->symmetries[0], 0, 1, comps, NULL, NULL, NULL, NULL));
    }
    // Set symmetry BCs in the y direction
    if (bc->num_symmetry[1] > 0) {
      PetscInt comps[1] = {2};
      PetscCall(DMAddBoundary(*dm, DM_BC_ESSENTIAL, "symmetry_y", label, bc->num_symmetry[1], bc->symmetries[1], 0, 1, comps, NULL, NULL, NULL, NULL));
    }
    // Set symmetry BCs in the z direction
    if (bc->num_symmetry[2] > 0) {
      PetscInt comps[1] = {3};
      PetscCall(DMAddBoundary(*dm, DM_BC_ESSENTIAL, "symmetry_z", label, bc->num_symmetry[2], bc->symmetries[2], 0, 1, comps, NULL, NULL, NULL, NULL));
    }
    {
      PetscBool use_strongstg = PETSC_FALSE;
      PetscCall(PetscOptionsGetBool(NULL, NULL, "-stg_strong", &use_strongstg, NULL));
      if (use_strongstg) PetscCall(SetupStrongStg(*dm, bc, problem, phys));
    }
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Before DMSetupByOrderEnd_FEMin src/setupdm.c : \n"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscCall(DMSetupByOrderEnd_FEM(setupCoord, *dm));

  if (has_NL_vector && !SkipProjection) {
    Vec old_InitialCondition_loc, new_InitialCondition_loc;
    PetscCall(DMGetNamedLocalVector(old_dm, vecName, &old_InitialCondition_loc));
    PetscCall(DMGetNamedLocalVector(*dm, vecName, &new_InitialCondition_loc));
    void (*funcs[])(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                    const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[],
                    PetscInt, const PetscScalar[], PetscScalar[]) = {evaluate_solution};

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Before DMProjectFieldLocal in src/setupdm.c : \n"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

    PetscCall(DMProjectFieldLocal(*dm, 0.0, old_InitialCondition_loc, funcs, INSERT_ALL_VALUES, new_InitialCondition_loc));
    PetscCall(DMRestoreNamedLocalVector(old_dm, vecName, &old_InitialCondition_loc));
    PetscCall(DMRestoreNamedLocalVector(*dm, vecName, &new_InitialCondition_loc));
    PetscCall(DMDestroy(&old_dm));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "After DMProjectFieldLocal in src/setupdm.c : \n"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  // Empty name for conserved field (because there is only one field)
  PetscSection section;
  PetscCall(DMGetLocalSection(*dm, &section));
  PetscCall(PetscSectionViewFromOptions(section, NULL, "-sectionAfterProject"));

  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  switch (phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      PetscCall(PetscSectionSetComponentName(section, 0, 0, "Density"));
      PetscCall(PetscSectionSetComponentName(section, 0, 1, "MomentumX"));
      PetscCall(PetscSectionSetComponentName(section, 0, 2, "MomentumY"));
      PetscCall(PetscSectionSetComponentName(section, 0, 3, "MomentumZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, 4, "TotalEnergy"));
      break;

    case STATEVAR_PRIMITIVE:
      PetscCall(PetscSectionSetComponentName(section, 0, 0, "Pressure"));
      PetscCall(PetscSectionSetComponentName(section, 0, 1, "VelocityX"));
      PetscCall(PetscSectionSetComponentName(section, 0, 2, "VelocityY"));
      PetscCall(PetscSectionSetComponentName(section, 0, 3, "VelocityZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, 4, "Temperature"));
      break;

    case STATEVAR_ENTROPY:
      PetscCall(PetscSectionSetComponentName(section, 0, 0, "EntropyDensity"));
      PetscCall(PetscSectionSetComponentName(section, 0, 1, "EntropyMomentumX"));
      PetscCall(PetscSectionSetComponentName(section, 0, 2, "EntropyMomentumY"));
      PetscCall(PetscSectionSetComponentName(section, 0, 3, "EntropyMomentumZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, 4, "EntropyTotalEnergy"));
      break;
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "bottom of SetUpDM in src/setupdm.c : \n"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Refine DM for high-order viz
PetscErrorCode VizRefineDM(DM dm, User user, ProblemData problem, SimpleBC bc, Physics phys) {
  DM      dm_hierarchy[user->app_ctx->viz_refine + 1];
  VecType vec_type;

  PetscFunctionBeginUser;
  PetscCall(DMPlexSetRefinementUniform(dm, PETSC_TRUE));

  dm_hierarchy[0] = dm;
  for (PetscInt i = 0, d = user->app_ctx->degree; i < user->app_ctx->viz_refine; i++) {
    Mat interp_next;
    PetscCall(DMRefine(dm_hierarchy[i], MPI_COMM_NULL, &dm_hierarchy[i + 1]));
    PetscCall(DMClearDS(dm_hierarchy[i + 1]));
    PetscCall(DMClearFields(dm_hierarchy[i + 1]));
    PetscCall(DMSetCoarseDM(dm_hierarchy[i + 1], dm_hierarchy[i]));
    d                = (d + 1) / 2;
    PetscInt q_order = d + user->app_ctx->q_extra;
    if (i + 1 == user->app_ctx->viz_refine) d = 1;
    PetscCall(DMGetVecType(dm, &vec_type));
    PetscCall(DMSetVecType(dm_hierarchy[i + 1], vec_type));
    PetscCall(SetUpDM(&dm_hierarchy[i + 1], problem, d, q_order, bc, phys));
    PetscCall(DMCreateInterpolation(dm_hierarchy[i], dm_hierarchy[i + 1], &interp_next, NULL));
    if (!i) user->interp_viz = interp_next;
    else {
      Mat C;
      PetscCall(MatMatMult(interp_next, user->interp_viz, MAT_INITIAL_MATRIX, PETSC_DECIDE, &C));
      PetscCall(MatDestroy(&interp_next));
      PetscCall(MatDestroy(&user->interp_viz));
      user->interp_viz = C;
    }
  }
  for (PetscInt i = 1; i < user->app_ctx->viz_refine; i++) {
    PetscCall(DMDestroy(&dm_hierarchy[i]));
  }
  user->dm_viz = dm_hierarchy[user->app_ctx->viz_refine];
  PetscFunctionReturn(PETSC_SUCCESS);
}
