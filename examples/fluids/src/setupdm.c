// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
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
PetscErrorCode CreateDM(MPI_Comm comm, ProblemData *problem, MatType mat_type, VecType vec_type, DM *dm) {
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
  PetscCall(PetscOptionsSetValue(NULL, "-dm_plex_simplex", "0"));
  PetscCall(PetscOptionsSetValue(NULL, "-dm_sparse_localize", "0"));
  // Set CL options
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Setup DM
PetscErrorCode SetUpDM(DM dm, ProblemData *problem, PetscInt degree, PetscInt q_extra, SimpleBC bc, Physics phys) {
  PetscInt q_order = degree + q_extra;
  PetscFunctionBeginUser;
  {
    PetscBool is_simplex = PETSC_TRUE;

    // Check if simplex or tensor-product mesh
    PetscCall(DMPlexIsSimplex(dm, &is_simplex));
    // Configure the finite element space and boundary conditions
    PetscFE  fe;
    PetscInt num_comp_q = 5;
    DMLabel  label;
    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, problem->dim, num_comp_q, is_simplex, degree, q_order, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "Q"));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(dm));
    {  // Project coordinates to enrich quadrature space
      DM             dm_coord;
      PetscDS        ds_coord;
      PetscFE        fe_coord_current, fe_coord_new, fe_coord_face_new;
      PetscDualSpace fe_coord_dual_space;
      PetscInt       fe_coord_order, num_comp_coord;

      PetscCall(DMGetCoordinateDM(dm, &dm_coord));
      PetscCall(DMGetCoordinateDim(dm, &num_comp_coord));
      PetscCall(DMGetRegionDS(dm_coord, NULL, NULL, &ds_coord, NULL));
      PetscCall(PetscDSGetDiscretization(ds_coord, 0, (PetscObject *)&fe_coord_current));
      PetscCall(PetscFEGetDualSpace(fe_coord_current, &fe_coord_dual_space));
      PetscCall(PetscDualSpaceGetOrder(fe_coord_dual_space, &fe_coord_order));

      // Create FE for coordinates
      PetscCheck(fe_coord_order == 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER_INPUT,
                 "Only linear mesh geometry supported. Recieved %d order", fe_coord_order);
      PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, problem->dim, num_comp_coord, is_simplex, fe_coord_order, q_order, &fe_coord_new));
      PetscCall(PetscFEGetHeightSubspace(fe_coord_new, 1, &fe_coord_face_new));
      PetscCall(DMProjectCoordinates(dm, fe_coord_new));
      PetscCall(PetscFEDestroy(&fe_coord_new));
    }

    PetscCall(DMGetLabel(dm, "Face Sets", &label));
    PetscCall(DMPlexLabelComplete(dm, label));
    // Set wall BCs
    if (bc->num_wall > 0) {
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, bc->num_wall, bc->walls, 0, bc->num_comps, bc->wall_comps, NULL, NULL, NULL, NULL));
    }
    // Set slip BCs in the x direction
    if (bc->num_slip[0] > 0) {
      PetscInt comps[1] = {1};
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", label, bc->num_slip[0], bc->slips[0], 0, 1, comps, NULL, NULL, NULL, NULL));
    }
    // Set slip BCs in the y direction
    if (bc->num_slip[1] > 0) {
      PetscInt comps[1] = {2};
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", label, bc->num_slip[1], bc->slips[1], 0, 1, comps, NULL, NULL, NULL, NULL));
    }
    // Set slip BCs in the z direction
    if (bc->num_slip[2] > 0) {
      PetscInt comps[1] = {3};
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", label, bc->num_slip[2], bc->slips[2], 0, 1, comps, NULL, NULL, NULL, NULL));
    }
    {
      PetscBool use_strongstg = PETSC_FALSE;
      PetscCall(PetscOptionsGetBool(NULL, NULL, "-stg_strong", &use_strongstg, NULL));
      if (use_strongstg) PetscCall(SetupStrongSTG(dm, bc, problem, phys));
    }

    if (!is_simplex) {
      DM dm_coord;
      PetscCall(DMGetCoordinateDM(dm, &dm_coord));
      PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
      PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
    }

    PetscCall(PetscFEDestroy(&fe));
  }

  // Empty name for conserved field (because there is only one field)
  PetscSection section;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  switch (phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      PetscCall(PetscSectionSetComponentName(section, 0, 0, "Density"));
      PetscCall(PetscSectionSetComponentName(section, 0, 1, "Momentum X"));
      PetscCall(PetscSectionSetComponentName(section, 0, 2, "Momentum Y"));
      PetscCall(PetscSectionSetComponentName(section, 0, 3, "Momentum Z"));
      PetscCall(PetscSectionSetComponentName(section, 0, 4, "Energy Density"));
      break;

    case STATEVAR_PRIMITIVE:
      PetscCall(PetscSectionSetComponentName(section, 0, 0, "Pressure"));
      PetscCall(PetscSectionSetComponentName(section, 0, 1, "Velocity X"));
      PetscCall(PetscSectionSetComponentName(section, 0, 2, "Velocity Y"));
      PetscCall(PetscSectionSetComponentName(section, 0, 3, "Velocity Z"));
      PetscCall(PetscSectionSetComponentName(section, 0, 4, "Temperature"));
      break;

    case STATEVAR_ENTROPY:
      PetscCall(PetscSectionSetComponentName(section, 0, 0, "EntropyDensity"));
      PetscCall(PetscSectionSetComponentName(section, 0, 1, "EntropyMomentumX"));
      PetscCall(PetscSectionSetComponentName(section, 0, 2, "EntropyMomentumY"));
      PetscCall(PetscSectionSetComponentName(section, 0, 3, "EntropyMomentumZ"));
      PetscCall(PetscSectionSetComponentName(section, 0, 4, "EntropyEnergy"));
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Refine DM for high-order viz
PetscErrorCode VizRefineDM(DM dm, User user, ProblemData *problem, SimpleBC bc, Physics phys) {
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
    PetscCall(SetUpDM(dm_hierarchy[i + 1], problem, d, q_order, bc, phys));
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
