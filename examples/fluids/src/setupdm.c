// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  PetscCall(PetscOptionsSetValue(NULL, "-dm_plex_simplex", "0"));
  PetscCall(PetscOptionsSetValue(NULL, "-dm_sparse_localize", "0"));
  PetscCall(PetscOptionsSetValue(NULL, "-dm_localize", "0"));  // Localization done in DMSetupByOrderEnd_FEM
  PetscCall(PetscOptionsSetValue(NULL, "-dm_blocking_type", "field_node"));

  // Set CL options
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Setup DM
PetscErrorCode SetUpDM(DM dm, ProblemData problem, PetscInt degree, PetscInt q_extra, SimpleBC bc, Physics phys) {
  PetscInt num_comp_q = 5;
  PetscFunctionBeginUser;

  PetscCall(DMSetupByOrderBegin_FEM(PETSC_TRUE, PETSC_TRUE, degree, PETSC_DECIDE, q_extra, 1, &num_comp_q, dm));

  {  // Add strong boundary conditions to DM
    DMLabel label;
    PetscCall(DMGetLabel(dm, "Face Sets", &label));
    PetscCall(DMPlexLabelComplete(dm, label));

    for (PetscInt i = 0; i < problem->num_bc_defs; i++) {
      BCDefinition    bc_def = problem->bc_defs[i];
      PetscInt        num_essential_comps, num_label_values;
      const PetscInt *essential_comps, *label_values;
      const char     *name;

      PetscCall(BCDefinitionGetEssential(bc_def, &num_essential_comps, &essential_comps));
      if (essential_comps > 0) {
        PetscCall(BCDefinitionGetInfo(bc_def, &name, &num_label_values, &label_values));
        PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, name, label, num_label_values, label_values, 0, num_essential_comps, essential_comps, NULL, NULL,
                                NULL, NULL));
      }
    }
    {
      PetscBool use_strongstg = PETSC_FALSE;
      PetscCall(PetscOptionsGetBool(NULL, NULL, "-stg_strong", &use_strongstg, NULL));
      if (use_strongstg) PetscCall(SetupStrongStg(dm, bc, problem, phys));
    }
  }

  PetscCall(DMSetupByOrderEnd_FEM(PETSC_TRUE, dm));

  // Empty name for conserved field (because there is only one field)
  PetscSection section;
  PetscCall(DMGetLocalSection(dm, &section));
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
