// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../navierstokes.h"

/**
   @brief Add `BCDefinition` to a `PetscSegBuffer`

   @param[in]     bc_def      `BCDefinition` to add
   @param[in,out] bc_defs_seg `PetscSegBuffer` to add to
**/
static PetscErrorCode AddBCDefinitionToSegBuffer(BCDefinition bc_def, PetscSegBuffer bc_defs_seg) {
  BCDefinition *bc_def_ptr;

  PetscFunctionBeginUser;
  if (bc_def == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSegBufferGet(bc_defs_seg, 1, &bc_def_ptr));
  *bc_def_ptr = bc_def;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Create and setup `BCDefinition`s and `SimpleBC` from commandline options

   @param[in]     user    `User`
   @param[in,out] problem `ProblemData`
   @param[in]     app_ctx `AppCtx`
   @param[in,out] bc      `SimpleBC`
**/
PetscErrorCode BoundaryConditionSetUp(User user, ProblemData problem, AppCtx app_ctx, SimpleBC bc) {
  PetscSegBuffer bc_defs_seg;
  PetscBool      flg;
  BCDefinition   bc_def;

  PetscFunctionBeginUser;
  PetscCall(PetscSegBufferCreate(sizeof(BCDefinition), 4, &bc_defs_seg));

  PetscOptionsBegin(user->comm, NULL, "Boundary Condition Options", NULL);

  PetscCall(PetscOptionsBCDefinition("-bc_wall", "Face IDs to apply wall BC", NULL, "wall", &bc_def, NULL));
  PetscCall(AddBCDefinitionToSegBuffer(bc_def, bc_defs_seg));
  if (bc_def) {
    PetscInt num_essential_comps = 16, essential_comps[16];

    PetscCall(PetscOptionsIntArray("-wall_comps", "An array of constrained component numbers", NULL, essential_comps, &num_essential_comps, &flg));
    PetscCall(BCDefinitionSetEssential(bc_def, num_essential_comps, essential_comps));

    app_ctx->wall_forces.num_wall = bc_def->num_label_values;
    PetscCall(PetscMalloc1(bc_def->num_label_values, &app_ctx->wall_forces.walls));
    PetscCall(PetscArraycpy(app_ctx->wall_forces.walls, bc_def->label_values, bc_def->num_label_values));
  }

  {  // Symmetry Boundary Conditions
    const char *deprecated[3] = {"-bc_slip_x", "-bc_slip_y", "-bc_slip_z"};
    const char *flags[3]      = {"-bc_symmetry_x", "-bc_symmetry_y", "-bc_symmetry_z"};

    for (PetscInt j = 0; j < 3; j++) {
      PetscCall(PetscOptionsDeprecated(deprecated[j], flags[j], "libCEED 0.12.0",
                                       "Use -bc_symmetry_[x,y,z] for direct equivalency, or -bc_slip for weak, Riemann-based, direction-invariant "
                                       "slip/no-penatration boundary conditions"));
      PetscCall(PetscOptionsBCDefinition(flags[j], "Face IDs to apply symmetry BC", NULL, "symmetry", &bc_def, NULL));
      if (!bc_def) {
        PetscCall(PetscOptionsBCDefinition(deprecated[j], "Face IDs to apply symmetry BC", NULL, "symmetry", &bc_def, NULL));
      }
      PetscCall(AddBCDefinitionToSegBuffer(bc_def, bc_defs_seg));
      if (bc_def) {
        PetscInt essential_comps[1] = {j + 1};

        PetscCall(BCDefinitionSetEssential(bc_def, 1, essential_comps));
      }
    }
  }

  // Inflow BCs
  bc->num_inflow = 16;
  PetscCall(PetscOptionsIntArray("-bc_inflow", "Face IDs to apply inflow BC", NULL, bc->inflows, &bc->num_inflow, NULL));
  // Outflow BCs
  bc->num_outflow = 16;
  PetscCall(PetscOptionsIntArray("-bc_outflow", "Face IDs to apply outflow BC", NULL, bc->outflows, &bc->num_outflow, NULL));
  // Freestream BCs
  bc->num_freestream = 16;
  PetscCall(PetscOptionsIntArray("-bc_freestream", "Face IDs to apply freestream BC", NULL, bc->freestreams, &bc->num_freestream, NULL));

  bc->num_slip = 16;
  PetscCall(PetscOptionsIntArray("-bc_slip", "Face IDs to apply slip BC", NULL, bc->slips, &bc->num_slip, NULL));

  PetscOptionsEnd();

  PetscCall(PetscSegBufferGetSize(bc_defs_seg, &problem->num_bc_defs));
  PetscCall(PetscSegBufferExtractAlloc(bc_defs_seg, &problem->bc_defs));
  PetscCall(PetscSegBufferDestroy(&bc_defs_seg));

  //TODO: Verify that the BCDefinition don't have overlapping claims to boundary faces

  PetscFunctionReturn(PETSC_SUCCESS);
}
