// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up statistics collection

#include "../navierstokes.h"

PetscErrorCode CreateStatsDM(User user, ProblemData *problem, PetscInt degree, SimpleBC bc) {
  user->spanstats.num_comp_stats = 1;
  PetscFunctionBeginUser;

  // Get DM from surface
  {
    DMLabel label;
    PetscCall(DMGetLabel(user->dm, "Face Sets", &label));
    PetscCall(DMPlexLabelComplete(user->dm, label));
    PetscCall(DMPlexFilter(user->dm, label, 1, &user->spanstats.dm));
    PetscCall(DMProjectCoordinates(user->spanstats.dm, NULL));  // Ensure that a coordinate FE exists
  }

  PetscCall(PetscObjectSetName((PetscObject)user->spanstats.dm, "Spanwise_Stats"));
  PetscCall(DMSetOptionsPrefix(user->spanstats.dm, "spanstats_"));
  PetscCall(PetscOptionsSetValue(NULL, "-spanstats_dm_sparse_localize", "0"));  // [Jed] Not relevant because not periodic in this direction

  PetscCall(DMSetFromOptions(user->spanstats.dm));
  PetscCall(DMViewFromOptions(user->spanstats.dm, NULL, "-dm_view"));  // -spanstats_dm_view (option includes prefix)
  {
    PetscFE fe;
    DMLabel label;

    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, problem->dim - 1, user->spanstats.num_comp_stats, PETSC_FALSE, degree, PETSC_DECIDE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "stats"));
    PetscCall(DMAddField(user->spanstats.dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(user->spanstats.dm));
    PetscCall(DMGetLabel(user->spanstats.dm, "Face Sets", &label));

    // // Set wall BCs
    // if (bc->num_wall > 0) {
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, bc->num_wall, bc->walls, 0, bc->num_comps, bc->wall_comps,
    //                           (void (*)(void))problem->bc, NULL, problem->bc_ctx, NULL));
    // }
    // // Set slip BCs in the x direction
    // if (bc->num_slip[0] > 0) {
    //   PetscInt comps[1] = {1};
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", label, bc->num_slip[0], bc->slips[0], 0, 1, comps, (void (*)(void))NULL, NULL,
    //                           problem->bc_ctx, NULL));
    // }
    // // Set slip BCs in the y direction
    // if (bc->num_slip[1] > 0) {
    //   PetscInt comps[1] = {2};
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", label, bc->num_slip[1], bc->slips[1], 0, 1, comps, (void (*)(void))NULL, NULL,
    //                           problem->bc_ctx, NULL));
    // }
    // // Set slip BCs in the z direction
    // if (bc->num_slip[2] > 0) {
    //   PetscInt comps[1] = {3};
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", label, bc->num_slip[2], bc->slips[2], 0, 1, comps, (void (*)(void))NULL, NULL,
    //                           problem->bc_ctx, NULL));
    // }

    PetscCall(DMPlexSetClosurePermutationTensor(user->spanstats.dm, PETSC_DETERMINE, NULL));
    PetscCall(PetscFEDestroy(&fe));
  }

  PetscSection section;
  PetscCall(DMGetLocalSection(user->spanstats.dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "Test"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 0, "Mean Velocity Products XX"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 1, "Mean Velocity Products YY"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 2, "Mean Velocity Products ZZ"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 3, "Mean Velocity Products YZ"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 4, "Mean Velocity Products XZ"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 5, "Mean Velocity Products XY"));

  // Vec test;
  // PetscCall(DMCreateLocalVector(user->spanstats.dm, &test));
  // PetscCall(VecZeroEntries(test));
  // PetscCall(VecViewFromOptions(test, NULL, "-test_view"));

  PetscFunctionReturn(0);
}
