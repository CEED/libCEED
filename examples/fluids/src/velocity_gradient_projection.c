// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
/// @file
/// Functions for setting up and projecting the velocity gradient

#include "../qfunctions/velocity_gradient_projection.h"

#include "../navierstokes.h"

PetscErrorCode VelocityGradientProjectionCreateDM(NodalProjectionData grad_velo_proj, User user, ProblemData *problem, PetscInt degree) {
  PetscFE      fe;
  PetscSection section;
  PetscInt     dim;

  PetscFunctionBeginUser;
  grad_velo_proj->num_comp = 9;  // 9 velocity gradient

  PetscCall(DMClone(user->dm, &grad_velo_proj->dm));
  PetscCall(DMGetDimension(grad_velo_proj->dm, &dim));
  PetscCall(PetscObjectSetName((PetscObject)grad_velo_proj->dm, "Velocity Gradient Projection"));

  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, grad_velo_proj->num_comp, PETSC_FALSE, degree, PETSC_DECIDE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "Velocity Gradient Projection"));
  PetscCall(DMAddField(grad_velo_proj->dm, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(grad_velo_proj->dm));
  PetscCall(DMPlexSetClosurePermutationTensor(grad_velo_proj->dm, PETSC_DETERMINE, NULL));

  PetscCall(DMGetLocalSection(grad_velo_proj->dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "VelocityGradientXX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "VelocityGradientXY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "VelocityGradientXZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "VelocityGradientYX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "VelocityGradientYY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 5, "VelocityGradientYZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 6, "VelocityGradientZX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 7, "VelocityGradientZY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 8, "VelocityGradientZZ"));

  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
};
