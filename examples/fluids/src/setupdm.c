// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Setup DM for Navier-Stokes example using PETSc

#include "../navierstokes.h"

// Read mesh and distribute DM in parallel
PetscErrorCode CreateDistributedDM(MPI_Comm comm, ProblemData *problem,
                                   SetupContext setup_ctx, DM *dm) {
  DM               dist_mesh = NULL;
  PetscPartitioner part;
  PetscInt         dim = problem->dim, faces[3] = {3, 3, 3};
  const PetscReal  scale[3] = {setup_ctx->lx, setup_ctx->ly, setup_ctx->lz};
  PetscErrorCode   ierr;
  PetscFunctionBeginUser;

  ierr = PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces",
                                 faces, &dim, NULL); CHKERRQ(ierr);
  if (!dim) dim = problem->dim;
  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, NULL, scale,
                             NULL, PETSC_TRUE, dm); CHKERRQ(ierr);

  // Distribute DM in parallel
  ierr = DMPlexGetPartitioner(*dm, &part); CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
  ierr = DMPlexDistribute(*dm, 0, NULL, &dist_mesh); CHKERRQ(ierr);
  if (dist_mesh) {
    ierr = DMDestroy(dm); CHKERRQ(ierr);
    *dm  = dist_mesh;
  }
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view"); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Setup DM
PetscErrorCode SetUpDM(DM dm, ProblemData *problem, PetscInt degree,
                       SimpleBC bc, Physics phys, void *setup_ctx) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  {
    // Configure the finite element space and boundary conditions
    PetscFE  fe;
    PetscInt num_comp_q = 5;
    ierr = PetscFECreateLagrange(PETSC_COMM_SELF, problem->dim, num_comp_q,
                                 PETSC_FALSE, degree, PETSC_DECIDE,
                                 &fe); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe, "Q"); CHKERRQ(ierr);
    ierr = DMAddField(dm, NULL,(PetscObject)fe); CHKERRQ(ierr);
    ierr = DMCreateDS(dm); CHKERRQ(ierr);
    ierr = problem->bc_func(dm, bc, phys, setup_ctx);
    ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);
    CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);
  }
  {
    // Empty name for conserved field (because there is only one field)
    PetscSection section;
    ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, 0, ""); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 0, "Density");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 1, "MomentumX");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 2, "MomentumY");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 3, "MomentumZ");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 4, "EnergyDensity");
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// Refine DM for high-order viz
PetscErrorCode VizRefineDM(DM dm, User user, ProblemData *problem,
                           SimpleBC bc, Physics phys, void *setup_ctx) {
  PetscErrorCode ierr;
  DM             dm_hierarchy[user->app_ctx->viz_refine + 1];
  VecType        vec_type;
  PetscFunctionBeginUser;

  ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE); CHKERRQ(ierr);

  dm_hierarchy[0] = dm;
  for (PetscInt i = 0, d = user->app_ctx->degree;
       i < user->app_ctx->viz_refine; i++) {
    Mat interp_next;
    ierr = DMRefine(dm_hierarchy[i], MPI_COMM_NULL, &dm_hierarchy[i+1]);
    CHKERRQ(ierr);
    ierr = DMClearDS(dm_hierarchy[i+1]); CHKERRQ(ierr);
    ierr = DMClearFields(dm_hierarchy[i+1]); CHKERRQ(ierr);
    ierr = DMSetCoarseDM(dm_hierarchy[i+1], dm_hierarchy[i]); CHKERRQ(ierr);
    d = (d + 1) / 2;
    if (i + 1 == user->app_ctx->viz_refine) d = 1;
    ierr = DMGetVecType(dm, &vec_type); CHKERRQ(ierr);
    ierr = DMSetVecType(dm_hierarchy[i+1], vec_type); CHKERRQ(ierr);
    ierr = SetUpDM(dm_hierarchy[i+1], problem, d, bc, phys, setup_ctx);
    CHKERRQ(ierr);
    ierr = DMCreateInterpolation(dm_hierarchy[i], dm_hierarchy[i+1], &interp_next,
                                 NULL); CHKERRQ(ierr);
    if (!i) user->interp_viz = interp_next;
    else {
      Mat C;
      ierr = MatMatMult(interp_next, user->interp_viz, MAT_INITIAL_MATRIX,
                        PETSC_DECIDE, &C); CHKERRQ(ierr);
      ierr = MatDestroy(&interp_next); CHKERRQ(ierr);
      ierr = MatDestroy(&user->interp_viz); CHKERRQ(ierr);
      user->interp_viz = C;
    }
  }
  for (PetscInt i=1; i<user->app_ctx->viz_refine; i++) {
    ierr = DMDestroy(&dm_hierarchy[i]); CHKERRQ(ierr);
  }
  user->dm_viz = dm_hierarchy[user->app_ctx->viz_refine];

  PetscFunctionReturn(0);
}
