#include "../navierstokes.h"

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
// Read mesh and distribute DM in parallel
PetscErrorCode CreateDistributedDM(MPI_Comm comm, problemData *problem,
                                   SetupContext setup_ctx, DM *dm) {

  PetscErrorCode   ierr;
  DM               distributed_mesh = NULL;
  PetscPartitioner part;
  PetscInt         dim = problem->dim;
  const PetscReal  scale[3] = {setup_ctx->lx, setup_ctx->ly, setup_ctx->lz};

  PetscFunctionBeginUser;

  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, NULL, NULL, scale,
                             NULL, PETSC_TRUE, dm); CHKERRQ(ierr);

  // Distribute DM in parallel
  ierr = DMPlexGetPartitioner(*dm, &part); CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
  ierr = DMPlexDistribute(*dm, 0, NULL, &distributed_mesh); CHKERRQ(ierr);
  if (distributed_mesh) {
    ierr = DMDestroy(dm); CHKERRQ(ierr);
    *dm  = distributed_mesh;
  }
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view"); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Setup DM
PetscErrorCode SetUpDM(DM dm, problemData *problem, PetscInt degree,
                       SimpleBC bc, Physics phys, void *ctxSetupData) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  {
    // Configure the finite element space and boundary conditions
    PetscFE fe;
    PetscInt ncompq = 5;
    ierr = PetscFECreateLagrange(PETSC_COMM_SELF, problem->dim, ncompq,
                                 PETSC_FALSE, degree, PETSC_DECIDE,
                                 &fe); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe, "Q"); CHKERRQ(ierr);
    ierr = DMAddField(dm, NULL,(PetscObject)fe); CHKERRQ(ierr);
    ierr = DMCreateDS(dm); CHKERRQ(ierr);
    ierr = problem->bc_fnc(dm, bc, phys, ctxSetupData);
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
PetscErrorCode VizRefineDM(DM dm, User user, problemData *problem,
                           SimpleBC bc, Physics phys, void *ctxSetupData) {
  PetscErrorCode ierr;
  DM dmhierarchy[user->app_ctx->viz_refine + 1];

  PetscFunctionBeginUser;
  ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE); CHKERRQ(ierr);

  dmhierarchy[0] = dm;
  for (PetscInt i = 0, d = user->app_ctx->degree;
       i < user->app_ctx->viz_refine; i++) {
    Mat interp_next;
    ierr = DMRefine(dmhierarchy[i], MPI_COMM_NULL, &dmhierarchy[i+1]);
    CHKERRQ(ierr);
    ierr = DMClearDS(dmhierarchy[i+1]); CHKERRQ(ierr);
    ierr = DMClearFields(dmhierarchy[i+1]); CHKERRQ(ierr);
    ierr = DMSetCoarseDM(dmhierarchy[i+1], dmhierarchy[i]); CHKERRQ(ierr);
    d = (d + 1) / 2;
    if (i + 1 == user->app_ctx->viz_refine) d = 1;
    ierr = SetUpDM(dmhierarchy[i+1], problem, d, bc, phys, ctxSetupData);
    CHKERRQ(ierr);
    ierr = DMCreateInterpolation(dmhierarchy[i], dmhierarchy[i+1], &interp_next,
                                 NULL); CHKERRQ(ierr);
    if (!i) user->interpviz = interp_next;
    else {
      Mat C;
      ierr = MatMatMult(interp_next, user->interpviz, MAT_INITIAL_MATRIX,
                        PETSC_DECIDE, &C); CHKERRQ(ierr);
      ierr = MatDestroy(&interp_next); CHKERRQ(ierr);
      ierr = MatDestroy(&user->interpviz); CHKERRQ(ierr);
      user->interpviz = C;
    }
  }
  for (PetscInt i=1; i<user->app_ctx->viz_refine; i++) {
    ierr = DMDestroy(&dmhierarchy[i]); CHKERRQ(ierr);
  }
  user->dmviz = dmhierarchy[user->app_ctx->viz_refine];

  PetscFunctionReturn(0);
}
