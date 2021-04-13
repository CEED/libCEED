#include "../navierstokes.h"

// -----------------------------------------------------------------------------
// Miscellaneous utility functions
// -----------------------------------------------------------------------------

int VectorPlacePetscVec(CeedVector c, Vec p) {
  PetscErrorCode ierr;
  PetscInt mceed, mpetsc;
  PetscScalar *a;

  PetscFunctionBeginUser;
  ierr = CeedVectorGetLength(c, &mceed); CHKERRQ(ierr);
  ierr = VecGetLocalSize(p, &mpetsc); CHKERRQ(ierr);
  if (mceed != mpetsc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                                  "Cannot place PETSc Vec of length %D in CeedVector of length %D",
                                  mpetsc, mceed);
  ierr = VecGetArray(p, &a); CHKERRQ(ierr);
  CeedVectorSetArray(c, CEED_MEM_HOST, CEED_USE_POINTER, a);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm,
    PetscBool insertEssential, Vec Qloc, PetscReal time, Vec faceGeomFVM,
    Vec cellGeomFVM, Vec gradFVM) {
  PetscErrorCode ierr;
  Vec Qbc;

  PetscFunctionBegin;
  ierr = DMGetNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
  ierr = VecAXPY(Qloc, 1., Qbc); CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTests_NS(AppCtx app_ctx, Vec Q) {

  PetscErrorCode ierr;

  Vec Qref;
  PetscViewer viewer;
  PetscReal error, Qrefnorm;

  // Read reference file
  ierr = VecDuplicate(Q, &Qref); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)Q),
                               app_ctx->file_path, FILE_MODE_READ,
                               &viewer); CHKERRQ(ierr);
  ierr = VecLoad(Qref, viewer); CHKERRQ(ierr);

  // Compute error with respect to reference solution
  ierr = VecAXPY(Q, -1.0, Qref);  CHKERRQ(ierr);
  ierr = VecNorm(Qref, NORM_MAX, &Qrefnorm); CHKERRQ(ierr);
  ierr = VecScale(Q, 1./Qrefnorm); CHKERRQ(ierr);
  ierr = VecNorm(Q, NORM_MAX, &error); CHKERRQ(ierr);

  // Check error
  if (error > app_ctx->test_tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Test failed with error norm %g\n",
                       (double)error); CHKERRQ(ierr);
  }

  // Destroy PETSc objects
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = VecDestroy(&Qref); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
