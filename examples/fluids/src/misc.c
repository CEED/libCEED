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
