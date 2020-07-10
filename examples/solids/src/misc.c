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
/// Helper functions for solid mechanics example using PETSc

#include "../elasticity.h"

// -----------------------------------------------------------------------------
// Create libCEED operator context
// -----------------------------------------------------------------------------
// Setup context data for Jacobian evaluation
PetscErrorCode SetupJacobianCtx(MPI_Comm comm, AppCtx appCtx, DM dm, Vec V,
                                Vec Vloc, CeedData ceedData, Ceed ceed,
                                Physics phys, Physics physSmoother,
                                UserMult jacobianCtx) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // PETSc objects
  jacobianCtx->comm = comm;
  jacobianCtx->dm = dm;

  // Work vectors
  jacobianCtx->Xloc = Vloc;
  ierr = VecDuplicate(Vloc, &jacobianCtx->Yloc); CHKERRQ(ierr);
  jacobianCtx->Xceed = ceedData->xceed;
  jacobianCtx->Yceed = ceedData->yceed;

  // libCEED operator
  jacobianCtx->op = ceedData->opJacob;
  jacobianCtx->qf = ceedData->qfJacob;

  // Ceed
  jacobianCtx->ceed = ceed;

  // Physics
  jacobianCtx->phys = phys;
  jacobianCtx->physSmoother = physSmoother;

  // Get/Restore Array
  jacobianCtx->memType = appCtx->memTypeRequested;
  if (appCtx->memTypeRequested == CEED_MEM_HOST) {
    jacobianCtx->VecGetArray = VecGetArray;
    jacobianCtx->VecGetArrayRead = VecGetArrayRead;
    jacobianCtx->VecRestoreArray = VecRestoreArray;
    jacobianCtx->VecRestoreArrayRead = VecRestoreArrayRead;
  } else {
    jacobianCtx->VecGetArray = VecCUDAGetArray;
    jacobianCtx->VecGetArrayRead = VecCUDAGetArrayRead;
    jacobianCtx->VecRestoreArray = VecCUDARestoreArray;
    jacobianCtx->VecRestoreArrayRead = VecCUDARestoreArrayRead;
  }

  PetscFunctionReturn(0);
};

// Setup context data for prolongation and restriction operators
PetscErrorCode SetupProlongRestrictCtx(MPI_Comm comm, AppCtx appCtx, DM dmC,
                                       DM dmF, Vec VF, Vec VlocC, Vec VlocF,
                                       CeedData ceedDataC, CeedData ceedDataF,
                                       Ceed ceed,
                                       UserMultProlongRestr prolongRestrCtx) {
  PetscFunctionBeginUser;

  // PETSc objects
  prolongRestrCtx->comm = comm;
  prolongRestrCtx->dmC = dmC;
  prolongRestrCtx->dmF = dmF;

  // Work vectors
  prolongRestrCtx->locVecC = VlocC;
  prolongRestrCtx->locVecF = VlocF;
  prolongRestrCtx->ceedVecC = ceedDataC->xceed;
  prolongRestrCtx->ceedVecF = ceedDataF->xceed;

  // libCEED operators
  prolongRestrCtx->opProlong = ceedDataF->opProlong;
  prolongRestrCtx->opRestrict = ceedDataF->opRestrict;

  // Ceed
  prolongRestrCtx->ceed = ceed;

  // Get/Restore Array
  prolongRestrCtx->memType = appCtx->memTypeRequested;
  if (appCtx->memTypeRequested == CEED_MEM_HOST) {
    prolongRestrCtx->VecGetArray = VecGetArray;
    prolongRestrCtx->VecGetArrayRead = VecGetArrayRead;
    prolongRestrCtx->VecRestoreArray = VecRestoreArray;
    prolongRestrCtx->VecRestoreArrayRead = VecRestoreArrayRead;
  } else {
    prolongRestrCtx->VecGetArray = VecCUDAGetArray;
    prolongRestrCtx->VecGetArrayRead = VecCUDAGetArrayRead;
    prolongRestrCtx->VecRestoreArray = VecCUDARestoreArray;
    prolongRestrCtx->VecRestoreArrayRead = VecCUDARestoreArrayRead;
  }

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Jacobian setup
// -----------------------------------------------------------------------------
PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat Jpre, void *ctx) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Context data
  FormJacobCtx  formJacobCtx = (FormJacobCtx)ctx;
  PetscInt      numLevels = formJacobCtx->numLevels;
  Mat           *jacobMat = formJacobCtx->jacobMat;

  // Update Jacobian on each level
  for (PetscInt level = 0; level < numLevels; level++) {
    ierr = MatAssemblyBegin(jacobMat[level], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jacobMat[level], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  // Form coarse assembled matrix
  ierr = VecZeroEntries(formJacobCtx->Ucoarse); CHKERRQ(ierr);
  ierr = SNESComputeJacobianDefaultColor(formJacobCtx->snesCoarse,
                                         formJacobCtx->Ucoarse,
                                         formJacobCtx->jacobMat[0],
                                         formJacobCtx->jacobMatCoarse, NULL);
  CHKERRQ(ierr);

  // Jpre might be AIJ (e.g., when using coloring), so we need to assemble it
  ierr = MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Output solution for visualization
// -----------------------------------------------------------------------------
PetscErrorCode ViewSolution(MPI_Comm comm, Vec U, PetscInt increment,
                            PetscScalar loadIncrement) {
  PetscErrorCode ierr;
  DM dm;
  PetscViewer viewer;
  char outputFilename[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;

  // Build file name
  ierr = PetscSNPrintf(outputFilename, sizeof outputFilename,
                       "solution-%03D.vtu", increment); CHKERRQ(ierr);

  // Increment sequence
  ierr = VecGetDM(U, &dm); CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, increment, loadIncrement); CHKERRQ(ierr);

  // Output solution vector
  ierr = PetscViewerVTKOpen(comm, outputFilename, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = VecView(U, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Output diagnostic quantities for visualization
// -----------------------------------------------------------------------------
PetscErrorCode ViewDiagnosticQuantities(MPI_Comm comm, DM dmU,
                                        UserMult user, Vec U,
                                        CeedElemRestriction ErestrictDiagnostic) {
  PetscErrorCode ierr;
  Vec Diagnostic, Yloc, MultVec;
  CeedVector Yceed;
  CeedScalar *x, *y;
  PetscInt lsz;
  PetscViewer viewer;
  const char *outputFilename = "diagnostic_quantities.vtu";

  PetscFunctionBeginUser;

  // ---------------------------------------------------------------------------
  // PETSc and libCEED vectors
  // ---------------------------------------------------------------------------
  ierr = DMCreateGlobalVector(user->dm, &Diagnostic); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Diagnostic, ""); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(user->dm, &Yloc); CHKERRQ(ierr);
  ierr = VecGetSize(Yloc, &lsz); CHKERRQ(ierr);
  CeedVectorCreate(user->ceed, lsz, &Yceed);

  // ---------------------------------------------------------------------------
  // Compute quantities
  // ---------------------------------------------------------------------------
  // -- Global-to-local
  ierr = VecZeroEntries(user->Xloc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dmU, PETSC_TRUE, user->Xloc,
                                    user->loadIncrement, NULL, NULL, NULL);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dmU, U, INSERT_VALUES, user->Xloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Yloc); CHKERRQ(ierr);

  // -- Setup CEED vectors
  ierr = user->VecGetArrayRead(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = user->VecGetArray(Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, user->memType, CEED_USE_POINTER, x);
  CeedVectorSetArray(Yceed, user->memType, CEED_USE_POINTER, y);

  // -- Apply CEED operator
  CeedOperatorApply(user->op, user->Xceed, Yceed, CEED_REQUEST_IMMEDIATE);

  // -- Restore PETSc vector
  CeedVectorTakeArray(user->Xceed, user->memType, NULL);
  ierr = user->VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);

  // -- Local-to-global
  ierr = VecZeroEntries(Diagnostic); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Yloc, ADD_VALUES, Diagnostic);
  CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Scale for multiplicity
  // ---------------------------------------------------------------------------
  // -- Setup vectors
  ierr = VecDuplicate(Diagnostic, &MultVec); CHKERRQ(ierr);
  ierr = VecZeroEntries(Yloc); CHKERRQ(ierr);

  // -- Compute multiplicity
  CeedElemRestrictionGetMultiplicity(ErestrictDiagnostic, Yceed);

  // -- Restore vectors
  CeedVectorTakeArray(Yceed, user->memType, NULL);
  ierr = user->VecRestoreArray(Yloc, &y); CHKERRQ(ierr);

  // -- Local-to-global
  ierr = VecZeroEntries(MultVec); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Yloc, ADD_VALUES, MultVec);
  CHKERRQ(ierr);

  // -- Scale
  ierr = VecReciprocal(MultVec); CHKERRQ(ierr);
  ierr = VecPointwiseMult(Diagnostic, Diagnostic, MultVec);

  // ---------------------------------------------------------------------------
  // Output solution vector
  // ---------------------------------------------------------------------------
  ierr = PetscViewerVTKOpen(comm, outputFilename, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = VecView(Diagnostic, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------
  ierr = VecDestroy(&Diagnostic); CHKERRQ(ierr);
  ierr = VecDestroy(&MultVec); CHKERRQ(ierr);
  ierr = VecDestroy(&Yloc); CHKERRQ(ierr);
  CeedVectorDestroy(&Yceed);

  PetscFunctionReturn(0);
};
