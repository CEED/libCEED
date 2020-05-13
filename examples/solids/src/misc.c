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

  // Ceed
  jacobianCtx->ceed = ceed;

  PetscFunctionReturn(0);
};

// Setup context data for prolongation and restriction operators
PetscErrorCode SetupProlongRestrictCtx(MPI_Comm comm, DM dmC, DM dmF, Vec VF,
                                       Vec VlocC, Vec VlocF, CeedData ceedDataC,
                                       CeedData ceedDataF, Ceed ceed,
                                       UserMultProlongRestr prolongRestrCtx) {
  PetscErrorCode ierr;
  PetscScalar *multArray;

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

  // Multiplicity vector
  // -- Set libCEED vector
  ierr = VecZeroEntries(VlocF);
  ierr = VecGetArray(VlocF, &multArray); CHKERRQ(ierr);
  CeedVectorSetArray(ceedDataF->xceed, CEED_MEM_HOST, CEED_USE_POINTER,
                     multArray);

  // -- Compute multiplicity
  CeedElemRestrictionGetMultiplicity(ceedDataF->Erestrictu, ceedDataF->xceed);

  // -- Restore PETSc vector
  ierr = VecRestoreArray(VlocF, &multArray); CHKERRQ(ierr);

  // -- Local-to-global
  ierr = VecZeroEntries(VF); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dmF, VlocF, ADD_VALUES, VF); CHKERRQ(ierr);

  // -- Global-to-local
  ierr = VecDuplicate(VlocF, &prolongRestrCtx->multVec); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dmF, VF, INSERT_VALUES, prolongRestrCtx->multVec);
  CHKERRQ(ierr);

  // -- Reciprocal
  ierr = VecReciprocal(prolongRestrCtx->multVec); CHKERRQ(ierr);

  // -- Reset work arrays
  ierr = VecZeroEntries(VF); CHKERRQ(ierr);
  ierr = VecZeroEntries(VlocF); CHKERRQ(ierr);

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
  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecGetArray(Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->Xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorSetArray(Yceed, CEED_MEM_HOST, CEED_USE_POINTER, y);

  // -- Apply CEED operator
  CeedOperatorApply(user->op, user->Xceed, Yceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorSyncArray(Yceed, CEED_MEM_HOST);

  // -- Restore PETSc vectors
  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x);
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
  ierr = VecRestoreArray(Yloc, &y); CHKERRQ(ierr);

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
