#include "../navierstokes.h"

// -----------------------------------------------------------------------------
// Miscellaneous utility functions
// -----------------------------------------------------------------------------

int VectorPlacePetscVec(CeedVector c, Vec p) {

  PetscInt m_ceed, mpetsc;
  PetscScalar *a;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = CeedVectorGetLength(c, &m_ceed); CHKERRQ(ierr);
  ierr = VecGetLocalSize(p, &mpetsc); CHKERRQ(ierr);
  if (m_ceed != mpetsc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                                  "Cannot place PETSc Vec of length %D in CeedVector of length %D",
                                  mpetsc, m_ceed);
  ierr = VecGetArray(p, &a); CHKERRQ(ierr);
  CeedVectorSetArray(c, CEED_MEM_HOST, CEED_USE_POINTER, a);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm,
    PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM,
    Vec cell_geom_FVM, Vec grad_FVM) {

  Vec Qbc;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMGetNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
  ierr = VecAXPY(Q_loc, 1., Qbc); CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTests_NS(AppCtx app_ctx, Vec Q) {

  Vec Qref;
  PetscViewer viewer;
  PetscReal error, Qrefnorm;
  PetscErrorCode ierr;

  PetscFunctionBegin;

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

  // Clean up objects
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = VecDestroy(&Qref); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Get error for problems with exact solutions
PetscErrorCode GetError_NS(CeedData ceed_data, DM dm, AppCtx app_ctx, Vec Q,
                           PetscScalar final_time) {

  PetscInt lnodes;
  Vec Qexact, Qexactloc;
  PetscReal rel_error, norm_error, norm_exact;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get exact solution at final time
  ierr = DMCreateGlobalVector(dm, &Qexact); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &Qexactloc); CHKERRQ(ierr);
  ierr = VecGetSize(Qexactloc, &lnodes); CHKERRQ(ierr);
  ierr = ICs_FixMultiplicity(ceed_data->op_ics, ceed_data->x_corners,
                             ceed_data->q0_ceed, dm, Qexactloc, Qexact,
                             ceed_data->elem_restr_q, ceed_data->setup_context, final_time); CHKERRQ(ierr);

  // Get |exact solution - obtained solution|
  ierr = VecNorm(Qexact, NORM_1, &norm_exact); CHKERRQ(ierr);
  ierr = VecAXPY(Q, -1.0, Qexact);  CHKERRQ(ierr);
  ierr = VecNorm(Q, NORM_1, &norm_error); CHKERRQ(ierr);

  // Compute relative error
  rel_error = norm_error / norm_exact;

  // Output relative error
  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "Relative Error: %g\n",
                     (double)rel_error); CHKERRQ(ierr);
  // Clean up vectors
  CeedVectorDestroy(&ceed_data->q0_ceed);
  ierr = DMRestoreLocalVector(dm, &Qexactloc); CHKERRQ(ierr);
  ierr = VecDestroy(&Qexact); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Post-processing
PetscErrorCode PostProcess_NS(TS ts, CeedData ceed_data, DM dm,
                              ProblemData *problem, AppCtx app_ctx,
                              Vec Q, PetscScalar final_time) {
  PetscInt steps;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Print relative error
  if (problem->non_zero_time && !app_ctx->test_mode) {
    ierr = GetError_NS(ceed_data, dm, app_ctx, Q, final_time); CHKERRQ(ierr);
  }

  // Print final time and number of steps
  ierr = TSGetStepNumber(ts, &steps); CHKERRQ(ierr);
  if (!app_ctx->test_mode) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time integrator took %D time steps to reach final time %g\n",
                       steps, (double)final_time); CHKERRQ(ierr);
  }

  // Output numerical values from command line
  ierr = VecViewFromOptions(Q, NULL, "-vec_view"); CHKERRQ(ierr);

  // Compare reference solution values with current test run for CI
  if (app_ctx->test_mode) {
    ierr = RegressionTests_NS(app_ctx, Q); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// -- Gather initial Q values in case of continuation of simulation
PetscErrorCode SetupICsFromBinary(MPI_Comm comm, AppCtx app_ctx, Vec Q) {

  PetscViewer viewer;
  char file_path[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Read input
  ierr = PetscSNPrintf(file_path, sizeof file_path, "%s/ns-solution.bin",
                       app_ctx->output_dir); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm, file_path, FILE_MODE_READ, &viewer);
  CHKERRQ(ierr);

  // Load Q from existent solution
  ierr = VecLoad(Q, viewer); CHKERRQ(ierr);

  // Clean up PETSc object
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Record boundary values from initial condition
PetscErrorCode SetBCsFromICs_NS(DM dm, Vec Q, Vec Q_loc) {

  Vec Qbc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
  ierr = VecCopy(Q_loc, Qbc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Q_loc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm, Q, INSERT_VALUES, Q_loc); CHKERRQ(ierr);
  ierr = VecAXPY(Qbc, -1., Q_loc); CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,
                                    "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_NS);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
