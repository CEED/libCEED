#include "../navierstokes.h"

// -----------------------------------------------------------------------------
// Miscellaneous utility functions
// -----------------------------------------------------------------------------
PetscErrorCode ICs_FixMultiplicity(CeedOperator op_ics, CeedVector x_corners,
                                   CeedVector q0_ceed, DM dm, Vec Q_loc, Vec Q,
                                   CeedElemRestriction elem_restr_q,
                                   CeedQFunctionContext setup_context, CeedScalar time) {
  CeedVector     mult_vec;
  Vec            multiplicity, multiplicity_loc;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  SetupContext setup_ctx;
  CeedQFunctionContextGetData(setup_context, CEED_MEM_HOST, (void **)&setup_ctx);
  setup_ctx->time = time;
  CeedQFunctionContextRestoreData(setup_context, (void **)&setup_ctx);

  CeedVector *q0;
  PetscMemType q0_mem_type;
  ierr = VecGetArrayAndMemType(Q_loc, &q0, &q0_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(q0_ceed, MemTypeP2C(q0_mem_type), CEED_USE_POINTER, q0);
  CHKERRQ(ierr);

  CeedOperatorApply(op_ics, x_corners, q0_ceed, CEED_REQUEST_IMMEDIATE);
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, Q_loc, ADD_VALUES, Q); CHKERRQ(ierr);

  // CEED Restriction
  CeedElemRestrictionCreateVector(elem_restr_q, &mult_vec, NULL);

  // Fix multiplicity for output of ICs
  CeedVector *m;
  PetscMemType m_mem_type;
  ierr = DMGetLocalVector(dm, &multiplicity_loc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(multiplicity_loc, &m, &m_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(mult_vec, MemTypeP2C(m_mem_type), CEED_USE_POINTER, m);
  CHKERRQ(ierr);

  CeedElemRestrictionGetMultiplicity(elem_restr_q, mult_vec);
  CeedVectorDestroy(&mult_vec);  // todo: takearray?
  ierr = DMGetGlobalVector(dm, &multiplicity); CHKERRQ(ierr);
  ierr = VecZeroEntries(multiplicity); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, multiplicity_loc, ADD_VALUES, multiplicity);
  CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Q, Q, multiplicity); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Q_loc, Q_loc, multiplicity_loc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &multiplicity_loc); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &multiplicity); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm,
    PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM,
    Vec cell_geom_FVM, Vec grad_FVM) {

  Vec            Qbc;
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
  PetscInt       loc_nodes;
  Vec            Q_exact, Q_exact_loc;
  PetscReal      rel_error, norm_error, norm_exact;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Get exact solution at final time
  ierr = DMCreateGlobalVector(dm, &Q_exact); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &Q_exact_loc); CHKERRQ(ierr);
  ierr = VecGetSize(Q_exact_loc, &loc_nodes); CHKERRQ(ierr);
  ierr = ICs_FixMultiplicity(ceed_data->op_ics, ceed_data->x_corners,
                             ceed_data->q0_ceed, dm, Q_exact_loc, Q_exact,
                             ceed_data->elem_restr_q, ceed_data->setup_context, final_time); CHKERRQ(ierr);

  // Get |exact solution - obtained solution|
  ierr = VecNorm(Q_exact, NORM_1, &norm_exact); CHKERRQ(ierr);
  ierr = VecAXPY(Q, -1.0, Q_exact);  CHKERRQ(ierr);
  ierr = VecNorm(Q, NORM_1, &norm_error); CHKERRQ(ierr);

  // Compute relative error
  rel_error = norm_error / norm_exact;

  // Output relative error
  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "Relative Error: %g\n",
                     (double)rel_error); CHKERRQ(ierr);
  // Clean up vectors
  CeedVectorDestroy(&ceed_data->q0_ceed);
  ierr = DMRestoreLocalVector(dm, &Q_exact_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&Q_exact); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Post-processing
PetscErrorCode PostProcess_NS(TS ts, CeedData ceed_data, DM dm,
                              ProblemData *problem, AppCtx app_ctx,
                              Vec Q, PetscScalar final_time) {
  PetscInt       steps;
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

  PetscViewer    viewer;
  char           file_path[PETSC_MAX_PATH_LEN];
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

  Vec            Qbc;
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
