/// @file
/// Helper functions for solid mechanics example using PETSc

#include "../include/misc.h"
#include "../include/utils.h"

// -----------------------------------------------------------------------------
// Create libCEED operator context
// -----------------------------------------------------------------------------
// Setup context data for Jacobian evaluation
PetscErrorCode SetupJacobianCtx(MPI_Comm comm, AppCtx app_ctx, DM dm, Vec V,
                                Vec V_loc, CeedData ceed_data, Ceed ceed,
                                CeedQFunctionContext ctx_phys,
                                CeedQFunctionContext ctx_phys_smoother,
                                UserMult jacobian_ctx) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // PETSc objects
  jacobian_ctx->comm = comm;
  jacobian_ctx->dm = dm;

  // Work vectors
  jacobian_ctx->X_loc = V_loc;
  ierr = VecDuplicate(V_loc, &jacobian_ctx->Y_loc); CHKERRQ(ierr);
  jacobian_ctx->x_ceed = ceed_data->x_ceed;
  jacobian_ctx->y_ceed = ceed_data->y_ceed;

  // libCEED operator
  jacobian_ctx->op = ceed_data->op_jacobian;
  jacobian_ctx->qf = ceed_data->qf_jacobian;

  // Ceed
  jacobian_ctx->ceed = ceed;

  // Physics
  jacobian_ctx->ctx_phys = ctx_phys;
  jacobian_ctx->ctx_phys_smoother = ctx_phys_smoother;
  PetscFunctionReturn(0);
};

// Setup context data for prolongation and restriction operators
PetscErrorCode SetupProlongRestrictCtx(MPI_Comm comm, AppCtx app_ctx, DM dm_c,
                                       DM dm_f, Vec V_f, Vec V_loc_c, Vec V_loc_f,
                                       CeedData ceed_data_c, CeedData ceed_data_f,
                                       Ceed ceed, UserMultProlongRestr prolong_restr_ctx) {
  PetscFunctionBeginUser;

  // PETSc objects
  prolong_restr_ctx->comm = comm;
  prolong_restr_ctx->dm_c = dm_c;
  prolong_restr_ctx->dm_f = dm_f;

  // Work vectors
  prolong_restr_ctx->loc_vec_c = V_loc_c;
  prolong_restr_ctx->loc_vec_f = V_loc_f;
  prolong_restr_ctx->ceed_vec_c = ceed_data_c->x_ceed;
  prolong_restr_ctx->ceed_vec_f = ceed_data_f->x_ceed;

  // libCEED operators
  prolong_restr_ctx->op_prolong = ceed_data_f->op_prolong;
  prolong_restr_ctx->op_restrict = ceed_data_f->op_restrict;

  // Ceed
  prolong_restr_ctx->ceed = ceed;
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Jacobian setup
// -----------------------------------------------------------------------------
PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat J_pre, void *ctx) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Context data
  FormJacobCtx  form_jacob_ctx = (FormJacobCtx)ctx;
  PetscInt      num_levels = form_jacob_ctx->num_levels;
  Mat           *jacob_mat = form_jacob_ctx->jacob_mat;

  // Update Jacobian on each level
  for (PetscInt level = 0; level < num_levels; level++) {
    ierr = MatAssemblyBegin(jacob_mat[level], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jacob_mat[level], MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  // Form coarse assembled matrix
  ierr = VecZeroEntries(form_jacob_ctx->u_coarse); CHKERRQ(ierr);
  ierr = SNESComputeJacobianDefaultColor(form_jacob_ctx->snes_coarse,
                                         form_jacob_ctx->u_coarse,
                                         form_jacob_ctx->jacob_mat[0],
                                         form_jacob_ctx->jacob_mat_coarse, NULL);
  CHKERRQ(ierr);

  // J_pre might be AIJ (e.g., when using coloring), so we need to assemble it
  ierr = MatAssemblyBegin(J_pre, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J_pre, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (J != J_pre) {
    ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Output solution for visualization
// -----------------------------------------------------------------------------
PetscErrorCode ViewSolution(MPI_Comm comm, AppCtx app_ctx, Vec U,
                            PetscInt increment, PetscScalar load_increment) {
  PetscErrorCode ierr;
  DM dm;
  PetscViewer viewer;
  char output_filename[PETSC_MAX_PATH_LEN];
  PetscMPIInt rank;

  PetscFunctionBeginUser;

  // Create output directory
  MPI_Comm_rank(comm, &rank);
  if (!rank) {ierr = PetscMkdir(app_ctx->output_dir); CHKERRQ(ierr);}

  // Build file name
  ierr = PetscSNPrintf(output_filename, sizeof output_filename,
                       "%s/solution-%03D.vtu", app_ctx->output_dir,
                       increment); CHKERRQ(ierr);

  // Increment sequence
  ierr = VecGetDM(U, &dm); CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, increment, load_increment); CHKERRQ(ierr);

  // Output solution vector
  ierr = PetscViewerVTKOpen(comm, output_filename, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = VecView(U, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Output diagnostic quantities for visualization
// -----------------------------------------------------------------------------
PetscErrorCode ViewDiagnosticQuantities(MPI_Comm comm, DM dmU,
                                        UserMult user, AppCtx app_ctx, Vec U,
                                        CeedElemRestriction elem_restr_diagnostic) {
  PetscErrorCode ierr;
  Vec Diagnostic, Y_loc, mult_vec;
  CeedVector y_ceed;
  CeedScalar *x, *y;
  PetscMemType x_mem_type, y_mem_type;
  PetscInt loc_size;
  PetscViewer viewer;
  char output_filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;

  // ---------------------------------------------------------------------------
  // PETSc and libCEED vectors
  // ---------------------------------------------------------------------------
  ierr = DMCreateGlobalVector(user->dm, &Diagnostic); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Diagnostic, ""); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(user->dm, &Y_loc); CHKERRQ(ierr);
  ierr = VecGetSize(Y_loc, &loc_size); CHKERRQ(ierr);
  CeedVectorCreate(user->ceed, loc_size, &y_ceed);

  // ---------------------------------------------------------------------------
  // Compute quantities
  // ---------------------------------------------------------------------------
  // -- Global-to-local
  ierr = VecZeroEntries(user->X_loc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dmU, PETSC_TRUE, user->X_loc,
                                    user->load_increment, NULL, NULL, NULL);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dmU, U, INSERT_VALUES, user->X_loc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Y_loc); CHKERRQ(ierr);

  // -- Setup CEED vectors
  ierr = VecGetArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x,
                                   &x_mem_type); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(Y_loc, &y, &y_mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // -- Apply CEED operator
  CeedOperatorApply(user->op, user->x_ceed, y_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Restore PETSc vector; keep y_ceed viewing memory of Y_loc for use below
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(x_mem_type), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x);
  CHKERRQ(ierr);

  // -- Local-to-global
  ierr = VecZeroEntries(Diagnostic); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Y_loc, ADD_VALUES, Diagnostic);
  CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Scale for multiplicity
  // ---------------------------------------------------------------------------
  // -- Setup vectors
  ierr = VecDuplicate(Diagnostic, &mult_vec); CHKERRQ(ierr);
  ierr = VecZeroEntries(Y_loc); CHKERRQ(ierr);

  // -- Compute multiplicity
  CeedElemRestrictionGetMultiplicity(elem_restr_diagnostic, y_ceed);

  // -- Restore vectors
  CeedVectorTakeArray(y_ceed, MemTypeP2C(y_mem_type), NULL);
  ierr = VecRestoreArrayAndMemType(Y_loc, &y); CHKERRQ(ierr);

  // -- Local-to-global
  ierr = VecZeroEntries(mult_vec); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Y_loc, ADD_VALUES, mult_vec);
  CHKERRQ(ierr);

  // -- Scale
  ierr = VecReciprocal(mult_vec); CHKERRQ(ierr);
  ierr = VecPointwiseMult(Diagnostic, Diagnostic, mult_vec);

  // ---------------------------------------------------------------------------
  // Output solution vector
  // ---------------------------------------------------------------------------
  ierr = PetscSNPrintf(output_filename, sizeof output_filename,
                       "%s/diagnostic_quantities.vtu",
                       app_ctx->output_dir); CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm, output_filename, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = VecView(Diagnostic, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------
  ierr = VecDestroy(&Diagnostic); CHKERRQ(ierr);
  ierr = VecDestroy(&mult_vec); CHKERRQ(ierr);
  ierr = VecDestroy(&Y_loc); CHKERRQ(ierr);
  CeedVectorDestroy(&y_ceed);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Regression testing
// -----------------------------------------------------------------------------
// test option change. could remove the loading step. Run only with one loading step and compare relatively to ref file
// option: expect_final_strain_energy and check against the relative error to ref is within tolerance (10^-5) I.e. one Newton solve then check final energy
PetscErrorCode RegressionTests_solids(AppCtx app_ctx, PetscReal energy) {
  PetscFunctionBegin;

  if (app_ctx->expect_final_strain >= 0.) {
    PetscReal energy_ref = app_ctx->expect_final_strain;
    PetscReal error = PetscAbsReal(energy - energy_ref) / energy_ref;

    if (error > app_ctx->test_tol) {
      PetscErrorCode ierr;
      ierr = PetscPrintf(PETSC_COMM_WORLD,
                         "Energy %e does not match expected energy %e: relative tolerance %e > %e\n",
                         (double)energy, (double)energy_ref, (double)error, app_ctx->test_tol);
      CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
};
