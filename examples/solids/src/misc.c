// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Helper functions for solid mechanics example using PETSc

#include "../include/misc.h"

#include "../include/utils.h"

// -----------------------------------------------------------------------------
// Create libCEED operator context
// -----------------------------------------------------------------------------
// Setup context data for Jacobian evaluation
PetscErrorCode SetupJacobianCtx(MPI_Comm comm, AppCtx app_ctx, DM dm, Vec V, Vec V_loc, CeedData ceed_data, Ceed ceed, CeedQFunctionContext ctx_phys,
                                CeedQFunctionContext ctx_phys_smoother, UserMult jacobian_ctx) {
  PetscFunctionBeginUser;

  // PETSc objects
  jacobian_ctx->comm = comm;
  jacobian_ctx->dm   = dm;

  // Work vectors
  jacobian_ctx->X_loc = V_loc;
  PetscCall(VecDuplicate(V_loc, &jacobian_ctx->Y_loc));
  jacobian_ctx->x_ceed = ceed_data->x_ceed;
  jacobian_ctx->y_ceed = ceed_data->y_ceed;

  // libCEED operator
  jacobian_ctx->op = ceed_data->op_jacobian;
  jacobian_ctx->qf = ceed_data->qf_jacobian;

  // Ceed
  jacobian_ctx->ceed = ceed;

  // Physics
  jacobian_ctx->ctx_phys          = ctx_phys;
  jacobian_ctx->ctx_phys_smoother = ctx_phys_smoother;
  PetscFunctionReturn(0);
};

// Setup context data for prolongation and restriction operators
PetscErrorCode SetupProlongRestrictCtx(MPI_Comm comm, AppCtx app_ctx, DM dm_c, DM dm_f, Vec V_f, Vec V_loc_c, Vec V_loc_f, CeedData ceed_data_c,
                                       CeedData ceed_data_f, Ceed ceed, UserMultProlongRestr prolong_restr_ctx) {
  PetscFunctionBeginUser;

  // PETSc objects
  prolong_restr_ctx->comm = comm;
  prolong_restr_ctx->dm_c = dm_c;
  prolong_restr_ctx->dm_f = dm_f;

  // Work vectors
  prolong_restr_ctx->loc_vec_c  = V_loc_c;
  prolong_restr_ctx->loc_vec_f  = V_loc_f;
  prolong_restr_ctx->ceed_vec_c = ceed_data_c->x_ceed;
  prolong_restr_ctx->ceed_vec_f = ceed_data_f->x_ceed;

  // libCEED operators
  prolong_restr_ctx->op_prolong  = ceed_data_f->op_prolong;
  prolong_restr_ctx->op_restrict = ceed_data_f->op_restrict;

  // Ceed
  prolong_restr_ctx->ceed = ceed;
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Jacobian setup
// -----------------------------------------------------------------------------
PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat J_pre, void *ctx) {
  PetscFunctionBeginUser;

  // Context data
  FormJacobCtx form_jacob_ctx = (FormJacobCtx)ctx;
  PetscInt     num_levels     = form_jacob_ctx->num_levels;
  Mat         *jacob_mat      = form_jacob_ctx->jacob_mat;

  // Update Jacobian on each level
  for (PetscInt level = 0; level < num_levels; level++) {
    PetscCall(MatAssemblyBegin(jacob_mat[level], MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(jacob_mat[level], MAT_FINAL_ASSEMBLY));
  }

  // Form coarse assembled matrix
  CeedOperatorLinearAssemble(form_jacob_ctx->op_coarse, form_jacob_ctx->coo_values);
  const CeedScalar *values;
  CeedVectorGetArrayRead(form_jacob_ctx->coo_values, CEED_MEM_HOST, &values);
  PetscCall(MatSetValuesCOO(form_jacob_ctx->jacob_mat_coarse, values, ADD_VALUES));
  CeedVectorRestoreArrayRead(form_jacob_ctx->coo_values, &values);

  // J_pre might be AIJ (e.g., when using coloring), so we need to assemble it
  PetscCall(MatAssemblyBegin(J_pre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J_pre, MAT_FINAL_ASSEMBLY));
  if (J != J_pre) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Output solution for visualization
// -----------------------------------------------------------------------------
PetscErrorCode ViewSolution(MPI_Comm comm, AppCtx app_ctx, Vec U, PetscInt increment, PetscScalar load_increment) {
  DM          dm;
  PetscViewer viewer;
  char        output_filename[PETSC_MAX_PATH_LEN];
  PetscMPIInt rank;

  PetscFunctionBeginUser;

  // Create output directory
  MPI_Comm_rank(comm, &rank);
  if (!rank) {
    PetscCall(PetscMkdir(app_ctx->output_dir));
  }

  // Build file name
  PetscCall(PetscSNPrintf(output_filename, sizeof output_filename, "%s/solution-%03" PetscInt_FMT ".vtu", app_ctx->output_dir, increment));

  // Increment sequence
  PetscCall(VecGetDM(U, &dm));
  PetscCall(DMSetOutputSequenceNumber(dm, increment, load_increment));

  // Output solution vector
  PetscCall(PetscViewerVTKOpen(comm, output_filename, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(U, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Output diagnostic quantities for visualization
// -----------------------------------------------------------------------------
PetscErrorCode ViewDiagnosticQuantities(MPI_Comm comm, DM dmU, UserMult user, AppCtx app_ctx, Vec U, CeedElemRestriction elem_restr_diagnostic) {
  Vec          Diagnostic, Y_loc, mult_vec;
  CeedVector   y_ceed;
  CeedScalar  *x, *y;
  PetscMemType x_mem_type, y_mem_type;
  PetscInt     loc_size;
  PetscViewer  viewer;
  char         output_filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;

  // ---------------------------------------------------------------------------
  // PETSc and libCEED vectors
  // ---------------------------------------------------------------------------
  PetscCall(DMCreateGlobalVector(user->dm, &Diagnostic));
  PetscCall(PetscObjectSetName((PetscObject)Diagnostic, ""));
  PetscCall(DMCreateLocalVector(user->dm, &Y_loc));
  PetscCall(VecGetSize(Y_loc, &loc_size));
  CeedVectorCreate(user->ceed, loc_size, &y_ceed);

  // ---------------------------------------------------------------------------
  // Compute quantities
  // ---------------------------------------------------------------------------
  // -- Global-to-local
  PetscCall(VecZeroEntries(user->X_loc));
  PetscCall(DMPlexInsertBoundaryValues(dmU, PETSC_TRUE, user->X_loc, user->load_increment, NULL, NULL, NULL));
  PetscCall(DMGlobalToLocal(dmU, U, INSERT_VALUES, user->X_loc));
  PetscCall(VecZeroEntries(Y_loc));

  // -- Setup CEED vectors
  PetscCall(VecGetArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x, &x_mem_type));
  PetscCall(VecGetArrayAndMemType(Y_loc, &y, &y_mem_type));
  CeedVectorSetArray(user->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // -- Apply CEED operator
  CeedOperatorApply(user->op, user->x_ceed, y_ceed, CEED_REQUEST_IMMEDIATE);

  // -- Restore PETSc vector; keep y_ceed viewing memory of Y_loc for use below
  CeedVectorTakeArray(user->x_ceed, MemTypeP2C(x_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(user->X_loc, (const PetscScalar **)&x));

  // -- Local-to-global
  PetscCall(VecZeroEntries(Diagnostic));
  PetscCall(DMLocalToGlobal(user->dm, Y_loc, ADD_VALUES, Diagnostic));

  // ---------------------------------------------------------------------------
  // Scale for multiplicity
  // ---------------------------------------------------------------------------
  // -- Setup vectors
  PetscCall(VecDuplicate(Diagnostic, &mult_vec));
  PetscCall(VecZeroEntries(Y_loc));

  // -- Compute multiplicity
  CeedElemRestrictionGetMultiplicity(elem_restr_diagnostic, y_ceed);

  // -- Restore vectors
  CeedVectorTakeArray(y_ceed, MemTypeP2C(y_mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(Y_loc, &y));

  // -- Local-to-global
  PetscCall(VecZeroEntries(mult_vec));
  PetscCall(DMLocalToGlobal(user->dm, Y_loc, ADD_VALUES, mult_vec));

  // -- Scale
  PetscCall(VecReciprocal(mult_vec));
  PetscCall(VecPointwiseMult(Diagnostic, Diagnostic, mult_vec));

  // ---------------------------------------------------------------------------
  // Output solution vector
  // ---------------------------------------------------------------------------
  PetscCall(PetscSNPrintf(output_filename, sizeof output_filename, "%s/diagnostic_quantities.vtu", app_ctx->output_dir));
  PetscCall(PetscViewerVTKOpen(comm, output_filename, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(Diagnostic, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------
  PetscCall(VecDestroy(&Diagnostic));
  PetscCall(VecDestroy(&mult_vec));
  PetscCall(VecDestroy(&Y_loc));
  CeedVectorDestroy(&y_ceed);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Regression testing
// -----------------------------------------------------------------------------
// test option change. could remove the loading step. Run only with one loading step and compare relatively to ref file
// option: expect_final_strain_energy and check against the relative error to ref is within tolerance (10^-5) I.e. one Newton solve then check final
// energy
PetscErrorCode RegressionTests_solids(AppCtx app_ctx, PetscReal energy) {
  PetscFunctionBegin;

  if (app_ctx->expect_final_strain >= 0.) {
    PetscReal energy_ref = app_ctx->expect_final_strain;
    PetscReal error      = PetscAbsReal(energy - energy_ref) / energy_ref;

    if (error > app_ctx->test_tol) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Energy %e does not match expected energy %e: relative tolerance %e > %e\n", (double)energy,
                            (double)energy_ref, (double)error, app_ctx->test_tol));
    }
  }
  PetscFunctionReturn(0);
};
