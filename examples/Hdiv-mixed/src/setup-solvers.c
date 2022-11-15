#include "../include/setup-solvers.h"
#include "../include/setup-matops.h"
#include "../include/setup-libceed.h"
#include "petscvec.h"

// -----------------------------------------------------------------------------
// Setup operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupJacobianOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data,
                                        VecType vec_type,
                                        OperatorApplyContext ctx_jacobian) {
  PetscFunctionBeginUser;

  ctx_jacobian->dm = dm;
  PetscCall( DMCreateLocalVector(dm, &ctx_jacobian->X_loc) );
  PetscCall( VecDuplicate(ctx_jacobian->X_loc, &ctx_jacobian->Y_loc) );
  ctx_jacobian->x_ceed = ceed_data->x_ceed;
  ctx_jacobian->y_ceed = ceed_data->y_ceed;
  ctx_jacobian->ceed = ceed;
  ctx_jacobian->op_apply = ceed_data->op_jacobian;
  ctx_jacobian->vec_type = vec_type;
  PetscFunctionReturn(0);
}

PetscErrorCode SetupResidualOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data,
                                        OperatorApplyContext ctx_residual) {
  PetscFunctionBeginUser;

  ctx_residual->dm = dm;
  PetscCall( DMCreateLocalVector(dm, &ctx_residual->X_loc) );
  PetscCall( VecDuplicate(ctx_residual->X_loc, &ctx_residual->Y_loc) );
  ctx_residual->x_ceed = ceed_data->x_ceed;
  ctx_residual->y_ceed = ceed_data->y_ceed;
  ctx_residual->ceed = ceed;
  ctx_residual->op_apply = ceed_data->op_residual;

  PetscFunctionReturn(0);
}

PetscErrorCode SetupErrorOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data,
                                     OperatorApplyContext ctx_error) {
  PetscFunctionBeginUser;

  ctx_error->dm = dm;
  PetscCall( DMCreateLocalVector(dm, &ctx_error->X_loc) );
  PetscCall( VecDuplicate(ctx_error->X_loc, &ctx_error->Y_loc) );
  ctx_error->x_ceed = ceed_data->x_ceed;
  ctx_error->y_ceed = ceed_data->y_ceed;
  ctx_error->ceed = ceed;
  ctx_error->op_apply = ceed_data->op_error;

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// This function wraps the libCEED operator for a MatShell
// -----------------------------------------------------------------------------
PetscErrorCode ApplyMatOp(Mat A, Vec X, Vec Y) {
  OperatorApplyContext op_apply_ctx;

  PetscFunctionBeginUser;

  PetscCall( MatShellGetContext(A, &op_apply_ctx) );

  // libCEED for local action of residual evaluator
  PetscCall( ApplyLocalCeedOp(X, Y, op_apply_ctx) );

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function uses libCEED to compute the non-linear residual
// -----------------------------------------------------------------------------
PetscErrorCode SNESFormResidual(SNES snes, Vec X, Vec Y, void *ctx_residual) {
  OperatorApplyContext ctx = (OperatorApplyContext)ctx_residual;

  PetscFunctionBeginUser;

  // Use computed BCs
  //PetscCall( DMPlexInsertBoundaryValues(ctx->dm, PETSC_TRUE,
  //                                      ctx->X_loc,
  //                                      1.0, NULL, NULL, NULL) );

  // libCEED for local action of residual evaluator
  PetscCall( ApplyLocalCeedOp(X, Y, ctx) );

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Jacobian setup
// -----------------------------------------------------------------------------
PetscErrorCode SNESFormJacobian(SNES snes, Vec U, Mat J, Mat J_pre,
                                void *ctx_jacobian) {
  OperatorApplyContext ctx = (OperatorApplyContext)ctx_jacobian;
  PetscFunctionBeginUser;

  Mat A;
  PetscCall(DMCreateMatrix(ctx->dm, &A));
  // Assemble matrix analytically
  PetscCount num_entries;
  CeedInt *rows, *cols;
  CeedVector coo_values;
  CeedOperatorLinearAssembleSymbolic(ctx->op_apply, &num_entries, &rows,
                                     &cols);
  PetscCall(MatSetPreallocationCOO(A, num_entries, rows, cols));
  free(rows);
  free(cols);
  CeedVectorCreate(ctx->ceed, num_entries, &coo_values);
  CeedOperatorLinearAssemble(ctx->op_apply, coo_values);
  const CeedScalar *values;
  CeedVectorGetArrayRead(coo_values, CEED_MEM_HOST, &values);
  PetscCall(MatSetValuesCOO(A, values, ADD_VALUES));
  CeedVectorRestoreArrayRead(coo_values, &values);
  MatView(A, PETSC_VIEWER_STDOUT_WORLD);
  //CeedVectorView(coo_values, "%12.8f", stdout);
  CeedVectorDestroy(&coo_values);
  PetscCall( MatDestroy(&A) );

  // J_pre might be AIJ (e.g., when using coloring), so we need to assemble it
  PetscCall( MatAssemblyBegin(J_pre, MAT_FINAL_ASSEMBLY) );
  PetscCall( MatAssemblyEnd(J_pre, MAT_FINAL_ASSEMBLY) );
  if (J != J_pre) {
    PetscCall( MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY) );
    PetscCall( MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY) );
  }
  PetscFunctionReturn(0);
};

// ---------------------------------------------------------------------------
// Setup Solver
// ---------------------------------------------------------------------------
PetscErrorCode PDESolver(CeedData ceed_data, AppCtx app_ctx,
                         SNES snes, KSP ksp, Vec *U) {

  PetscInt       U_l_size, U_g_size;

  PetscFunctionBeginUser;

  // Create global unknown solution U
  PetscCall( VecGetSize(*U, &U_g_size) );
  // Local size for matShell
  PetscCall( VecGetLocalSize(*U, &U_l_size) );
  Vec R;
  PetscCall( VecDuplicate(*U, &R) );

  // ---------------------------------------------------------------------------
  // Setup SNES
  // ---------------------------------------------------------------------------
  // Operator
  Mat mat_jacobian;
  PetscCall( SNESSetDM(snes, app_ctx->ctx_jacobian->dm) );
  // -- Form Action of Jacobian on delta_u
  PetscCall( MatCreateShell(app_ctx->comm, U_l_size, U_l_size, U_g_size,
                            U_g_size, app_ctx->ctx_jacobian, &mat_jacobian) );
  PetscCall( MatShellSetOperation(mat_jacobian, MATOP_MULT,
                                  (void (*)(void))ApplyMatOp) );
  PetscCall( MatShellSetVecType(mat_jacobian, app_ctx->ctx_jacobian->vec_type) );

  // Set SNES residual evaluation function
  PetscCall( SNESSetFunction(snes, R, SNESFormResidual,
                             app_ctx->ctx_residual) );
  // -- SNES Jacobian
  PetscCall( SNESSetJacobian(snes, mat_jacobian, mat_jacobian,
                             SNESFormJacobian, app_ctx->ctx_jacobian) );

  // Setup KSP
  PetscCall( KSPSetFromOptions(ksp) );

  // Default to critical-point (CP) line search (related to Wolfe's curvature condition)
  SNESLineSearch line_search;

  PetscCall( SNESGetLineSearch(snes, &line_search) );
  PetscCall( SNESLineSearchSetType(line_search, SNESLINESEARCHCP) );
  PetscCall( SNESSetFromOptions(snes) );

  // Solve
  PetscCall( VecSet(*U, 0.0));
  PetscCall( SNESSolve(snes, NULL, *U));

  // Free PETSc objects
  PetscCall( MatDestroy(&mat_jacobian) );
  PetscCall( VecDestroy(&R) );

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function calculates the L2 error in the final solution
// -----------------------------------------------------------------------------
PetscErrorCode ComputeL2Error(CeedData ceed_data, AppCtx app_ctx, Vec U,
                              CeedScalar *l2_error_u, CeedScalar *l2_error_p) {
  PetscScalar *x;
  PetscMemType mem_type;
  CeedVector collocated_error;

  PetscFunctionBeginUser;

  CeedInt dim, num_elem, num_qpts;
  PetscCall( DMGetDimension(app_ctx->ctx_error->dm, &dim) );
  CeedBasisGetNumQuadraturePoints(ceed_data->basis_u, &num_qpts);
  num_elem = ceed_data->num_elem;
  CeedVectorCreate(app_ctx->ctx_error->ceed, num_elem*num_qpts*(dim+1),
                   &collocated_error);

  // Global-to-local
  PetscCall( DMGlobalToLocal(app_ctx->ctx_error->dm, U, INSERT_VALUES,
                             app_ctx->ctx_error->X_loc) );

  // Setup CEED vector
  PetscCall( VecGetArrayAndMemType(app_ctx->ctx_error->X_loc, &x, &mem_type) );
  CeedVectorSetArray(app_ctx->ctx_error->x_ceed, MemTypeP2C(mem_type),
                     CEED_USE_POINTER,
                     x);

  // Apply CEED operator
  CeedOperatorApply(app_ctx->ctx_error->op_apply, app_ctx->ctx_error->x_ceed,
                    collocated_error,
                    CEED_REQUEST_IMMEDIATE);
  // Restore PETSc vector
  CeedVectorTakeArray(app_ctx->ctx_error->x_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall( VecRestoreArrayReadAndMemType(app_ctx->ctx_error->X_loc,
             (const PetscScalar **)&x) );
  // Compute L2 error for each field
  CeedInt cent_qpts = num_qpts / 2;
  CeedVector collocated_error_u, collocated_error_p;
  const CeedScalar *E_U; // to store total error
  CeedInt length_u, length_p;
  length_p = num_elem;
  length_u = num_elem*num_qpts*dim;
  CeedScalar e_u[length_u], e_p[length_p];
  CeedVectorCreate(app_ctx->ctx_error->ceed, length_p, &collocated_error_p);
  CeedVectorCreate(app_ctx->ctx_error->ceed, length_u, &collocated_error_u);
  // E_U is ordered as [p_0,u_0/.../p_n,u_n] for 0 to n elements
  // For each element p_0 size is num_qpts, and u_0 is dim*num_qpts
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &E_U);
  for (CeedInt n=0; n < num_elem; n++) {
    for (CeedInt i=0; i < 1; i++) {
      CeedInt j = i + n*1;
      CeedInt k = cent_qpts + n*num_qpts*(dim+1);
      e_p[j] = E_U[k];
    }
  }

  for (CeedInt n=0; n < num_elem; n++) {
    for (CeedInt i=0; i < dim*num_qpts; i++) {
      CeedInt j = i + n*num_qpts*dim;
      CeedInt k = num_qpts + i + n*num_qpts*(dim+1);
      e_u[j] = E_U[k];
    }
  }

  CeedVectorSetArray(collocated_error_p, CEED_MEM_HOST, CEED_USE_POINTER, e_p);
  CeedVectorSetArray(collocated_error_u, CEED_MEM_HOST, CEED_USE_POINTER, e_u);
  CeedVectorRestoreArrayRead(collocated_error, &E_U);

  CeedScalar error_u, error_p;
  CeedVectorNorm(collocated_error_u, CEED_NORM_1, &error_u);
  CeedVectorNorm(collocated_error_p, CEED_NORM_1, &error_p);
  *l2_error_u = sqrt(error_u);
  *l2_error_p = sqrt(error_p);
  // Cleanup
  CeedVectorDestroy(&collocated_error);
  CeedVectorDestroy(&collocated_error_u);
  CeedVectorDestroy(&collocated_error_p);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
