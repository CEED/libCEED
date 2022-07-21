#include "../include/setup-solvers.h"
#include "../include/setup-matops.h"
#include "../include/setup-libceed.h"

// -----------------------------------------------------------------------------
// Setup operator context data
// -----------------------------------------------------------------------------
PetscErrorCode SetupJacobianOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data,
                                        OperatorApplyContext ctx_jacobian) {
  PetscFunctionBeginUser;

  ctx_jacobian->dm = dm;
  PetscCall( DMCreateLocalVector(dm, &ctx_jacobian->X_loc) );
  PetscCall( VecDuplicate(ctx_jacobian->X_loc, &ctx_jacobian->Y_loc) );
  ctx_jacobian->x_ceed = ceed_data->x_ceed;
  ctx_jacobian->y_ceed = ceed_data->y_ceed;
  ctx_jacobian->ceed = ceed;
  ctx_jacobian->op_apply = ceed_data->op_jacobian;

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

  PetscFunctionBeginUser;

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
PetscErrorCode PDESolver(MPI_Comm comm, DM dm, Ceed ceed, CeedData ceed_data,
                         VecType vec_type, SNES snes, KSP ksp, Vec *U_g) {

  PetscInt       U_l_size, U_g_size;

  PetscFunctionBeginUser;

  // Create global unknown solution U_g
  PetscCall( DMCreateGlobalVector(dm, U_g) );
  PetscCall( VecGetSize(*U_g, &U_g_size) );
  // Local size for matShell
  PetscCall( VecGetLocalSize(*U_g, &U_l_size) );
  Vec R;
  PetscCall( VecDuplicate(*U_g, &R) );

  // ---------------------------------------------------------------------------
  // Setup SNES
  // ---------------------------------------------------------------------------
  // Operator
  Mat mat_jacobian;
  PetscCall( PetscCalloc1(1, &ceed_data->ctx_jacobian) );
  SetupJacobianOperatorCtx(dm, ceed, ceed_data, ceed_data->ctx_jacobian);
  PetscCall( SNESSetDM(snes, ceed_data->ctx_jacobian->dm) );
  // -- Form Action of Jacobian on delta_u
  PetscCall( MatCreateShell(comm, U_l_size, U_l_size, U_g_size,
                            U_g_size, ceed_data->ctx_jacobian, &mat_jacobian) );
  PetscCall( MatShellSetOperation(mat_jacobian, MATOP_MULT,
                                  (void (*)(void))ApplyMatOp) );
  PetscCall( MatShellSetVecType(mat_jacobian, vec_type) );

  // Set SNES residual evaluation function
  PetscCall( PetscCalloc1(1, &ceed_data->ctx_residual) );
  SetupResidualOperatorCtx(dm, ceed, ceed_data, ceed_data->ctx_residual);
  PetscCall( SNESSetFunction(snes, R, SNESFormResidual,
                             ceed_data->ctx_residual) );
  // -- SNES Jacobian
  PetscCall( SNESSetJacobian(snes, mat_jacobian, mat_jacobian,
                             SNESFormJacobian, ceed_data->ctx_jacobian) );

  // Setup KSP
  PetscCall( KSPSetFromOptions(ksp) );

  // Default to critical-point (CP) line search (related to Wolfe's curvature condition)
  SNESLineSearch line_search;

  PetscCall( SNESGetLineSearch(snes, &line_search) );
  PetscCall( SNESLineSearchSetType(line_search, SNESLINESEARCHCP) );
  PetscCall( SNESSetFromOptions(snes) );

  // Solve
  PetscCall( VecSet(*U_g, 0.0));
  PetscCall( SNESSolve(snes, NULL, *U_g));

  // Free PETSc objects
  PetscCall( MatDestroy(&mat_jacobian) );
  PetscCall( VecDestroy(&R) );
  PetscCall( VecDestroy(&ceed_data->ctx_jacobian->Y_loc) );
  PetscCall( VecDestroy(&ceed_data->ctx_jacobian->X_loc) );
  PetscCall( VecDestroy(&ceed_data->ctx_residual->Y_loc) );
  PetscCall( VecDestroy(&ceed_data->ctx_residual->X_loc) );
  PetscCall( PetscFree(ceed_data->ctx_jacobian) );
  PetscCall( PetscFree(ceed_data->ctx_residual) );

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function calculates the L2 error in the final solution
// -----------------------------------------------------------------------------
PetscErrorCode ComputeL2Error(DM dm, Ceed ceed, CeedData ceed_data, Vec U,
                              CeedScalar *l2_error_u,
                              CeedScalar *l2_error_p) {
  PetscScalar *x;
  PetscMemType mem_type;
  CeedVector collocated_error;

  PetscFunctionBeginUser;

  PetscCall( PetscCalloc1(1, &ceed_data->ctx_error) );
  SetupErrorOperatorCtx(dm, ceed, ceed_data, ceed_data->ctx_error);
  CeedInt c_start, c_end, dim, num_elem, num_qpts;
  PetscCall( DMGetDimension(ceed_data->ctx_error->dm, &dim) );
  CeedBasisGetNumQuadraturePoints(ceed_data->basis_u, &num_qpts);
  PetscCall( DMPlexGetHeightStratum(ceed_data->ctx_error->dm, 0, &c_start,
                                    &c_end) );
  num_elem = c_end -c_start;
  CeedVectorCreate(ceed, num_elem*num_qpts*(dim+1), &collocated_error);

  // Global-to-local
  PetscCall( DMGlobalToLocal(ceed_data->ctx_error->dm, U, INSERT_VALUES,
                             ceed_data->ctx_error->X_loc) );

  // Setup CEED vector
  PetscCall( VecGetArrayAndMemType(ceed_data->ctx_error->X_loc, &x, &mem_type) );
  CeedVectorSetArray(ceed_data->ctx_error->x_ceed, MemTypeP2C(mem_type),
                     CEED_USE_POINTER,
                     x);

  // Apply CEED operator
  CeedOperatorApply(ceed_data->ctx_error->op_apply, ceed_data->ctx_error->x_ceed,
                    collocated_error,
                    CEED_REQUEST_IMMEDIATE);
  // Restore PETSc vector
  CeedVectorTakeArray(ceed_data->ctx_error->x_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall( VecRestoreArrayReadAndMemType(ceed_data->ctx_error->X_loc,
             (const PetscScalar **)&x) );
  // Compute L2 error for each field
  CeedInt cent_qpts = num_qpts / 2;
  CeedVector collocated_error_u, collocated_error_p;
  const CeedScalar *E_U; // to store total error
  CeedInt length_u, length_p;
  length_p = num_elem;
  length_u = num_elem*num_qpts*dim;
  CeedScalar e_u[length_u], e_p[length_p];
  CeedVectorCreate(ceed_data->ctx_error->ceed, length_p, &collocated_error_p);
  CeedVectorCreate(ceed_data->ctx_error->ceed, length_u, &collocated_error_u);
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
  PetscCall( VecDestroy(&ceed_data->ctx_error->Y_loc) );
  PetscCall( VecDestroy(&ceed_data->ctx_error->X_loc) );
  PetscCall( PetscFree(ceed_data->ctx_error) );

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function print the output
// -----------------------------------------------------------------------------
PetscErrorCode PrintOutput(Ceed ceed, AppCtx app_ctx, PetscBool has_ts,
                           CeedMemType mem_type_backend,
                           TS ts, SNES snes, KSP ksp,
                           Vec U, CeedScalar l2_error_u, CeedScalar l2_error_p) {

  PetscFunctionBeginUser;

  const char *used_resource;
  CeedGetResource(ceed, &used_resource);
  char hostname[PETSC_MAX_PATH_LEN];
  PetscCall( PetscGetHostName(hostname, sizeof hostname) );
  PetscInt comm_size;
  PetscCall( MPI_Comm_size(app_ctx->comm, &comm_size) );
  PetscCall( PetscPrintf(app_ctx->comm,
                         "\n-- Mixed H(div) Example - libCEED + PETSc --\n"
                         "  MPI:\n"
                         "    Hostname                           : %s\n"
                         "    Total ranks                        : %d\n"
                         "  libCEED:\n"
                         "    libCEED Backend                    : %s\n"
                         "    libCEED Backend MemType            : %s\n",
                         hostname, comm_size, used_resource, CeedMemTypes[mem_type_backend]) );

  VecType vecType;
  PetscCall( VecGetType(U, &vecType) );
  PetscCall( PetscPrintf(app_ctx->comm,
                         "  PETSc:\n"
                         "    PETSc Vec Type                     : %s\n",
                         vecType) );

  PetscInt       U_l_size, U_g_size;
  PetscCall( VecGetSize(U, &U_g_size) );
  PetscCall( VecGetLocalSize(U, &U_l_size) );
  PetscCall( PetscPrintf(app_ctx->comm,
                         "  Problem:\n"
                         "    Problem Name                       : %s\n"
                         "    Global nodes (u + p)               : %" PetscInt_FMT "\n"
                         "    Owned nodes (u + p)                : %" PetscInt_FMT "\n",
                         app_ctx->problem_name, U_g_size, U_l_size
                        ) );
  // --TS
  if (has_ts) {
    PetscInt ts_steps;
    TSType ts_type;
    TSConvergedReason ts_reason;
    PetscCall( TSGetStepNumber(ts, &ts_steps) );
    PetscCall( TSGetType(ts, &ts_type) );
    PetscCall( TSGetConvergedReason(ts, &ts_reason) );
    PetscCall( PetscPrintf(app_ctx->comm,
                           "  TS:\n"
                           "    TS Type                            : %s\n"
                           "    TS Convergence                     : %s\n"
                           "    Number of TS steps                 : %" PetscInt_FMT "\n"
                           "    Final time                         : %g\n",
                           ts_type, TSConvergedReasons[ts_reason],
                           ts_steps, (double)app_ctx->t_final) );

    PetscCall( TSGetSNES(ts, &snes) );
  }
  // -- SNES
  PetscInt its, snes_its = 0;
  PetscCall( SNESGetIterationNumber(snes, &its) );
  snes_its += its;
  SNESType snes_type;
  SNESConvergedReason snes_reason;
  PetscReal snes_rnorm;
  PetscCall( SNESGetType(snes, &snes_type) );
  PetscCall( SNESGetConvergedReason(snes, &snes_reason) );
  PetscCall( SNESGetFunctionNorm(snes, &snes_rnorm) );
  PetscCall( PetscPrintf(app_ctx->comm,
                         "  SNES:\n"
                         "    SNES Type                          : %s\n"
                         "    SNES Convergence                   : %s\n"
                         "    Total SNES Iterations              : %" PetscInt_FMT "\n"
                         "    Final rnorm                        : %e\n",
                         snes_type, SNESConvergedReasons[snes_reason],
                         snes_its, (double)snes_rnorm) );
  if (!has_ts) {
    PetscInt ksp_its = 0;
    PetscCall( SNESGetLinearSolveIterations(snes, &its) );
    ksp_its += its;
    KSPType ksp_type;
    KSPConvergedReason ksp_reason;
    PetscReal ksp_rnorm;
    PC pc;
    PCType pc_type;
    PetscCall( KSPGetPC(ksp, &pc) );
    PetscCall( PCGetType(pc, &pc_type) );
    PetscCall( KSPGetType(ksp, &ksp_type) );
    PetscCall( KSPGetConvergedReason(ksp, &ksp_reason) );
    PetscCall( KSPGetIterationNumber(ksp, &ksp_its) );
    PetscCall( KSPGetResidualNorm(ksp, &ksp_rnorm) );
    PetscCall( PetscPrintf(app_ctx->comm,
                           "  KSP:\n"
                           "    KSP Type                           : %s\n"
                           "    PC Type                            : %s\n"
                           "    KSP Convergence                    : %s\n"
                           "    Total KSP Iterations               : %" PetscInt_FMT "\n"
                           "    Final rnorm                        : %e\n",
                           ksp_type, pc_type, KSPConvergedReasons[ksp_reason], ksp_its,
                           (double)ksp_rnorm ) );
  }

  PetscCall( PetscPrintf(app_ctx->comm,
                         "  L2 Error (MMS):\n"
                         "    L2 error of u and p                : %e, %e\n",
                         (double)l2_error_u,
                         (double)l2_error_p) );
  PetscFunctionReturn(0);
};
// -----------------------------------------------------------------------------
