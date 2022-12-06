#ifndef setup_solvers_h
#define setup_solvers_h

#include <ceed.h>
#include <petsc.h>

#include "petscvec.h"
#include "structs.h"

PetscErrorCode SetupJacobianOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data, OperatorApplyContext ctx_jacobian);
PetscErrorCode SetupResidualOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data, OperatorApplyContext ctx_residual);
PetscErrorCode SetupErrorOperatorCtx_u(DM dm, Ceed ceed, CeedData ceed_data, OperatorApplyContext ctx_error_u);
PetscErrorCode SetupErrorOperatorCtx_p(DM dm, Ceed ceed, CeedData ceed_data, OperatorApplyContext ctx_error_p);
PetscErrorCode ApplyMatOp(Mat A, Vec X, Vec Y);
PetscErrorCode PDESolver(CeedData ceed_data, AppCtx app_ctx, SNES snes, KSP ksp, Vec rhs, Vec *X);
PetscErrorCode ComputeL2Error(Vec X, PetscScalar *l2_error, OperatorApplyContext op_error_ctx);
PetscErrorCode CtxVecDestroy(ProblemData problem_data, AppCtx app_ctx);
#endif  // setup_solvers_h
