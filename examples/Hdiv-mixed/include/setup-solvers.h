#ifndef setup_solvers_h
#define setup_solvers_h

#include <ceed.h>
#include <petsc.h>

#include "petscvec.h"
#include "structs.h"

PetscErrorCode SetupJacobianOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data,
                                        VecType vec_type,
                                        OperatorApplyContext ctx_jacobian);
PetscErrorCode SetupResidualOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data,
                                        OperatorApplyContext ctx_residual);
PetscErrorCode SetupErrorOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data,
                                     OperatorApplyContext ctx_error);
PetscErrorCode ApplyMatOp(Mat A, Vec X, Vec Y);
PetscErrorCode SNESFormResidual(SNES snes, Vec X, Vec Y, void *ctx);
PetscErrorCode SNESFormJacobian(SNES snes, Vec U, Mat J, Mat J_pre, void *ctx);
PetscErrorCode PDESolver(CeedData ceed_data, AppCtx app_ctx,
                         SNES snes, KSP ksp, Vec *U);
PetscErrorCode ComputeL2Error(CeedData ceed_data, AppCtx app_ctx, Vec U,
                              CeedScalar *l2_error_u, CeedScalar *l2_error_p);

#endif // setup_solvers_h
