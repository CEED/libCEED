#ifndef setup_solvers_h
#define setup_solvers_h

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

PetscErrorCode SetupJacobianOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data,
                                        OperatorApplyContext ctx_jacobian);
PetscErrorCode SetupResidualOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data,
                                        OperatorApplyContext ctx_residual);
PetscErrorCode SetupErrorOperatorCtx(DM dm, Ceed ceed, CeedData ceed_data,
                                     OperatorApplyContext ctx_error);
PetscErrorCode ApplyMatOp(Mat A, Vec X, Vec Y);
PetscErrorCode SNESFormResidual(SNES snes, Vec X, Vec Y, void *ctx);
PetscErrorCode SNESFormJacobian(SNES snes, Vec U, Mat J, Mat J_pre, void *ctx);
PetscErrorCode PDESolver(MPI_Comm comm, DM dm, Ceed ceed, CeedData ceed_data,
                         VecType vec_type, SNES snes, KSP ksp, Vec *U);
PetscErrorCode ComputeL2Error(DM dm, Ceed ceed, CeedData ceed_data, Vec U,
                              CeedScalar *l2_error_u, CeedScalar *l2_error_p);
PetscErrorCode PrintOutput(Ceed ceed, AppCtx app_ctx, PetscBool has_ts,
                           CeedMemType mem_type_backend,
                           TS ts, SNES snes, KSP ksp,
                           Vec U, CeedScalar l2_error_u,
                           CeedScalar l2_error_p);

#endif // setup_solvers_h
