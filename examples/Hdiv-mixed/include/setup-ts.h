#ifndef setup_ts_h
#define setup_ts_h

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

PetscErrorCode CreateInitialConditions(DM dm, CeedData ceed_data, Vec *U0);
PetscErrorCode SetupResidualOperatorCtx_Ut(DM dm, Ceed ceed, CeedData ceed_data,
    OperatorApplyContext ctx_residual_ut);
PetscErrorCode TSFormIResidual(TS ts, PetscReal time, Vec X, Vec X_t, Vec Y,
                               void *ctx_residual_ut);
PetscErrorCode TSSolveRichard(DM dm, CeedData ceed_data, AppCtx app_ctx,
                              Vec *U, PetscScalar *f_time, TS *ts);

#endif // setup_ts_h
