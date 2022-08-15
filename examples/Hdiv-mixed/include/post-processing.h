#ifndef post_processing_h
#define post_processing_h

#include <ceed.h>
#include <petsc.h>

#include "structs.h"
#include "../include/setup-libceed.h"
PetscErrorCode PrintOutput(DM dm, Ceed ceed, AppCtx app_ctx, PetscBool has_ts,
                           CeedMemType mem_type_backend,
                           TS ts, SNES snes, KSP ksp,
                           Vec U, CeedScalar l2_error_u,
                           CeedScalar l2_error_p);
PetscErrorCode SetupProjectVelocityCtx_Hdiv(MPI_Comm comm, DM dm, Ceed ceed,
    CeedData ceed_data, OperatorApplyContext ctx_Hdiv);
PetscErrorCode SetupProjectVelocityCtx_H1(MPI_Comm comm, DM dm_H1, Ceed ceed,
    CeedData ceed_data, VecType vec_type, OperatorApplyContext ctx_H1);
PetscErrorCode ProjectVelocity(AppCtx app_ctx,
                               Vec U, Vec *U_H1);
PetscErrorCode CtxVecDestroy(AppCtx app_ctx);
#endif // post_processing_h
