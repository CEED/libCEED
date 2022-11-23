#ifndef post_processing_h
#define post_processing_h

#include <ceed.h>
#include <petsc.h>

#include "setup-fe.h"
#include "structs.h"
PetscErrorCode PrintOutput(DM dm, Ceed ceed, AppCtx app_ctx, KSP ksp, Vec X, CeedScalar l2_error_u);

#endif  // post_processing_h
