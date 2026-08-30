#ifndef setuplibceed_h
#define setuplibceed_h

#include "setup-fe.h"
#include "structs.h"

// Destroy libCEED objects
PetscErrorCode CeedDataDestroy(CeedData ceed_data);
PetscErrorCode SetupLibceed(DM dm, Ceed ceed, AppCtx app_ctx, ProblemData problem_data, CeedData ceed_data, CeedVector rhs_ceed);
#endif  // setuplibceed_h
