#ifndef setuplibceed_h
#define setuplibceed_h

#include "setup-fe.h"
#include "structs.h"

// Destroy libCEED objects
PetscErrorCode CeedDataDestroy(CeedData ceed_data, ProblemData problem_data);
PetscErrorCode SetupLibceed(DM dm, DM dm_u0, DM dm_p0, DM dm_H1, Ceed ceed, AppCtx app_ctx, ProblemData problem_data, CeedData ceed_data);
#endif  // setuplibceed_h
