#include <ceed.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ceed/backend.h>
#include "../navierstokes.h"

// This function provides a debug target for weak symbols
// LCOV_EXCL_START
static PetscErrorCode ProblemRegister_Weak(const char *name) {
  if (getenv("CEED_DEBUG")) fprintf(stderr, "Weak %s\n", name);
  PetscFunctionReturn(0);
}
// LCOV_EXCL_STOP

#define MACRO(name)                                                     \
  PetscErrorCode name(AppCtx app_ctx) __attribute__((weak));                     \
  PetscErrorCode name(AppCtx app_ctx) { return ProblemRegister_Weak(__func__); }
#include "problem-list.h"
#undef MACRO
