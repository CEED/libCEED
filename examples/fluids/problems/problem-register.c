#include <stdbool.h>
#include <stdio.h>
#include "../navierstokes.h"

static bool register_all_called;

#define MACRO(name,...) PetscErrorCode name(AppCtx app_ctx);
#include "problem-list.h"
#undef MACRO


PetscErrorCode RegisterProblems_NS(AppCtx app_ctx) {
  app_ctx->problems = NULL;
  PetscFunctionBeginUser;
  if (register_all_called) return 0;
  register_all_called = true;

#define MACRO(name) CHKERRQ(name(app_ctx));
#include "problem-list.h"
#undef MACRO
  PetscFunctionReturn(0);
}
