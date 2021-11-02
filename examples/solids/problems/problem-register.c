#include <ceed.h>
#include <petsc.h>
#include "../problems/problems.h"

static bool register_all_called;

#define MACRO(name) PetscErrorCode name(ProblemFunctions problem_functions);
#include "problem-list.h"
#undef MACRO

PetscErrorCode RegisterProblems(ProblemFunctions problem_functions) {
  PetscFunctionBeginUser;
  if (register_all_called) return 0;
  register_all_called = true;

  PetscFunctionBegin;

#define MACRO(name) CHKERRQ(name(problem_functions));
#include "problem-list.h"
#undef MACRO
  PetscFunctionReturn(0);
};
