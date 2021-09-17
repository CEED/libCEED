#include <ceed.h>
#include <petsc.h>
#include "../problems/problems.h"

PetscErrorCode RegisterProblems(ProblemFunctions problem_functions) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  SOLIDS_PROBLEM_REGISTER(problem_functions, "Linear", ElasLinear, NH);
  SOLIDS_PROBLEM_REGISTER(problem_functions, "SS-NH", ElasSSNH, NH);
  SOLIDS_PROBLEM_REGISTER(problem_functions, "FSCurrent-NH1", ElasFSCurrentNH1,
                          NH);
  SOLIDS_PROBLEM_REGISTER(problem_functions, "FSCurrent-NH2", ElasFSCurrentNH2,
                          NH);
  SOLIDS_PROBLEM_REGISTER(problem_functions, "FSInitial-NH1", ElasFSInitialNH1,
                          NH);
  SOLIDS_PROBLEM_REGISTER(problem_functions, "FSInitial-NH2", ElasFSInitialNH2,
                          NH);
  SOLIDS_PROBLEM_REGISTER(problem_functions, "FSInitial-MR1", ElasFSInitialMR1,
                          MR);

  PetscFunctionReturn(0);
};
