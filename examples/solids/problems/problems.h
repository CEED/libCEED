// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef problems_h
#define problems_h

#include <ceed.h>
#include <petsc.h>

#include "../problems/cl-problems.h"
#include "../problems/mooney-rivlin.h"
#include "../problems/neo-hookean.h"

// Physics options
#define SOLIDS_PROBLEM_REGISTER(list, name, fname, physics)                                             \
  PetscCall(PetscFunctionListAdd(&list->setupPhysics, name, PhysicsContext_##physics));                 \
  PetscCall(PetscFunctionListAdd(&list->setupSmootherPhysics, name, PhysicsSmootherContext_##physics)); \
  PetscCall(PetscFunctionListAdd(&list->setupLibceedFineLevel, name, SetupLibceedFineLevel_##fname));   \
  PetscCall(PetscFunctionListAdd(&list->setupLibceedLevel, name, SetupLibceedLevel_##fname));

typedef struct ProblemFunctions_ *ProblemFunctions;
struct ProblemFunctions_ {
  PetscFunctionList setupPhysics, setupSmootherPhysics, setupLibceedFineLevel, setupLibceedLevel;
};

PetscErrorCode RegisterProblems(ProblemFunctions problem_functions);

#define SOLIDS_PROBLEM(name)                                                                                                                   \
  PetscErrorCode SetupLibceedFineLevel_##name(DM dm, DM dm_energy, DM dm_diagnostic, Ceed ceed, AppCtx app_ctx, CeedQFunctionContext phys_ctx, \
                                              PetscInt fine_level, PetscInt num_comp_u, PetscInt U_g_size, PetscInt U_loc_size,                \
                                              CeedVector force_ceed, CeedVector neumann_ceed, CeedData *data);                                 \
  PetscErrorCode SetupLibceedLevel_##name(DM dm, Ceed ceed, AppCtx app_ctx, PetscInt level, PetscInt num_comp_u, PetscInt U_g_size,            \
                                          PetscInt u_loc_size, CeedVector fine_mult, CeedData *data);

SOLIDS_PROBLEM(ElasLinear);
SOLIDS_PROBLEM(ElasSSNH);
SOLIDS_PROBLEM(ElasFSCurrentNH1);
SOLIDS_PROBLEM(ElasFSCurrentNH2);
SOLIDS_PROBLEM(ElasFSInitialNH1);
SOLIDS_PROBLEM(ElasFSInitialNH2);
SOLIDS_PROBLEM(ElasFSInitialMR1);

#endif  // problems_h
