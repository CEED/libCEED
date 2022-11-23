// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

//                        libCEED + PETSc Example: Mixed-Poisson in H(div) space
//
// This example demonstrates a simple usage of libCEED with PETSc to solve
//   mixed-elasticity problems.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with: make
// Run with:
//   ./main -pc_type svd -problem elasticity3d -dm_plex_dim 3 -dm_plex_box_faces 4,4,4 -ksp_type minres
//   ./main -pc_type none -problem elasticity3d -dm_plex_filename /path to the mesh file
#include <stdio.h>
const char help[] = "Solve mixed-elasticity problem using PETSc and libCEED\n";

#include "main.h"

int main(int argc, char **argv) {
  // ---------------------------------------------------------------------------
  // Initialize PETSc
  // ---------------------------------------------------------------------------
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  MPI_Comm comm = PETSC_COMM_WORLD;

  // ---------------------------------------------------------------------------
  // Create structs
  // ---------------------------------------------------------------------------
  AppCtx app_ctx;
  PetscCall(PetscCalloc1(1, &app_ctx));

  ProblemData problem_data = NULL;
  PetscCall(PetscCalloc1(1, &problem_data));
  app_ctx->comm = comm;
  // ---------------------------------------------------------------------------
  // Initialize libCEED
  // ---------------------------------------------------------------------------
  // -- Initialize backend
  Ceed ceed;
  CeedInit("/cpu/self/ref/serial", &ceed);
  // CeedInit(app_ctx->ceed_resource, &ceed);

  // ---------------------------------------------------------------------------
  // Setup DM for mixed-problem
  // ---------------------------------------------------------------------------
  DM dm;
  PetscCall(CreateDM(comm, ceed, &dm));

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  // -- Register problems to be available on the command line
  PetscCall(RegisterProblems_MixedElasticity(app_ctx));
  PetscCall(ProcessCommandLineOptions(app_ctx));

  // ---------------------------------------------------------------------------
  // Choose the problem from the list of registered problems
  // ---------------------------------------------------------------------------
  {
    PetscErrorCode (*p)(Ceed, ProblemData, DM, void *);
    PetscCall(PetscFunctionListFind(app_ctx->problems, app_ctx->problem_name, &p));
    if (!p) SETERRQ(PETSC_COMM_SELF, 1, "Problem '%s' not found", app_ctx->problem_name);
    PetscCall((*p)(ceed, problem_data, dm, &app_ctx));
  }

  // ---------------------------------------------------------------------------
  // Setup libCEED qfunctions and operators
  // ---------------------------------------------------------------------------
  PetscCall(SetupFEByOrder(app_ctx, dm));
  CeedElemRestriction aa;
  PetscCall(CreateRestrictionFromPlex(ceed, dm, 0, 0, 0, &aa));
  CeedElemRestrictionView(aa, stdout);
  CeedBasis b;
  PetscCall(CreateBasisFromPlex(ceed, dm, 0, 0, 0, 0, problem_data, &b));
  CeedBasisView(b, stdout);
  Vec U;
  PetscCall(DMCreateGlobalVector(dm, &U));
  VecView(U, PETSC_VIEWER_STDOUT_WORLD);

  // ---------------------------------------------------------------------------
  // Free objects
  // ---------------------------------------------------------------------------

  // Free PETSc objects
  PetscCall(DMDestroy(&dm));

  // -- Function list
  PetscCall(PetscFunctionListDestroy(&app_ctx->problems));

  // -- Structs
  PetscCall(PetscFree(app_ctx));
  PetscCall(PetscFree(problem_data));

  // Free libCEED objects
  CeedDestroy(&ceed);

  return PetscFinalize();
}
