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

//                        libCEED + PETSc Example: Projection problem in H(div) space
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with: make
// Run with:
//          ./main pc_type svd
//          ./main -pc_type svd -problem mass2d -dm_plex_dim 2 -dm_plex_box_faces 4,4
//          ./main -pc_type svd -problem mass3d -dm_plex_dim 3 -dm_plex_box_faces 4,4,4

#include <stdio.h>
const char help[] = "Solve Projection problem in H(div) space using PETSc and libCEED\n";

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

  CeedData ceed_data;
  PetscCall(PetscCalloc1(1, &ceed_data));

  OperatorApplyContext ctx_residual, ctx_error_u;
  PetscCall(PetscCalloc1(1, &ctx_residual));
  PetscCall(PetscCalloc1(1, &ctx_error_u));
  // Context for residual
  app_ctx->ctx_residual = ctx_residual;
  // Context for computing error
  app_ctx->ctx_error_u = ctx_error_u;
  app_ctx->comm        = comm;

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  PetscCall(ProcessCommandLineOptions(app_ctx));

  // ---------------------------------------------------------------------------
  // Initialize libCEED
  // ---------------------------------------------------------------------------
  // -- Initialize backend
  Ceed ceed;
  CeedInit("/cpu/self/ref/serial", &ceed);
  // CeedInit(app_ctx->ceed_resource, &ceed);

  // ---------------------------------------------------------------------------
  // Choose the problem from the list of registered problems
  // ---------------------------------------------------------------------------
  PetscCall(RegisterProblems_Hdiv(app_ctx));
  {
    PetscErrorCode (*p)(ProblemData, void *);
    PetscCall(PetscFunctionListFind(app_ctx->problems, app_ctx->problem_name, &p));
    if (!p) SETERRQ(PETSC_COMM_SELF, 1, "Problem '%s' not found", app_ctx->problem_name);
    PetscCall((*p)(problem_data, &app_ctx));
  }

  // ---------------------------------------------------------------------------
  // Create DM and Setup FE space
  // ---------------------------------------------------------------------------
  DM dm;
  PetscCall(CreateDM(comm, ceed, &dm));
  PetscCall(SetupFEHdiv(app_ctx, problem_data, dm));

  // ---------------------------------------------------------------------------
  //  Create zero rhs local vector
  // ---------------------------------------------------------------------------
  CeedVector   rhs_ceed;
  Vec          rhs_loc;
  PetscScalar *r;
  PetscMemType mem_type;
  PetscInt     rhs_l_size;
  // Create global and local solution vectors
  PetscCall(DMCreateLocalVector(dm, &rhs_loc));
  PetscCall(VecGetSize(rhs_loc, &rhs_l_size));
  PetscCall(VecZeroEntries(rhs_loc));
  PetscCall(VecGetArrayAndMemType(rhs_loc, &r, &mem_type));
  CeedVectorCreate(ceed, rhs_l_size, &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  // ---------------------------------------------------------------------------
  // Setup libCEED qfunctions and operators
  // ---------------------------------------------------------------------------
  PetscCall(SetupLibceed(dm, ceed, app_ctx, problem_data, ceed_data, rhs_ceed));

  // ---------------------------------------------------------------------------
  // Setup rhs global vector entries with the computed rhs_ceed
  // ---------------------------------------------------------------------------
  Vec rhs;
  PetscCall(DMCreateGlobalVector(dm, &rhs));
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(rhs_loc, &r));
  PetscCall(VecZeroEntries(rhs));
  PetscCall(DMLocalToGlobal(dm, rhs_loc, ADD_VALUES, rhs));
  CeedVectorDestroy(&rhs_ceed);

  // ---------------------------------------------------------------------------
  // Solve A*X=rhs; setup-solver.c
  // ---------------------------------------------------------------------------
  Vec X;
  KSP ksp;
  PetscCall(SetupResidualOperatorCtx(app_ctx->comm, dm, ceed, ceed_data, app_ctx->ctx_residual));
  // Create global and local solution vectors
  PetscCall(DMCreateGlobalVector(dm, &X));
  PetscCall(KSPCreate(app_ctx->comm, &ksp));
  PetscCall(KSPSetDM(ksp, dm));
  PetscCall(KSPSetDMActive(ksp, PETSC_FALSE));
  PetscCall(PDESolver(ceed_data, app_ctx, ksp, rhs, &X));

  // ---------------------------------------------------------------------------
  // Compute L2 error of mms problem; setup-solver.c
  // ---------------------------------------------------------------------------
  CeedScalar l2_error_u = 0.0;
  PetscCall(SetupErrorOperatorCtx(app_ctx->comm, dm, ceed, ceed_data, app_ctx->ctx_error_u));
  PetscCall(ComputeL2Error(X, &l2_error_u, app_ctx->ctx_error_u));

  // ---------------------------------------------------------------------------
  // Print solver iterations and final norms; post-processing
  // ---------------------------------------------------------------------------
  PetscCall(PrintOutput(dm, ceed, app_ctx, ksp, X, l2_error_u));

  // ---------------------------------------------------------------------------
  // Free objects
  // ---------------------------------------------------------------------------

  // Free PETSc objects
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&rhs_loc));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(CtxVecDestroy(problem_data, app_ctx));
  // -- Function list
  PetscCall(PetscFunctionListDestroy(&app_ctx->problems));
  // -- Structs
  PetscCall(CeedDataDestroy(ceed_data));
  PetscCall(PetscFree(app_ctx));
  PetscCall(PetscFree(ctx_residual));
  PetscCall(PetscFree(ctx_error_u));
  PetscCall(PetscFree(problem_data));
  // Free libCEED objects
  CeedVectorDestroy(&rhs_ceed);
  CeedDestroy(&ceed);

  return PetscFinalize();
}
