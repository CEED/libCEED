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
//   elasticity problems.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with: make
// Run with:
//   ./main -pc_type svd -problem darcy2d -dm_plex_dim 2 -dm_plex_box_faces 4,4
//   ./main -pc_type svd -problem darcy3d -dm_plex_dim 3 -dm_plex_box_faces 4,4,4
//   ./main -pc_type svd -problem darcy3d -dm_plex_filename /path to the mesh file
//   ./main -pc_type svd -problem darcy2d -dm_plex_dim 2 -dm_plex_box_faces 4,4 -bc_pressure 1
//   ./main -pc_type svd -problem darcy2d -dm_plex_dim 2 -dm_plex_box_faces 4,4 -bc_pressure 1,2,3,4
const char help[] = "Solve H(div)-mixed problem using PETSc and libCEED\n";

#include "main.h"

int main(int argc, char **argv) {
  // ---------------------------------------------------------------------------
  // Initialize PETSc
  // ---------------------------------------------------------------------------
  PetscCall( PetscInitialize(&argc, &argv, NULL, help) );

  // ---------------------------------------------------------------------------
  // Initialize libCEED
  // ---------------------------------------------------------------------------
  // -- Initialize backend
  Ceed ceed;
  CeedInit("/cpu/self/ref/serial", &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  VecType        vec_type = NULL;
  switch (mem_type_backend) {
  case CEED_MEM_HOST: vec_type = VECSTANDARD; break;
  case CEED_MEM_DEVICE: {
    const char *resolved;
    CeedGetResource(ceed, &resolved);
    if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
    else if (strstr(resolved, "/gpu/hip/occa"))
      vec_type = VECSTANDARD; // https://github.com/CEED/libCEED/issues/678
    else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
    else vec_type = VECSTANDARD;
  }
  }

  // ---------------------------------------------------------------------------
  // Create structs
  // ---------------------------------------------------------------------------
  AppCtx app_ctx;
  PetscCall( PetscCalloc1(1, &app_ctx) );

  ProblemData problem_data = NULL;
  PetscCall( PetscCalloc1(1, &problem_data) );

  CeedData ceed_data;
  PetscCall( PetscCalloc1(1, &ceed_data) );

  OperatorApplyContext ctx_residual_ut, ctx_initial_u0, ctx_initial_p0;
  PetscCall( PetscCalloc1(1, &ctx_residual_ut) );
  PetscCall( PetscCalloc1(1, &ctx_initial_u0) );
  PetscCall( PetscCalloc1(1, &ctx_initial_p0) );
  ceed_data->ctx_residual_ut = ctx_residual_ut;
  ceed_data->ctx_initial_u0 = ctx_initial_u0;
  ceed_data->ctx_initial_p0 = ctx_initial_p0;
  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  // -- Register problems to be available on the command line
  PetscCall( RegisterProblems_Hdiv(app_ctx) );

  // -- Process general command line options
  MPI_Comm comm = PETSC_COMM_WORLD;
  app_ctx->comm = comm;
  PetscCall( ProcessCommandLineOptions(app_ctx) );

  // ---------------------------------------------------------------------------
  // Choose the problem from the list of registered problems
  // ---------------------------------------------------------------------------
  {
    PetscErrorCode (*p)(Ceed, ProblemData, void *);
    PetscCall( PetscFunctionListFind(app_ctx->problems, app_ctx->problem_name,
                                     &p) );
    if (!p) SETERRQ(PETSC_COMM_SELF, 1, "Problem '%s' not found",
                      app_ctx->problem_name);
    PetscCall( (*p)(ceed, problem_data, &app_ctx) );
  }

  // ---------------------------------------------------------------------------
  // Create DM
  // ---------------------------------------------------------------------------
  DM  dm, dm_u0, dm_p0;
  PetscCall( CreateDM(comm, vec_type, &dm) );
  PetscCall( CreateDM(comm, vec_type, &dm_u0) );
  PetscCall( CreateDM(comm, vec_type, &dm_p0) );
  // TODO: add mesh option
  // perturb to have smooth random mesh
  //PetscCall( PerturbVerticesSmooth(dm) );

  // ---------------------------------------------------------------------------
  // Setup FE
  // ---------------------------------------------------------------------------
  SetupFE(comm, dm, dm_u0, dm_p0);

  // ---------------------------------------------------------------------------
  // Create local Force vector
  // ---------------------------------------------------------------------------
  Vec U; // U=[p,u], U0=u0
  PetscCall( DMCreateGlobalVector(dm, &U) );
  PetscCall( VecZeroEntries(U) );

  // ---------------------------------------------------------------------------
  // Setup libCEED
  // ---------------------------------------------------------------------------
  // -- Set up libCEED objects
  PetscCall( SetupLibceed(dm, dm_u0, dm_p0, ceed, app_ctx, problem_data,
                          ceed_data) );
  //CeedVectorView(force_ceed, "%12.8f", stdout);
  //PetscCall( DMAddBoundariesPressure(ceed, ceed_data, app_ctx, problem_data, dm,
  //                                   bc_pressure) );

  // ---------------------------------------------------------------------------
  // Setup TSSolve for Richard problem
  // ---------------------------------------------------------------------------
  TS ts;
  if (problem_data->has_ts) {
    // ---------------------------------------------------------------------------
    // Create global initial conditions
    // ---------------------------------------------------------------------------
    SetupResidualOperatorCtx_U0(comm, dm_u0, ceed, ceed_data,
                                ceed_data->ctx_initial_u0);
    SetupResidualOperatorCtx_P0(comm, dm_p0, ceed, ceed_data,
                                ceed_data->ctx_initial_p0);
    SetupResidualOperatorCtx_Ut(comm, dm, ceed, ceed_data,
                                ceed_data->ctx_residual_ut);
    CreateInitialConditions(ceed_data, U, vec_type,
                            ceed_data->ctx_initial_u0,
                            ceed_data->ctx_initial_p0,
                            ceed_data->ctx_residual_ut);
    //VecView(U, PETSC_VIEWER_STDOUT_WORLD);
    // Solve Richards problem
    PetscCall( VecZeroEntries(ceed_data->ctx_residual_ut->X_loc) );
    PetscCall( VecZeroEntries(ceed_data->ctx_residual_ut->X_t_loc) );
    PetscCall( TSSolveRichard(dm, ceed_data, app_ctx,
                              &U, &ts) );
    //VecView(U, PETSC_VIEWER_STDOUT_WORLD);
  }

  SNES snes;
  KSP ksp;
  if (!problem_data->has_ts) {
    // ---------------------------------------------------------------------------
    // Setup SNES for Darcy problem
    // ---------------------------------------------------------------------------
    // Create SNES
    PetscCall( SNESCreate(comm, &snes) );
    PetscCall( SNESGetKSP(snes, &ksp) );
    PetscCall( PDESolver(comm, dm, ceed, ceed_data, vec_type, snes, ksp, &U) );
    //VecView(U, PETSC_VIEWER_STDOUT_WORLD);
  }

  // ---------------------------------------------------------------------------
  // Compute L2 error of mms problem
  // ---------------------------------------------------------------------------
  CeedScalar l2_error_u, l2_error_p;
  PetscCall( ComputeL2Error(dm, ceed, ceed_data, U, &l2_error_u,
                            &l2_error_p) );

  // ---------------------------------------------------------------------------
  // Print output results
  // ---------------------------------------------------------------------------
  PetscCall( PrintOutput(ceed, app_ctx, problem_data->has_ts, mem_type_backend,
                         ts, snes, ksp, U, l2_error_u, l2_error_p) );

  // ---------------------------------------------------------------------------
  // Save solution (paraview)
  // ---------------------------------------------------------------------------
  PetscViewer viewer;
  PetscCall( PetscViewerVTKOpen(comm,"solution.vtu",FILE_MODE_WRITE,&viewer) );
  PetscCall( VecView(U, viewer) );
  PetscCall( PetscViewerDestroy(&viewer) );

  // ---------------------------------------------------------------------------
  // Free objects
  // ---------------------------------------------------------------------------

  // Free PETSc objects
  PetscCall( DMDestroy(&dm) );
  PetscCall( DMDestroy(&dm_u0) );
  PetscCall( DMDestroy(&dm_p0) );
  PetscCall( VecDestroy(&U) );
  if (problem_data->has_ts) {
    PetscCall( TSDestroy(&ts) );
  } else {
    PetscCall( SNESDestroy(&snes) );
  }
  PetscCall( CeedDataDestroy(ceed_data, problem_data) );

  // -- Function list
  PetscCall( PetscFunctionListDestroy(&app_ctx->problems) );

  // -- Structs
  PetscCall( PetscFree(app_ctx) );
  PetscCall( PetscFree(problem_data) );
  PetscCall( PetscFree(ctx_initial_u0) );
  PetscCall( PetscFree(ctx_initial_p0) );
  PetscCall( PetscFree(ctx_residual_ut) );

  // Free libCEED objects
  //CeedVectorDestroy(&bc_pressure);
  CeedDestroy(&ceed);

  return PetscFinalize();
}
