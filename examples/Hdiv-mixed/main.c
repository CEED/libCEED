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
//   ./main -pc_type none -problem darcy2d -dm_plex_dim 2 -dm_plex_box_faces 4,4 -ksp_type minres
//   ./main -pc_type svd -problem darcy3d -dm_plex_dim 3 -dm_plex_box_faces 4,4,4
//   ./main -pc_type svd -problem darcy3d -dm_plex_filename /path to the mesh file
//   ./main -pc_type svd -problem richard2d -dm_plex_dim 2 -dm_plex_box_faces 4,4
// (boundary is not working)
//   ./main -pc_type svd -problem darcy2d -dm_plex_dim 2 -dm_plex_box_faces 4,4 -bc_pressure 1
//   ./main -pc_type svd -problem darcy2d -dm_plex_dim 2 -dm_plex_box_faces 4,4 -bc_pressure 1,2,3,4
//   ./main -pc_type svd -problem darcy3d -dm_plex_dim 3 -dm_plex_box_faces 4,4,4 -view_solution -dm_plex_box_lower 0,0,0 -dm_plex_box_upper 1,0.1,1
const char help[] = "Solve H(div)-mixed problem using PETSc and libCEED\n";

#include "main.h"

int main(int argc, char **argv) {
  // ---------------------------------------------------------------------------
  // Initialize PETSc
  // ---------------------------------------------------------------------------
  PetscCall( PetscInitialize(&argc, &argv, NULL, help) );

  // ---------------------------------------------------------------------------
  // Create structs
  // ---------------------------------------------------------------------------
  AppCtx app_ctx;
  PetscCall( PetscCalloc1(1, &app_ctx) );

  ProblemData problem_data = NULL;
  PetscCall( PetscCalloc1(1, &problem_data) );

  CeedData ceed_data;
  PetscCall( PetscCalloc1(1, &ceed_data) );

  OperatorApplyContext ctx_jacobian, ctx_residual, ctx_residual_ut,
                       ctx_initial_u0, ctx_initial_p0,
                       ctx_error, ctx_Hdiv, ctx_H1;
  PetscCall( PetscCalloc1(1, &ctx_jacobian) );
  PetscCall( PetscCalloc1(1, &ctx_residual) );
  PetscCall( PetscCalloc1(1, &ctx_residual_ut) );
  PetscCall( PetscCalloc1(1, &ctx_initial_u0) );
  PetscCall( PetscCalloc1(1, &ctx_initial_p0) );
  PetscCall( PetscCalloc1(1, &ctx_error) );
  PetscCall( PetscCalloc1(1, &ctx_Hdiv) );
  PetscCall( PetscCalloc1(1, &ctx_H1) );
  // Context for Darcy problem
  app_ctx->ctx_residual = ctx_residual;
  app_ctx->ctx_jacobian = ctx_jacobian;
  // Context for Richards problem
  app_ctx->ctx_residual_ut = ctx_residual_ut;
  // Context for initial velocity
  app_ctx->ctx_initial_u0 = ctx_initial_u0;
  // Context for initial pressure
  app_ctx->ctx_initial_p0 = ctx_initial_p0;
  // Context for MMS
  app_ctx->ctx_error = ctx_error;
  // Context for post-processing
  app_ctx->ctx_Hdiv = ctx_Hdiv;
  app_ctx->ctx_H1 = ctx_H1;

  // ---------------------------------------------------------------------------
  // Initialize libCEED
  // ---------------------------------------------------------------------------
  // -- Initialize backend
  Ceed ceed;
  CeedInit("/cpu/self/ref/serial", &ceed);
  //CeedInit(app_ctx->ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  VecType vec_type = NULL;
  MatType mat_type = NULL;
  switch (mem_type_backend) {
  case CEED_MEM_HOST: vec_type = VECSTANDARD; break;
  case CEED_MEM_DEVICE: {
    const char *resolved;
    CeedGetResource(ceed, &resolved);
    if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
    else if (strstr(resolved, "/gpu/hip")) vec_type = VECKOKKOS;
    else vec_type = VECSTANDARD;
  }
  }
  if (strstr(vec_type, VECCUDA)) mat_type = MATAIJCUSPARSE;
  else if (strstr(vec_type, VECKOKKOS)) mat_type = MATAIJKOKKOS;
  else mat_type = MATAIJ;

  // -- Process general command line options
  MPI_Comm comm = PETSC_COMM_WORLD;
  // ---------------------------------------------------------------------------
  // Create DM
  // ---------------------------------------------------------------------------
  DM  dm, dm_u0, dm_p0, dm_H1;
  // DM for mixed problem
  PetscCall( CreateDM(comm, mat_type, vec_type, &dm) );
  // DM for projecting initial velocity to Hdiv space
  PetscCall( CreateDM(comm, mat_type, vec_type, &dm_u0) );
  // DM for projecting initial pressure in L2
  PetscCall( CreateDM(comm, mat_type, vec_type, &dm_p0) );
  // DM for projecting solution U into H1 space for PetscViewer
  PetscCall( CreateDM(comm, mat_type, vec_type, &dm_H1) );
  // TODO: add mesh option
  // perturb dm to have smooth random mesh
  //PetscCall( PerturbVerticesSmooth(dm) );
  //PetscCall( PerturbVerticesSmooth(dm_H1) );

  // perturb dm to have random mesh
  //PetscCall(PerturbVerticesRandom(dm) );
  //PetscCall(PerturbVerticesRandom(dm_H1) );

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  // -- Register problems to be available on the command line
  PetscCall( RegisterProblems_Hdiv(app_ctx) );

  app_ctx->comm = comm;
  PetscCall( ProcessCommandLineOptions(app_ctx) );

  // ---------------------------------------------------------------------------
  // Choose the problem from the list of registered problems
  // ---------------------------------------------------------------------------
  {
    PetscErrorCode (*p)(Ceed, ProblemData, DM, void *);
    PetscCall( PetscFunctionListFind(app_ctx->problems, app_ctx->problem_name,
                                     &p) );
    if (!p) SETERRQ(PETSC_COMM_SELF, 1, "Problem '%s' not found",
                      app_ctx->problem_name);
    PetscCall( (*p)(ceed, problem_data, dm, &app_ctx) );
  }

  // ---------------------------------------------------------------------------
  // Setup FE for H(div) mixed-problem and H1 projection in post-processing.c
  // ---------------------------------------------------------------------------
  PetscCall( SetupFEHdiv(comm, dm, dm_u0, dm_p0) );
  PetscCall( SetupFEH1(problem_data, app_ctx, dm_H1) );

  // ---------------------------------------------------------------------------
  // Create global unkown solution
  // ---------------------------------------------------------------------------
  Vec U; // U=[p,u]
  PetscCall( DMCreateGlobalVector(dm, &U) );

  // ---------------------------------------------------------------------------
  // Setup libCEED
  // ---------------------------------------------------------------------------
  // -- Set up libCEED objects
  PetscCall( SetupLibceed(dm, dm_u0, dm_p0, dm_H1, ceed, app_ctx,
                          problem_data, ceed_data) );

  // ---------------------------------------------------------------------------
  // Setup pressure boundary conditions
  // ---------------------------------------------------------------------------
  // --Create empty local vector for libCEED
  Vec P_loc;
  PetscInt P_loc_size;
  CeedScalar *p0;
  CeedVector P_ceed;
  PetscMemType pressure_mem_type;
  PetscCall( DMCreateLocalVector(dm, &P_loc) );
  PetscCall( VecGetSize(P_loc, &P_loc_size) );
  PetscCall( VecZeroEntries(P_loc) );
  PetscCall( VecGetArrayAndMemType(P_loc, &p0, &pressure_mem_type) );
  CeedVectorCreate(ceed, P_loc_size, &P_ceed);
  CeedVectorSetArray(P_ceed, MemTypeP2C(pressure_mem_type), CEED_USE_POINTER,
                     p0);
  // -- Apply operator to create local pressure vector on boundary
  PetscCall( DMAddBoundariesPressure(ceed, ceed_data, app_ctx, problem_data, dm,
                                     P_ceed) );
  //CeedVectorView(P_ceed, "%12.8f", stdout);
  // -- Map local to global
  Vec P;
  CeedVectorTakeArray(P_ceed, MemTypeP2C(pressure_mem_type), NULL);
  PetscCall( VecRestoreArrayAndMemType(P_loc, &p0) );
  PetscCall( DMCreateGlobalVector(dm, &P) );
  PetscCall( VecZeroEntries(P) );
  PetscCall( DMLocalToGlobal(dm, P_loc, ADD_VALUES, P) );

  // ---------------------------------------------------------------------------
  // Setup context for projection problem; post-processing.c
  // ---------------------------------------------------------------------------
  PetscCall( SetupProjectVelocityCtx_Hdiv(comm, dm, ceed, ceed_data,
                                          app_ctx->ctx_Hdiv) );
  PetscCall( SetupProjectVelocityCtx_H1(comm, dm_H1, ceed, ceed_data,
                                        vec_type, app_ctx->ctx_H1) );

  // ---------------------------------------------------------------------------
  // Setup TSSolve for Richard problem
  // ---------------------------------------------------------------------------
  TS ts;
  if (problem_data->has_ts) {
    // ---------------------------------------------------------------------------
    // Setup context for initial conditions
    // ---------------------------------------------------------------------------
    PetscCall( SetupResidualOperatorCtx_U0(comm, dm_u0, ceed, ceed_data,
                                           app_ctx->ctx_initial_u0) );
    PetscCall( SetupResidualOperatorCtx_P0(comm, dm_p0, ceed, ceed_data,
                                           app_ctx->ctx_initial_p0) );
    PetscCall( SetupResidualOperatorCtx_Ut(comm, dm, ceed, ceed_data,
                                           app_ctx->ctx_residual_ut) );
    PetscCall( CreateInitialConditions(ceed_data, app_ctx, vec_type, U) );
    //VecView(U, PETSC_VIEWER_STDOUT_WORLD);
    // Solve Richards problem
    PetscCall( TSCreate(comm, &ts) );
    PetscCall( VecZeroEntries(app_ctx->ctx_residual_ut->X_loc) );
    PetscCall( VecZeroEntries(app_ctx->ctx_residual_ut->X_t_loc) );
    PetscCall( TSSolveRichard(ceed_data, app_ctx, ts, &U) );
    //VecView(U, PETSC_VIEWER_STDOUT_WORLD);
  }

  // ---------------------------------------------------------------------------
  // Setup SNES for Darcy problem
  // ---------------------------------------------------------------------------
  SNES snes;
  KSP ksp;
  if (!problem_data->has_ts) {
    PetscCall( SetupJacobianOperatorCtx(dm, ceed, ceed_data, vec_type,
                                        app_ctx->ctx_jacobian) );
    PetscCall( SetupResidualOperatorCtx(dm, ceed, ceed_data,
                                        app_ctx->ctx_residual) );
    // Create SNES
    PetscCall( SNESCreate(comm, &snes) );
    PetscCall( SNESGetKSP(snes, &ksp) );
    PetscCall( PDESolver(ceed_data, app_ctx, snes, ksp, &U) );
    //VecView(U, PETSC_VIEWER_STDOUT_WORLD);
  }

  // ---------------------------------------------------------------------------
  // Compute L2 error of mms problem
  // ---------------------------------------------------------------------------
  PetscCall( SetupErrorOperatorCtx(dm, ceed, ceed_data, app_ctx->ctx_error) );
  CeedScalar l2_error_u, l2_error_p;
  PetscCall( ComputeL2Error(ceed_data, app_ctx, U,
                            &l2_error_u, &l2_error_p) );

  // ---------------------------------------------------------------------------
  // Print solver iterations and final norms
  // ---------------------------------------------------------------------------
  PetscCall( PrintOutput(dm, ceed, app_ctx, problem_data->has_ts,
                         mem_type_backend,
                         ts, snes, ksp, U, l2_error_u, l2_error_p) );

  // ---------------------------------------------------------------------------
  // Save solution (paraview)
  // ---------------------------------------------------------------------------
  if (app_ctx->view_solution) {
    PetscViewer viewer_p;
    PetscCall( PetscViewerVTKOpen(comm,"darcy_pressure.vtu",FILE_MODE_WRITE,
                                  &viewer_p) );
    PetscCall( VecView(U, viewer_p) );
    PetscCall( PetscViewerDestroy(&viewer_p) );

    Vec U_H1; // velocity in H1 space for post-processing
    PetscCall( DMCreateGlobalVector(dm_H1, &U_H1) );
    PetscCall( ProjectVelocity(app_ctx, U, &U_H1) );

    PetscViewer viewer_u;
    PetscCall( PetscViewerVTKOpen(comm,"darcy_velocity.vtu",FILE_MODE_WRITE,
                                  &viewer_u) );
    PetscCall( VecView(U_H1, viewer_u) );
    PetscCall( PetscViewerDestroy(&viewer_u) );
    PetscCall( VecDestroy(&U_H1) );
  }
  // ---------------------------------------------------------------------------
  // Free objects
  // ---------------------------------------------------------------------------

  // Free PETSc objects
  PetscCall( DMDestroy(&dm) );
  PetscCall( DMDestroy(&dm_u0) );
  PetscCall( DMDestroy(&dm_p0) );
  PetscCall( DMDestroy(&dm_H1) );
  PetscCall( VecDestroy(&U) );
  PetscCall( CtxVecDestroy(app_ctx) );
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
  PetscCall( PetscFree(ctx_residual) );
  PetscCall( PetscFree(ctx_jacobian) );
  PetscCall( PetscFree(ctx_error) );
  PetscCall( PetscFree(ctx_H1) );
  PetscCall( PetscFree(ctx_Hdiv) );

  // Free libCEED objects
  //CeedVectorDestroy(&bc_pressure);
  CeedDestroy(&ceed);

  return PetscFinalize();
}
