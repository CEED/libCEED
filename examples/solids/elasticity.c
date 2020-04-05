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

//                        libCEED + PETSc Example: Elasticity
//
// This example demonstrates a simple usage of libCEED with PETSc to solve
//   elasticity problems.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make elasticity [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     ./elasticity -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -problem linElas -forcing mms
//     ./elasticity -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -bc_zero 999 -bc_clamp 998 -problem hyperSS -forcing none -ceed /cpu/self
//     ./elasticity -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -bc_zero 999 -bc_clamp 998 -problem hyperFS -forcing none -ceed /gpu/occa
//
// Sample meshes can be found at https://github.com/jeremylt/ceedSampleMeshes
//
//TESTARGS -ceed {ceed_resource} -test -degree 2 -nu 0.3 -E 1

/// @file
/// CEED elasticity example using PETSc with DMPlex

const char help[] = "Solve solid Problems with CEED and PETSc DMPlex\n";

#include "elasticity.h"

int main(int argc, char **argv) {
  PetscInt       ierr;
  MPI_Comm       comm;
  // Context structs
  AppCtx         appCtx;                 // Contains problem options
  Physics        phys;                   // Contains physical constants
  Units          units;                  // Contains units scaling
  // PETSc objects
  PetscLogStage  stageDMSetup, stageLibceedSetup,
                 stageSnesSetup, stageSnesSolve;
  DM             dmOrig;                 // Distributed DM to clone
  DM             *levelDMs;
  Vec            U, *Ug, *Uloc;          // U: solution, R: residual, F: forcing
  Vec            R, Rloc, F, Floc;       // g: global, loc: local
  SNES           snes, snesCoarse;
  Mat            *jacobMat, jacobMatCoarse, *prolongRestrMat;
  // PETSc data
  UserMult       resCtx, jacobCoarseCtx, *jacobCtx;
  FormJacobCtx   formJacobCtx;
  UserMultProlongRestr *prolongRestrCtx;
  PCMGCycleType  pcmgCycleType = PC_MG_CYCLE_V;
  // libCEED objects
  Ceed           ceed, ceedFine = NULL;
  CeedData       *ceedData;
  CeedQFunction  qfRestrict, qfProlong;
  // Parameters
  PetscInt       ncompu = 3;             // 3 DoFs in 3D
  PetscInt       numLevels = 1, fineLevel = 0;
  PetscInt       *Ugsz, *Ulsz, *Ulocsz;  // sz: size
  PetscInt       snesIts = 0;
  // Timing
  double         startTime, elapsedTime, minTime, maxTime;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr)
    return ierr;

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  comm = PETSC_COMM_WORLD;

  // -- Set mesh file, polynomial degree, problem type
  ierr = PetscCalloc1(1, &appCtx); CHKERRQ(ierr);
  ierr = ProcessCommandLineOptions(comm, appCtx); CHKERRQ(ierr);
  numLevels = appCtx->numLevels;
  fineLevel = numLevels - 1;

  // -- Set Poison's ratio, Young's Modulus
  ierr = PetscMalloc1(1, &phys); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &units); CHKERRQ(ierr);
  ierr = ProcessPhysics(comm, phys, units); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Setup DM
  // ---------------------------------------------------------------------------
  // Performance logging
  ierr = PetscLogStageRegister("DM and Vector Setup Stage", &stageDMSetup);
  CHKERRQ(ierr);
  ierr = PetscLogStagePush(stageDMSetup); CHKERRQ(ierr);

  // -- Create distributed DM from mesh file
  ierr = CreateDistributedDM(comm, appCtx, &dmOrig); CHKERRQ(ierr);

  // -- Setup DM by polynomial degree
  ierr = PetscMalloc1(numLevels, &levelDMs); CHKERRQ(ierr);
  for (int level = 0; level < numLevels; level++) {
    ierr = DMClone(dmOrig, &levelDMs[level]); CHKERRQ(ierr);
    ierr = SetupDMByDegree(levelDMs[level], appCtx, appCtx->levelDegrees[level],
                           ncompu); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Setup solution and work vectors
  // ---------------------------------------------------------------------------
  // Allocate arrays
  ierr = PetscMalloc1(numLevels, &Ug); CHKERRQ(ierr);
  ierr = PetscMalloc1(numLevels, &Uloc); CHKERRQ(ierr);
  ierr = PetscMalloc1(numLevels, &Ugsz); CHKERRQ(ierr);
  ierr = PetscMalloc1(numLevels, &Ulsz); CHKERRQ(ierr);
  ierr = PetscMalloc1(numLevels, &Ulocsz); CHKERRQ(ierr);

  // -- Setup solution vectors for each level
  for (int level = 0; level < numLevels; level++) {
    // -- Create global unknown vector U
    ierr = DMCreateGlobalVector(levelDMs[level], &Ug[level]); CHKERRQ(ierr);
    ierr = VecGetSize(Ug[level], &Ugsz[level]); CHKERRQ(ierr);
    // Note: Local size for matShell
    ierr = VecGetLocalSize(Ug[level], &Ulsz[level]); CHKERRQ(ierr);

    // -- Create local unknown vector Uloc
    ierr = DMCreateLocalVector(levelDMs[level], &Uloc[level]); CHKERRQ(ierr);
    // Note: local size for libCEED
    ierr = VecGetSize(Uloc[level], &Ulocsz[level]); CHKERRQ(ierr);
  }

  // -- Create residual and forcing vectors
  ierr = VecDuplicate(Ug[fineLevel], &U); CHKERRQ(ierr);
  ierr = VecDuplicate(Ug[fineLevel], &R); CHKERRQ(ierr);
  ierr = VecDuplicate(Ug[fineLevel], &F); CHKERRQ(ierr);
  ierr = VecDuplicate(Uloc[fineLevel], &Rloc); CHKERRQ(ierr);
  ierr = VecDuplicate(Uloc[fineLevel], &Floc); CHKERRQ(ierr);

  // Performance logging
  ierr = PetscLogStagePop();

  // ---------------------------------------------------------------------------
  // Set up libCEED
  // ---------------------------------------------------------------------------
  // Performance logging
  ierr = PetscLogStageRegister("libCEED Setup Stage", &stageLibceedSetup);
  CHKERRQ(ierr);
  ierr = PetscLogStagePush(stageLibceedSetup); CHKERRQ(ierr);

  // Initalize backend
  CeedInit(appCtx->ceedResource, &ceed);
  if (appCtx->degree > 4 && appCtx->ceedResourceFine[0])
    CeedInit(appCtx->ceedResourceFine, &ceedFine);

  // -- Create libCEED local forcing vector
  CeedVector forceCeed;
  CeedScalar *f;
  if (appCtx->forcingChoice != FORCE_NONE) {
    ierr = VecGetArray(Floc, &f); CHKERRQ(ierr);
    CeedVectorCreate(ceed, Ulocsz[fineLevel], &forceCeed);
    CeedVectorSetArray(forceCeed, CEED_MEM_HOST, CEED_USE_POINTER, f);
  }

  // -- Restriction and prolongation QFunction
  if (appCtx->multigridChoice != MULTIGRID_NONE) {
    CeedQFunctionCreateIdentity(ceed, ncompu, CEED_EVAL_NONE, CEED_EVAL_INTERP,
                                &qfRestrict);
    CeedQFunctionCreateIdentity(ceed, ncompu, CEED_EVAL_INTERP, CEED_EVAL_NONE,
                                &qfProlong);
  }

  // -- Setup libCEED objects
  ierr = PetscMalloc1(numLevels, &ceedData); CHKERRQ(ierr);
  // ---- Setup residual evaluator and geometric information
  ierr = PetscCalloc1(1, &ceedData[fineLevel]); CHKERRQ(ierr);
  {
    bool highOrder = (appCtx->levelDegrees[fineLevel] > 4 && ceedFine);
    Ceed levelCeed = highOrder ? ceedFine : ceed;
    ierr = SetupLibceedFineLevel(levelDMs[fineLevel], levelCeed, appCtx,
                                 phys, ceedData, fineLevel, ncompu,
                                 Ugsz[fineLevel], Ulocsz[fineLevel], forceCeed,
                                 qfRestrict, qfProlong);
    CHKERRQ(ierr);
  }
  // ---- Setup Jacobian evaluator and prolongation/restriction
  for (int level = 0; level < numLevels; level++) {
    if (level != fineLevel) {
      ierr = PetscCalloc1(1, &ceedData[level]); CHKERRQ(ierr);
    }

    // Note: use high order ceed, if specified and degree > 3
    bool highOrder = (appCtx->levelDegrees[level] > 4 && ceedFine);
    Ceed levelCeed = highOrder ? ceedFine : ceed;
    ierr = SetupLibceedLevel(levelDMs[level], levelCeed, appCtx, phys,
                             ceedData,  level, ncompu, Ugsz[level],
                             Ulocsz[level], forceCeed, qfRestrict,
                             qfProlong); CHKERRQ(ierr);
  }

  // Performance logging
  ierr = PetscLogStagePop();

  // ---------------------------------------------------------------------------
  // Setup global forcing vector
  // ---------------------------------------------------------------------------
  ierr = VecZeroEntries(F); CHKERRQ(ierr);

  if (appCtx->forcingChoice != FORCE_NONE) {
    ierr = VecRestoreArray(Floc, &f); CHKERRQ(ierr);
    ierr = DMLocalToGlobal(levelDMs[fineLevel], Floc, ADD_VALUES, F);
    CHKERRQ(ierr);
    CeedVectorDestroy(&forceCeed);
  }

  // ---------------------------------------------------------------------------
  // Print problem summary
  // ---------------------------------------------------------------------------
  if (!appCtx->testMode) {
    const char *usedresource;
    CeedGetResource(ceed, &usedresource);

    ierr = PetscPrintf(comm,
                       "\n-- Elastisticy Example - libCEED + PETSc --\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n",
                       usedresource); CHKERRQ(ierr);

    if (ceedFine) {
      ierr = PetscPrintf(comm,
                         "    libCEED Backend - high order       : %s\n",
                         appCtx->ceedResourceFine); CHKERRQ(ierr);
    }

    ierr = PetscPrintf(comm,
                       "  Problem:\n"
                       "    Problem Name                       : %s\n"
                       "    Forcing Function                   : %s\n"
                       "  Mesh:\n"
                       "    File                               : %s\n"
                       "    Number of 1D Basis Nodes (p)       : %d\n"
                       "    Number of 1D Quadrature Points (q) : %d\n"
                       "    Global nodes                       : %D\n"
                       "    Owned nodes                        : %D\n"
                       "    DoF per node                       : %D\n"
                       "  Multigrid:\n"
                       "    Type                               : %s\n"
                       "    Number of Levels                   : %d\n",
                       problemTypesForDisp[appCtx->problemChoice],
                       forcingTypesForDisp[appCtx->forcingChoice],
                       appCtx->meshFile[0] ? appCtx->meshFile : "Box Mesh",
                       appCtx->degree + 1, appCtx->degree + 1,
                       Ugsz[fineLevel]/ncompu, Ulsz[fineLevel]/ncompu, ncompu,
                       multigridTypesForDisp[appCtx->multigridChoice],
                       numLevels); CHKERRQ(ierr);

    if (appCtx->multigridChoice != MULTIGRID_NONE) {
      for (int i = 0; i < 2; i++) {
        CeedInt level = i ? fineLevel : 0;
        ierr = PetscPrintf(comm,
                           "    Level %D (%s):\n"
                           "      Number of 1D Basis Nodes (p)     : %d\n"
                           "      Global Nodes                     : %D\n"
                           "      Owned Nodes                      : %D\n",
                           level, i ? "fine" : "coarse",
                           appCtx->levelDegrees[level] + 1,
                           Ugsz[level]/ncompu, Ulsz[level]/ncompu);
        CHKERRQ(ierr);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Setup SNES
  // ---------------------------------------------------------------------------
  // Performance logging
  ierr = PetscLogStageRegister("SNES Setup Stage", &stageSnesSetup);
  CHKERRQ(ierr);
  ierr = PetscLogStagePush(stageSnesSetup); CHKERRQ(ierr);

  // Create SNES
  ierr = SNESCreate(comm, &snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes, levelDMs[fineLevel]); CHKERRQ(ierr);

  // -- Jacobian evaluators
  ierr = PetscMalloc1(numLevels, &jacobCtx); CHKERRQ(ierr);
  ierr = PetscMalloc1(numLevels, &jacobMat); CHKERRQ(ierr);
  for (int level = 0; level < numLevels; level++) {
    // -- Jacobian context for level
    ierr = PetscMalloc1(1, &jacobCtx[level]); CHKERRQ(ierr);
    ierr = SetupJacobianCtx(comm, appCtx, levelDMs[level], Ug[level],
                            Uloc[level], ceedData[level], ceed,
                            jacobCtx[level]); CHKERRQ(ierr);

    // -- Form Action of Jacobian on delta_u
    ierr = MatCreateShell(comm, Ulsz[level], Ulsz[level], Ugsz[level],
                          Ugsz[level], jacobCtx[level], &jacobMat[level]);
    CHKERRQ(ierr);
    ierr = MatShellSetOperation(jacobMat[level], MATOP_MULT,
                                (void (*)(void))ApplyJacobian_Ceed);
    CHKERRQ(ierr);
    ierr = MatShellSetOperation(jacobMat[level], MATOP_GET_DIAGONAL,
                                (void(*)(void))GetDiag_Ceed);

  }
  // Note: FormJacobian updates the state count of the Jacobian diagonals
  //   and assembles the Jpre matrix, if needed
  ierr = PetscMalloc1(1, &formJacobCtx); CHKERRQ(ierr);
  formJacobCtx->jacobCtx = jacobCtx;
  formJacobCtx->numLevels = numLevels;
  formJacobCtx->jacobMat = jacobMat;
  ierr = SNESSetJacobian(snes, jacobMat[fineLevel], jacobMat[fineLevel],
                         FormJacobian, formJacobCtx); CHKERRQ(ierr);

  // -- Residual evaluation function
  ierr = PetscMalloc1(1, &resCtx); CHKERRQ(ierr);
  ierr = PetscMemcpy(resCtx, jacobCtx[fineLevel],
                     sizeof(*jacobCtx[fineLevel])); CHKERRQ(ierr);
  resCtx->op = ceedData[fineLevel]->opApply;
  ierr = SNESSetFunction(snes, R, FormResidual_Ceed, resCtx); CHKERRQ(ierr);

  // -- Prolongation/Restriction evaluation
  ierr = PetscMalloc1(numLevels, &prolongRestrCtx); CHKERRQ(ierr);
  ierr = PetscMalloc1(numLevels, &prolongRestrMat); CHKERRQ(ierr);
  for (int level = 1; level < numLevels; level++) {
    // ---- Prolongation/restriction context for level
    ierr = PetscMalloc1(1, &prolongRestrCtx[level]); CHKERRQ(ierr);
    ierr = SetupProlongRestrictCtx(comm, levelDMs[level-1], levelDMs[level],
                                   Ug[level], Uloc[level-1], Uloc[level],
                                   ceedData[level-1], ceedData[level], ceed,
                                   prolongRestrCtx[level]); CHKERRQ(ierr);

    // ---- Form Action of Jacobian on delta_u
    ierr = MatCreateShell(comm, Ulsz[level], Ulsz[level-1], Ugsz[level],
                          Ugsz[level-1], prolongRestrCtx[level],
                          &prolongRestrMat[level]); CHKERRQ(ierr);
    // Note: In PCMG, restriction is the transpose of prolongation
    ierr = MatShellSetOperation(prolongRestrMat[level], MATOP_MULT,
                                (void (*)(void))Prolong_Ceed);
    ierr = MatShellSetOperation(prolongRestrMat[level], MATOP_MULT_TRANSPOSE,
                                (void (*)(void))Restrict_Ceed);
    CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Setup dummy SNES for AMG coarse solve
  // ---------------------------------------------------------------------------
  ierr = SNESCreate(comm, &snesCoarse); CHKERRQ(ierr);
  ierr = SNESSetDM(snesCoarse, levelDMs[0]); CHKERRQ(ierr);
  ierr = SNESSetSolution(snesCoarse, Ug[0]); CHKERRQ(ierr);

  // -- Jacobian matrix
  ierr = DMSetMatType(levelDMs[0], MATAIJ); CHKERRQ(ierr);
  ierr = DMCreateMatrix(levelDMs[0], &jacobMatCoarse); CHKERRQ(ierr);
  ierr = SNESSetJacobian(snesCoarse, jacobMatCoarse, jacobMatCoarse, NULL,
                         NULL); CHKERRQ(ierr);

  // -- Residual evaluation function
  ierr = PetscMalloc1(1, &jacobCoarseCtx); CHKERRQ(ierr);
  ierr = PetscMemcpy(jacobCoarseCtx, jacobCtx[0], sizeof(*jacobCtx[0]));
  CHKERRQ(ierr);
  ierr = SNESSetFunction(snesCoarse, Ug[0], ApplyJacobianCoarse_Ceed,
                         jacobCoarseCtx); CHKERRQ(ierr);

  // -- Update formJacobCtx
  formJacobCtx->Ucoarse = Ug[0];
  formJacobCtx->snesCoarse = snesCoarse;
  formJacobCtx->jacobMatCoarse = jacobMatCoarse;

  // ---------------------------------------------------------------------------
  // Setup KSP
  // ---------------------------------------------------------------------------
  {
    PC pc;
    KSP ksp;

    // -- KSP
    ierr = SNESGetKSP(snes, &ksp); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ksp, "outer_"); CHKERRQ(ierr);

    // -- Preconditioning
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetDM(pc, levelDMs[fineLevel]); CHKERRQ(ierr);
    ierr = PCSetOptionsPrefix(pc, "outer_"); CHKERRQ(ierr);

    if (appCtx->multigridChoice == MULTIGRID_NONE) {
      // ---- No Multigrid
      ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
      ierr = PCJacobiSetType(pc, PC_JACOBI_DIAGONAL); CHKERRQ(ierr);
    } else {
      // ---- PCMG
      ierr = PCSetType(pc, PCMG); CHKERRQ(ierr);

      // ------ PCMG levels
      ierr = PCMGSetLevels(pc, numLevels, NULL); CHKERRQ(ierr);
      for (int level = 0; level < numLevels; level++) {
        // -------- Smoother
        KSP kspSmoother, kspEst;
        PC pcSmoother;

        // ---------- Smoother KSP
        ierr = PCMGGetSmoother(pc, level, &kspSmoother); CHKERRQ(ierr);
        ierr = KSPSetDM(kspSmoother, levelDMs[level]); CHKERRQ(ierr);
        ierr = KSPSetDMActive(kspSmoother, PETSC_FALSE); CHKERRQ(ierr);

        // ---------- Chebyshev options
        ierr = KSPSetType(kspSmoother, KSPCHEBYSHEV); CHKERRQ(ierr);
        ierr = KSPChebyshevEstEigSet(kspSmoother, 0, 0.1, 0, 1.1);
        CHKERRQ(ierr);
        ierr = KSPChebyshevEstEigGetKSP(kspSmoother, &kspEst); CHKERRQ(ierr);
        ierr = KSPSetType(kspEst, KSPCG); CHKERRQ(ierr);
        ierr = KSPChebyshevEstEigSetUseNoisy(kspSmoother, PETSC_TRUE);
        CHKERRQ(ierr);
        ierr = KSPSetOperators(kspSmoother, jacobMat[level], jacobMat[level]);
        CHKERRQ(ierr);

        // ---------- Smoother preconditioner
        ierr = KSPGetPC(kspSmoother, &pcSmoother); CHKERRQ(ierr);
        ierr = PCSetType(pcSmoother, PCJACOBI); CHKERRQ(ierr);
        ierr = PCJacobiSetType(pcSmoother, PC_JACOBI_DIAGONAL); CHKERRQ(ierr);

        // -------- Work vector
        if (level != fineLevel) {
          ierr = PCMGSetX(pc, level, Ug[level]); CHKERRQ(ierr);
        }

        // -------- Level prolongation/restriction operator
        if (level > 0) {
          ierr = PCMGSetInterpolation(pc, level, prolongRestrMat[level]);
          CHKERRQ(ierr);
          ierr = PCMGSetRestriction(pc, level, prolongRestrMat[level]);
          CHKERRQ(ierr);
        }
      }

      // ------ PCMG coarse solve
      KSP kspCoarse;
      PC pcCoarse;

      // -------- Coarse KSP
      ierr = PCMGGetCoarseSolve(pc, &kspCoarse); CHKERRQ(ierr);
      ierr = KSPSetType(kspCoarse, KSPPREONLY); CHKERRQ(ierr);
      ierr = KSPSetOperators(kspCoarse, jacobMatCoarse, jacobMatCoarse);
      CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(kspCoarse, "coarse_"); CHKERRQ(ierr);

      // -------- Coarse preconditioner
      ierr = KSPGetPC(kspCoarse, &pcCoarse); CHKERRQ(ierr);
      ierr = PCSetType(pcCoarse, PCGAMG); CHKERRQ(ierr);
      ierr = PCSetOptionsPrefix(pcCoarse, "coarse_"); CHKERRQ(ierr);

      ierr = KSPSetFromOptions(kspCoarse); CHKERRQ(ierr);
      ierr = PCSetFromOptions(pcCoarse); CHKERRQ(ierr);

      // ------ PCMG options
      ierr = PCMGSetType(pc, PC_MG_MULTIPLICATIVE); CHKERRQ(ierr);
      ierr = PCMGSetNumberSmooth(pc, 3); CHKERRQ(ierr);
      ierr = PCMGSetCycleType(pc, pcmgCycleType); CHKERRQ(ierr);
    }
    ierr = KSPSetFromOptions(ksp);
    ierr = PCSetFromOptions(pc);
  }
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  // Performance logging
  ierr = PetscLogStagePop();

  // ---------------------------------------------------------------------------
  // Set initial guess
  // ---------------------------------------------------------------------------
  ierr = VecSet(U, 0.0); CHKERRQ(ierr);

  // View solution
  if (appCtx->viewSoln) {
    ierr = ViewSolution(comm, U, 0, 0.0); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Solve SNES
  // ---------------------------------------------------------------------------
  PetscBool snesMonitor = PETSC_FALSE;
  ierr = PetscOptionsHasName(NULL, NULL, "-snes_monitor", &snesMonitor);
  CHKERRQ(ierr);

  // Performance logging
  ierr = PetscLogStageRegister("SNES Solve Stage", &stageSnesSolve);
  CHKERRQ(ierr);
  ierr = PetscLogStagePush(stageSnesSolve); CHKERRQ(ierr);

  // Timing
  ierr = PetscBarrier((PetscObject)snes); CHKERRQ(ierr);
  startTime = MPI_Wtime();

  // Solve for each load increment
  PetscInt increment;
  for (increment = 1; increment <= appCtx->numIncrements; increment++) {
    // -- Log increment count
    if (snesMonitor) {
      ierr = PetscPrintf(comm, "%d Load Increment\n", increment - 1);
      CHKERRQ(ierr);
    }

    // -- Scale the problem
    PetscScalar loadIncrement = 1.0*increment / appCtx->numIncrements,
                scalingFactor = loadIncrement /
                                (increment == 1 ? 1 : resCtx->loadIncrement);
    resCtx->loadIncrement = loadIncrement;
    if (appCtx->numIncrements > 1 && appCtx->forcingChoice != FORCE_NONE) {
      ierr = VecScale(F, scalingFactor); CHKERRQ(ierr);
    }

    // -- Solve
    ierr = SNESSolve(snes, F, U); CHKERRQ(ierr);

    // -- View solution
    if (appCtx->viewSoln) {
      ierr = ViewSolution(comm, U, increment, loadIncrement); CHKERRQ(ierr);
    }

    // -- Update SNES iteration count
    PetscInt its;
    ierr = SNESGetIterationNumber(snes, &its); CHKERRQ(ierr);
    snesIts += its;
  }

  // Timing
  elapsedTime = MPI_Wtime() - startTime;

  // Performance logging
  ierr = PetscLogStagePop();

  // ---------------------------------------------------------------------------
  // Output summary
  // ---------------------------------------------------------------------------
  if (!appCtx->testMode) {
    // -- SNES
    SNESType snesType;
    SNESConvergedReason reason;
    PetscReal rnorm;
    ierr = SNESGetType(snes, &snesType); CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);
    ierr = SNESGetFunctionNorm(snes, &rnorm); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "  SNES:\n"
                       "    SNES Type                          : %s\n"
                       "    SNES Convergence                   : %s\n"
                       "    Number of Load Increments          : %d\n"
                       "    Completed Load Increments          : %d\n"
                       "    Total SNES Iterations              : %D\n"
                       "    Final rnorm                        : %e\n",
                       snesType, SNESConvergedReasons[reason],
                       appCtx->numIncrements, increment - 1,
                       snesIts, (double)rnorm); CHKERRQ(ierr);

    // -- KSP
    KSP ksp;
    KSPType kspType;
    ierr = SNESGetKSP(snes, &ksp); CHKERRQ(ierr);
    ierr = KSPGetType(ksp, &kspType); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "  Linear Solver:\n"
                       "    KSP Type                           : %s\n",
                       kspType); CHKERRQ(ierr);

    // -- PC
    if (appCtx->multigridChoice != MULTIGRID_NONE) {
      PC pc;
      PCMGType pcmgType;
      ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
      ierr = PCMGGetType(pc, &pcmgType); CHKERRQ(ierr);
      ierr = PetscPrintf(comm,
                         "  P-Multigrid:\n"
                         "    PCMG Type                          : %s\n"
                         "    PCMG Cycle Type                    : %s\n",
                         PCMGTypes[pcmgType],
                         PCMGCycleTypes[pcmgCycleType]); CHKERRQ(ierr);

      // -- Coarse Solve
      KSP kspCoarse;
      PC pcCoarse;
      PCType pcType;

      ierr = PCMGGetCoarseSolve(pc, &kspCoarse); CHKERRQ(ierr);
      ierr = KSPGetType(kspCoarse, &kspType); CHKERRQ(ierr);
      ierr = KSPGetPC(kspCoarse, &pcCoarse); CHKERRQ(ierr);
      ierr = PCGetType(pcCoarse, &pcType); CHKERRQ(ierr);
      ierr = PetscPrintf(comm,
                         "    Coarse Solve:\n"
                         "      KSP Type                         : %s\n"
                         "      PC Type                          : %s\n",
                         kspType, pcType); CHKERRQ(ierr);
    }
  }

  // ---------------------------------------------------------------------------
  // Compute solve time
  // ---------------------------------------------------------------------------
  if (!appCtx->testMode) {
    ierr = MPI_Allreduce(&elapsedTime, &minTime, 1, MPI_DOUBLE, MPI_MIN, comm);
    CHKERRQ(ierr);
    ierr = MPI_Allreduce(&elapsedTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, comm);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "  Performance:\n"
                       "    SNES Solve Time                    : %g (%g) sec\n",
                       maxTime, minTime); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Compute error
  // ---------------------------------------------------------------------------
  if (appCtx->forcingChoice == FORCE_MMS) {
    CeedScalar l2Error = 1., l2Unorm = 1.;
    const CeedScalar *truearray;
    Vec errorVec, trueVec;

    // -- Work vectors
    ierr = VecDuplicate(U, &errorVec); CHKERRQ(ierr);
    ierr = VecSet(errorVec, 0.0); CHKERRQ(ierr);
    ierr = VecDuplicate(U, &trueVec); CHKERRQ(ierr);
    ierr = VecSet(trueVec, 0.0); CHKERRQ(ierr);

    // -- Assemble global true solution vector
    CeedVectorGetArrayRead(ceedData[fineLevel]->truesoln, CEED_MEM_HOST,
                           &truearray);
    ierr = VecPlaceArray(resCtx->Yloc, truearray); CHKERRQ(ierr);
    ierr = DMLocalToGlobal(resCtx->dm, resCtx->Yloc, INSERT_VALUES, trueVec);
    CHKERRQ(ierr);
    ierr = VecResetArray(resCtx->Yloc); CHKERRQ(ierr);
    CeedVectorRestoreArrayRead(ceedData[fineLevel]->truesoln, &truearray);

    // -- Compute L2 error
    ierr = VecWAXPY(errorVec, -1.0, U, trueVec); CHKERRQ(ierr);
    ierr = VecNorm(errorVec, NORM_2, &l2Error); CHKERRQ(ierr);
    ierr = VecNorm(U, NORM_2, &l2Unorm); CHKERRQ(ierr);
    l2Error /= l2Unorm;

    // -- Output
    if (!appCtx->testMode || l2Error > 0.05) {
      ierr = PetscPrintf(comm,
                         "    L2 Error                           : %e\n",
                         l2Error); CHKERRQ(ierr);
    }

    // -- Cleanup
    ierr = VecDestroy(&errorVec); CHKERRQ(ierr);
    ierr = VecDestroy(&trueVec); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Free objects
  // ---------------------------------------------------------------------------
  // Data in arrays per level
  for (int level = 0; level < numLevels; level++) {
    // Vectors
    ierr = VecDestroy(&Ug[level]); CHKERRQ(ierr);
    ierr = VecDestroy(&Uloc[level]); CHKERRQ(ierr);

    // Jacobian matrix and data
    ierr = VecDestroy(&jacobCtx[level]->Yloc); CHKERRQ(ierr);
    ierr = MatDestroy(&jacobMat[level]); CHKERRQ(ierr);
    ierr = PetscFree(jacobCtx[level]); CHKERRQ(ierr);

    // Prolongation/Restriction matrix and data
    if (level > 0) {
      ierr = VecDestroy(&prolongRestrCtx[level]->multVec); CHKERRQ(ierr);
      ierr = PetscFree(prolongRestrCtx[level]); CHKERRQ(ierr);
      ierr = MatDestroy(&prolongRestrMat[level]); CHKERRQ(ierr);
    }

    // DM
    ierr = DMDestroy(&levelDMs[level]); CHKERRQ(ierr);

    // libCEED objects
    ierr = CeedDataDestroy(level, ceedData[level]); CHKERRQ(ierr);
  }

  // Arrays
  ierr = PetscFree(Ug); CHKERRQ(ierr);
  ierr = PetscFree(Uloc); CHKERRQ(ierr);
  ierr = PetscFree(Ugsz); CHKERRQ(ierr);
  ierr = PetscFree(Ulsz); CHKERRQ(ierr);
  ierr = PetscFree(Ulocsz); CHKERRQ(ierr);
  ierr = PetscFree(jacobCtx); CHKERRQ(ierr);
  ierr = PetscFree(jacobMat); CHKERRQ(ierr);
  ierr = PetscFree(prolongRestrCtx); CHKERRQ(ierr);
  ierr = PetscFree(prolongRestrMat); CHKERRQ(ierr);
  ierr = PetscFree(appCtx->levelDegrees); CHKERRQ(ierr);
  ierr = PetscFree(ceedData); CHKERRQ(ierr);

  // libCEED objects
  CeedQFunctionDestroy(&qfRestrict);
  CeedQFunctionDestroy(&qfProlong);
  CeedDestroy(&ceed);
  CeedDestroy(&ceedFine);

  // PETSc objects
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = VecDestroy(&R); CHKERRQ(ierr);
  ierr = VecDestroy(&Rloc); CHKERRQ(ierr);
  ierr = VecDestroy(&F); CHKERRQ(ierr);
  ierr = VecDestroy(&Floc); CHKERRQ(ierr);
  ierr = MatDestroy(&jacobMatCoarse); CHKERRQ(ierr);
  ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  ierr = SNESDestroy(&snesCoarse); CHKERRQ(ierr);
  ierr = DMDestroy(&dmOrig); CHKERRQ(ierr);
  ierr = PetscFree(levelDMs); CHKERRQ(ierr);

  // Structs
  ierr = PetscFree(resCtx); CHKERRQ(ierr);
  ierr = PetscFree(formJacobCtx); CHKERRQ(ierr);
  ierr = PetscFree(jacobCoarseCtx); CHKERRQ(ierr);
  ierr = PetscFree(appCtx); CHKERRQ(ierr);
  ierr = PetscFree(phys); CHKERRQ(ierr);
  ierr = PetscFree(units); CHKERRQ(ierr);

  return PetscFinalize();
}
