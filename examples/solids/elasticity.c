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
//     ./elasticity -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -bc_clamp 998,999 -bc_clamp_998_translate 0.1,0.2,0.3 -problem hyperSS -forcing none -ceed /cpu/self
//     ./elasticity -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -bc_clamp 998,999 -bc_clamp_998_rotate 1,0,0,0.2 -problem hyperFS -forcing none -ceed /gpu/occa
//
// Sample meshes can be found at https://github.com/jeremylt/ceedSampleMeshes
//
//TESTARGS -ceed {ceed_resource} -test -degree 3 -nu 0.3 -E 1 -dm_plex_box_faces 3,3,3

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
  Physics        physSmoother = NULL;    // Separate context if nuSmoother set
  Units          units;                  // Contains units scaling
  // PETSc objects
  PetscLogStage  stageDMSetup, stageLibceedSetup,
                 stageSnesSetup, stageSnesSolve;
  DM             dmOrig;                 // Distributed DM to clone
  DM             dmEnergy, dmDiagnostic; // DMs for postprocessing
  DM             *levelDMs;
  Vec            U, *Ug, *Uloc;          // U: solution, R: residual, F: forcing
  Vec            R, Rloc, F, Floc;       // g: global, loc: local
  SNES           snes, snesCoarse = NULL;
  Mat            *jacobMat, jacobMatCoarse, *prolongRestrMat;
  // PETSc data
  UserMult       resCtx, jacobCoarseCtx = NULL, *jacobCtx;
  FormJacobCtx   formJacobCtx;
  UserMultProlongRestr *prolongRestrCtx;
  PCMGCycleType  pcmgCycleType = PC_MG_CYCLE_V;
  // libCEED objects
  Ceed           ceed;
  CeedData       *ceedData;
  CeedQFunction  qfRestrict = NULL, qfProlong = NULL;
  // Parameters
  PetscInt       ncompu = 3;             // 3 DoFs in 3D
  PetscInt       ncompe = 1, ncompd = 5; // 1 energy output, 5 diagnostic
  PetscInt       numLevels = 1, fineLevel = 0;
  PetscInt       *Ugsz, *Ulsz, *Ulocsz;  // sz: size
  PetscInt       snesIts = 0, kspIts = 0;
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
  if (fabs(appCtx->nuSmoother) > 1E-14) {
    ierr = PetscMalloc1(1, &physSmoother); CHKERRQ(ierr);
    ierr = PetscMemcpy(physSmoother, phys, sizeof(*phys)); CHKERRQ(ierr);
    physSmoother->nu = appCtx->nuSmoother;
  }

  // ---------------------------------------------------------------------------
  // Initalize libCEED
  // ---------------------------------------------------------------------------
  // Initalize backend
  CeedInit(appCtx->ceedResource, &ceed);

  // Check preferred MemType
  CeedMemType memTypeBackend;
  CeedGetPreferredMemType(ceed, &memTypeBackend);
  if (!appCtx->setMemTypeRequest)
    appCtx->memTypeRequested = memTypeBackend;
  else if (!appCtx->petscHaveCuda && appCtx->memTypeRequested == CEED_MEM_DEVICE)
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP_SYS,
             "PETSc was not built with CUDA. "
             "Requested MemType CEED_MEM_DEVICE is not supported.", NULL);

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
  for (PetscInt level = 0; level < numLevels; level++) {
    ierr = DMClone(dmOrig, &levelDMs[level]); CHKERRQ(ierr);
    if (appCtx->memTypeRequested == CEED_MEM_DEVICE) {
      ierr = DMSetVecType(levelDMs[level], VECCUDA); CHKERRQ(ierr);
    }
    ierr = SetupDMByDegree(levelDMs[level], appCtx, appCtx->levelDegrees[level],
                           PETSC_TRUE, ncompu); CHKERRQ(ierr);
    // -- Label field components for viewing
    // Empty name for conserved field (because there is only one field)
    PetscSection section;
    ierr = DMGetLocalSection(levelDMs[level], &section); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, 0, "Displacement"); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 0, "DisplacementX");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 1, "DisplacementY");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 2, "DisplacementZ");
    CHKERRQ(ierr);
  }

  // -- Setup postprocessing DMs
  ierr = DMClone(dmOrig, &dmEnergy); CHKERRQ(ierr);
  ierr = SetupDMByDegree(dmEnergy, appCtx, appCtx->levelDegrees[fineLevel],
                         PETSC_FALSE, ncompe); CHKERRQ(ierr);
  ierr = DMClone(dmOrig, &dmDiagnostic); CHKERRQ(ierr);
  ierr = SetupDMByDegree(dmDiagnostic, appCtx, appCtx->levelDegrees[fineLevel],
                         PETSC_FALSE, ncompu + ncompd); CHKERRQ(ierr);
  if (appCtx->memTypeRequested == CEED_MEM_DEVICE) {
    ierr = DMSetVecType(dmEnergy, VECCUDA); CHKERRQ(ierr);
    ierr = DMSetVecType(dmDiagnostic, VECCUDA); CHKERRQ(ierr);
  }
  {
    // -- Label field components for viewing
    // Empty name for conserved field (because there is only one field)
    PetscSection section;
    ierr = DMGetLocalSection(dmDiagnostic, &section); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, 0, "Diagnostics"); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 0, "DisplacementX");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 1, "DisplacementY");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 2, "DisplacementZ");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 3, "Pressure");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 4, "VolumentricStrain");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 5, "TraceE2");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 6, "detJ");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 7, "StrainEnergyDensity");
    CHKERRQ(ierr);
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
  for (PetscInt level = 0; level < numLevels; level++) {
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

  // -- Create libCEED local forcing vector
  CeedVector forceCeed;
  CeedScalar *f;
  if (appCtx->forcingChoice != FORCE_NONE) {
    if (appCtx->memTypeRequested == CEED_MEM_HOST) {
      ierr = VecGetArray(Floc, &f); CHKERRQ(ierr);
    } else {
      ierr = VecCUDAGetArray(Floc, &f); CHKERRQ(ierr);
    }
    CeedVectorCreate(ceed, Ulocsz[fineLevel], &forceCeed);
    CeedVectorSetArray(forceCeed, appCtx->memTypeRequested, CEED_USE_POINTER, f);
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
  // ---- Setup residual, Jacobian evaluator and geometric information
  ierr = PetscCalloc1(1, &ceedData[fineLevel]); CHKERRQ(ierr);
  {
    ierr = SetupLibceedFineLevel(levelDMs[fineLevel], dmEnergy, dmDiagnostic,
                                 ceed, appCtx, phys, ceedData, fineLevel,
                                 ncompu, Ugsz[fineLevel], Ulocsz[fineLevel],
                                 forceCeed, qfRestrict, qfProlong);
    CHKERRQ(ierr);
  }
  // ---- Setup coarse Jacobian evaluator and prolongation/restriction
  for (PetscInt level = numLevels - 2; level >= 0; level--) {
    ierr = PetscCalloc1(1, &ceedData[level]); CHKERRQ(ierr);

    // Get global communication restriction
    ierr = VecZeroEntries(Ug[level+1]); CHKERRQ(ierr);
    ierr = VecSet(Uloc[level+1], 1.0); CHKERRQ(ierr);
    ierr = DMLocalToGlobal(levelDMs[level+1], Uloc[level+1], ADD_VALUES,
                           Ug[level+1]); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(levelDMs[level+1], Ug[level+1], INSERT_VALUES,
                           Uloc[level+1]); CHKERRQ(ierr);

    // Place in libCEED array
    const PetscScalar *m;
    if (appCtx->memTypeRequested == CEED_MEM_HOST) {
      ierr = VecGetArrayRead(Uloc[level+1], &m); CHKERRQ(ierr);
    } else {
      ierr = VecCUDAGetArrayRead(Uloc[level+1], &m); CHKERRQ(ierr);
    }
    CeedVectorSetArray(ceedData[level+1]->xceed, appCtx->memTypeRequested,
                       CEED_USE_POINTER, (CeedScalar *)m);

    // Note: use high order ceed, if specified and degree > 4
    ierr = SetupLibceedLevel(levelDMs[level], ceed, appCtx, phys,
                             ceedData, level, ncompu, Ugsz[level],
                             Ulocsz[level], ceedData[level+1]->xceed, qfRestrict,
                             qfProlong); CHKERRQ(ierr);

    // Restore PETSc vector
    CeedVectorTakeArray(ceedData[level+1]->xceed, appCtx->memTypeRequested,
                        (CeedScalar **)&m);
    if (appCtx->memTypeRequested == CEED_MEM_HOST) {
      ierr = VecRestoreArrayRead(Uloc[level+1], &m); CHKERRQ(ierr);
    } else {
      ierr = VecCUDARestoreArrayRead(Uloc[level+1], &m); CHKERRQ(ierr);
    }
    ierr = VecZeroEntries(Ug[level+1]); CHKERRQ(ierr);
    ierr = VecZeroEntries(Uloc[level+1]); CHKERRQ(ierr);
  }

  // Performance logging
  ierr = PetscLogStagePop();

  // ---------------------------------------------------------------------------
  // Setup global forcing vector
  // ---------------------------------------------------------------------------
  ierr = VecZeroEntries(F); CHKERRQ(ierr);

  if (appCtx->forcingChoice != FORCE_NONE) {
    CeedVectorTakeArray(forceCeed, appCtx->memTypeRequested, NULL);
    if (appCtx->memTypeRequested == CEED_MEM_HOST) {
      ierr = VecRestoreArray(Floc, &f); CHKERRQ(ierr);
    } else {
      ierr = VecCUDARestoreArray(Floc, &f); CHKERRQ(ierr);
    }
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
                       "\n-- Elasticity Example - libCEED + PETSc --\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "    libCEED User Requested MemType     : %s\n",
                       usedresource, CeedMemTypes[memTypeBackend],
                       (appCtx->setMemTypeRequest) ?
                       CeedMemTypes[appCtx->memTypeRequested] : "none");
    CHKERRQ(ierr);

    VecType vecType;
    ierr = VecGetType(U, &vecType); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "  PETSc:\n"
                       "    PETSc Vec Type                     : %s\n",
                       vecType); CHKERRQ(ierr);

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
                       (appCtx->degree == 1 &&
                        appCtx->multigridChoice != MULTIGRID_NONE) ?
                       "Algebraic multigrid" :
                       multigridTypesForDisp[appCtx->multigridChoice],
                       (appCtx->degree == 1 ||
                        appCtx->multigridChoice == MULTIGRID_NONE) ?
                       0 : numLevels); CHKERRQ(ierr);

    if (appCtx->multigridChoice != MULTIGRID_NONE) {
      for (PetscInt i = 0; i < 2; i++) {
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
  for (PetscInt level = 0; level < numLevels; level++) {
    // -- Jacobian context for level
    ierr = PetscMalloc1(1, &jacobCtx[level]); CHKERRQ(ierr);
    ierr = SetupJacobianCtx(comm, appCtx, levelDMs[level], Ug[level],
                            Uloc[level], ceedData[level], ceed, phys,
                            physSmoother, jacobCtx[level]); CHKERRQ(ierr);

    // -- Form Action of Jacobian on delta_u
    ierr = MatCreateShell(comm, Ulsz[level], Ulsz[level], Ugsz[level],
                          Ugsz[level], jacobCtx[level], &jacobMat[level]);
    CHKERRQ(ierr);
    ierr = MatShellSetOperation(jacobMat[level], MATOP_MULT,
                                (void (*)(void))ApplyJacobian_Ceed);
    CHKERRQ(ierr);
    ierr = MatShellSetOperation(jacobMat[level], MATOP_GET_DIAGONAL,
                                (void(*)(void))GetDiag_Ceed);
    if (appCtx->memTypeRequested == CEED_MEM_DEVICE) {
      ierr = MatShellSetVecType(jacobMat[level], VECCUDA); CHKERRQ(ierr);
    }
  }
  // Note: FormJacobian updates Jacobian matrices on each level
  //   and assembles the Jpre matrix, if needed
  ierr = PetscMalloc1(1, &formJacobCtx); CHKERRQ(ierr);
  formJacobCtx->jacobCtx = jacobCtx;
  formJacobCtx->numLevels = numLevels;
  formJacobCtx->jacobMat = jacobMat;

  // -- Residual evaluation function
  ierr = PetscMalloc1(1, &resCtx); CHKERRQ(ierr);
  ierr = PetscMemcpy(resCtx, jacobCtx[fineLevel],
                     sizeof(*jacobCtx[fineLevel])); CHKERRQ(ierr);
  resCtx->op = ceedData[fineLevel]->opApply;
  resCtx->qf = ceedData[fineLevel]->qfApply;
  ierr = SNESSetFunction(snes, R, FormResidual_Ceed, resCtx); CHKERRQ(ierr);

  // -- Prolongation/Restriction evaluation
  ierr = PetscMalloc1(numLevels, &prolongRestrCtx); CHKERRQ(ierr);
  ierr = PetscMalloc1(numLevels, &prolongRestrMat); CHKERRQ(ierr);
  for (PetscInt level = 1; level < numLevels; level++) {
    // ---- Prolongation/restriction context for level
    ierr = PetscMalloc1(1, &prolongRestrCtx[level]); CHKERRQ(ierr);
    ierr = SetupProlongRestrictCtx(comm, appCtx, levelDMs[level-1],
                                   levelDMs[level], Ug[level], Uloc[level-1],
                                   Uloc[level], ceedData[level-1],
                                   ceedData[level], ceed,
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
    if (appCtx->memTypeRequested == CEED_MEM_DEVICE) {
      ierr = MatShellSetVecType(prolongRestrMat[level], VECCUDA); CHKERRQ(ierr);
    }
  }

  // ---------------------------------------------------------------------------
  // Setup dummy SNES for AMG coarse solve
  // ---------------------------------------------------------------------------
  if (appCtx->multigridChoice != MULTIGRID_NONE) {
    // -- Jacobian Matrix
    ierr = DMSetMatType(levelDMs[0], MATAIJ); CHKERRQ(ierr);
    ierr = DMCreateMatrix(levelDMs[0], &jacobMatCoarse); CHKERRQ(ierr);

    if (appCtx->degree > 1) {
      ierr = SNESCreate(comm, &snesCoarse); CHKERRQ(ierr);
      ierr = SNESSetDM(snesCoarse, levelDMs[0]); CHKERRQ(ierr);
      ierr = SNESSetSolution(snesCoarse, Ug[0]); CHKERRQ(ierr);

      // -- Jacobian function
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
    }
  }

  // Set Jacobian function
  if (appCtx->degree > 1) {
    ierr = SNESSetJacobian(snes, jacobMat[fineLevel], jacobMat[fineLevel],
                           FormJacobian, formJacobCtx); CHKERRQ(ierr);
  } else {
    ierr = SNESSetJacobian(snes, jacobMat[0], jacobMatCoarse,
                           SNESComputeJacobianDefaultColor, NULL);
    CHKERRQ(ierr);
  }

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
    } else if (appCtx->degree == 1) {
      // ---- AMG for degree 1
      ierr = PCSetType(pc, PCGAMG); CHKERRQ(ierr);
    } else {
      // ---- PCMG
      ierr = PCSetType(pc, PCMG); CHKERRQ(ierr);

      // ------ PCMG levels
      ierr = PCMGSetLevels(pc, numLevels, NULL); CHKERRQ(ierr);
      for (PetscInt level = 0; level < numLevels; level++) {
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
  {
    // Default to critical-point (CP) line search (related to Wolfe's curvature condition)
    SNESLineSearch linesearch;

    ierr = SNESGetLineSearch(snes, &linesearch); CHKERRQ(ierr);
    ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHCP); CHKERRQ(ierr);
  }

  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  // Performance logging
  ierr = PetscLogStagePop();

  // ---------------------------------------------------------------------------
  // Set initial guess
  // ---------------------------------------------------------------------------
  ierr = PetscObjectSetName((PetscObject)U, ""); CHKERRQ(ierr);
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
    ierr = SNESGetLinearSolveIterations(snes, &its); CHKERRQ(ierr);
    kspIts += its;

    // -- Check for divergence
    SNESConvergedReason reason;
    ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);
    if (reason < 0)
      break;
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
                       "    KSP Type                           : %s\n"
                       "    Total KSP Iterations               : %D\n",
                       kspType, kspIts); CHKERRQ(ierr);

    // -- PC
    PC pc;
    PCType pcType;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCGetType(pc, &pcType); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "    PC Type                            : %s\n",
                       pcType); CHKERRQ(ierr);

    if (!strcmp(pcType, PCMG)) {
      PCMGType pcmgType;
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
                       "    SNES Solve Time                    : %g (%g) sec\n"
                       "    DoFs/Sec in SNES                   : %g (%g) million\n",
                       maxTime, minTime, 1e-6*Ugsz[fineLevel]*kspIts/maxTime,
                       1e-6*Ugsz[fineLevel]*kspIts/minTime); CHKERRQ(ierr);
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
    CeedVectorGetArrayRead(ceedData[fineLevel]->truesoln,
                           appCtx->memTypeRequested, &truearray);
    if (appCtx->memTypeRequested == CEED_MEM_HOST) {
      ierr = VecPlaceArray(resCtx->Yloc, (PetscScalar *)truearray);
      CHKERRQ(ierr);
    } else {
      ierr = VecCUDAPlaceArray(resCtx->Yloc, (PetscScalar *)truearray);
      CHKERRQ(ierr);
    }
    ierr = DMLocalToGlobal(resCtx->dm, resCtx->Yloc, INSERT_VALUES, trueVec);
    CHKERRQ(ierr);
    if (appCtx->memTypeRequested == CEED_MEM_HOST) {
      ierr = VecResetArray(resCtx->Yloc); CHKERRQ(ierr);
    } else {
      ierr = VecCUDAResetArray(resCtx->Yloc); CHKERRQ(ierr);
    }
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
  // Compute energy
  // ---------------------------------------------------------------------------
  if (!appCtx->testMode) {
    // -- Compute L2 error
    CeedScalar energy;
    ierr = ComputeStrainEnergy(dmEnergy, resCtx, ceedData[fineLevel]->opEnergy,
                               U, &energy); CHKERRQ(ierr);

    // -- Output
    ierr = PetscPrintf(comm,
                       "    Strain Energy                      : %e\n",
                       energy); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Output diagnostic quantities
  // ---------------------------------------------------------------------------
  if (appCtx->viewSoln || appCtx->viewFinalSoln) {
    // -- Setup context
    UserMult diagnosticCtx;
    ierr = PetscMalloc1(1, &diagnosticCtx); CHKERRQ(ierr);
    ierr = PetscMemcpy(diagnosticCtx, resCtx, sizeof(*resCtx)); CHKERRQ(ierr);
    diagnosticCtx->dm = dmDiagnostic;
    diagnosticCtx->op = ceedData[fineLevel]->opDiagnostic;

    // -- Compute and output
    ierr = ViewDiagnosticQuantities(comm, levelDMs[fineLevel], diagnosticCtx, U,
                                    ceedData[fineLevel]->ErestrictDiagnostic);
    CHKERRQ(ierr);

    // -- Cleanup
    ierr = PetscFree(diagnosticCtx); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Free objects
  // ---------------------------------------------------------------------------
  // Data in arrays per level
  for (PetscInt level = 0; level < numLevels; level++) {
    // Vectors
    ierr = VecDestroy(&Ug[level]); CHKERRQ(ierr);
    ierr = VecDestroy(&Uloc[level]); CHKERRQ(ierr);

    // Jacobian matrix and data
    ierr = VecDestroy(&jacobCtx[level]->Yloc); CHKERRQ(ierr);
    ierr = MatDestroy(&jacobMat[level]); CHKERRQ(ierr);
    ierr = PetscFree(jacobCtx[level]); CHKERRQ(ierr);

    // Prolongation/Restriction matrix and data
    if (level > 0) {
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
  ierr = DMDestroy(&dmEnergy); CHKERRQ(ierr);
  ierr = DMDestroy(&dmDiagnostic); CHKERRQ(ierr);
  ierr = PetscFree(levelDMs); CHKERRQ(ierr);

  // Structs
  ierr = PetscFree(resCtx); CHKERRQ(ierr);
  ierr = PetscFree(formJacobCtx); CHKERRQ(ierr);
  ierr = PetscFree(jacobCoarseCtx); CHKERRQ(ierr);
  ierr = PetscFree(appCtx); CHKERRQ(ierr);
  ierr = PetscFree(phys); CHKERRQ(ierr);
  ierr = PetscFree(physSmoother); CHKERRQ(ierr);
  ierr = PetscFree(units); CHKERRQ(ierr);

  return PetscFinalize();
}
