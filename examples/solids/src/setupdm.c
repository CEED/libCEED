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

#include "../elasticity.h"

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
PetscErrorCode CreateBCLabel(DM dm, const char name[]) {
  PetscErrorCode ierr;
  DMLabel label;

  PetscFunctionBeginUser;

  ierr = DMCreateLabel(dm, name); CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label); CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, 1, label); CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm, label); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// Read mesh and distribute DM in parallel
PetscErrorCode CreateDistributedDM(MPI_Comm comm, AppCtx appCtx, DM *dm) {
  PetscErrorCode  ierr;
  const char      *filename = appCtx->meshFile;
  // Note: interpolate if polynomial degree > 1
  PetscBool       interpolate = PETSC_TRUE;
  DM              distributedMesh = NULL;
  PetscPartitioner part;

  PetscFunctionBeginUser;

  // Read mesh
  if (appCtx->degree >= 2)
    interpolate = PETSC_TRUE;

  if (appCtx->testMode) {
    PetscInt dim = 3, cells[3] = {3, 3, 3};
    ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, cells, NULL,
                               NULL, NULL, interpolate, dm); CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm); CHKERRQ(ierr);
  }

  // Distribute DM in parallel
  ierr = DMPlexGetPartitioner(*dm, &part); CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
  ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh); CHKERRQ(ierr);
  if (distributedMesh) {
    ierr = DMDestroy(dm); CHKERRQ(ierr);
    *dm  = distributedMesh;
  }

  PetscFunctionReturn(0);
};

// Setup DM with FE space of appropriate degree
PetscErrorCode SetupDMByDegree(DM dm, AppCtx appCtx, PetscInt order,
                               PetscInt ncompu) {
  PetscErrorCode  ierr;
  PetscInt        dim;
  PetscSpace      P;
  PetscDualSpace  Q;
  DM              K;
  PetscFE         fe;
  PetscInt        quadPointsPerEdge;
  PetscQuadrature q;  // quadrature points
  PetscQuadrature fq; // face quadrature points (For future: Nuemman boundary)
  // Variables for Dirichlet (Essential) Boundary
  IS              faceSetIS;           // Index Set for Face Sets
  const char      *name = "Face Sets"; // PETSc internal requirement
  PetscInt        numFaceSets;         // Number of FaceSets in faceSetIS
  const PetscInt  *faceSetIds;         // id of each FaceSet

  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim);

  // Setup FE space by polynomial order
  // Note: This is a modification of the built in PETSc function
  //         PetscFECreateDefault(). PETSc declined to add a 'degree' option.
  // -- Setup FE space (Space P) for tensor polynomials
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) dm), &P); CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetTensor(P, PETSC_TRUE); CHKERRQ(ierr);
  ierr = PetscSpaceSetFromOptions(P); CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, ncompu); CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(P, dim); CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(P, order, order); CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P); CHKERRQ(ierr);
  // -- Setup FE dual space (Space Q) for tensor polynomials
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) dm), &Q);
  CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(Q,PETSCDUALSPACELAGRANGE); CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, PETSC_FALSE, &K);
  CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K); CHKERRQ(ierr);
  ierr = DMDestroy(&K); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(Q, ncompu); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order); CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(Q, PETSC_TRUE); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(Q); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q); CHKERRQ(ierr);
  // -- Create element
  ierr = PetscFECreate(PetscObjectComm((PetscObject) dm), &fe); CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(fe); CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(fe, P); CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(fe, Q); CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(fe, ncompu); CHKERRQ(ierr);
  ierr = PetscFESetUp(fe); CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P); CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q); CHKERRQ(ierr);
  // -- Create quadrature
  quadPointsPerEdge = PetscMax(order + 1,1);
  ierr = PetscDTGaussTensorQuadrature(dim, 1, quadPointsPerEdge, -1.0, 1.0, &q);
  CHKERRQ(ierr);
  ierr = PetscDTGaussTensorQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0,
                                      &fq); CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fe, q); CHKERRQ(ierr);
  ierr = PetscFESetFaceQuadrature(fe, fq); CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q); CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fq); CHKERRQ(ierr);
  // -- End of modified PetscFECreateDefault()

  // Setup DM
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMAddField(dm, NULL, (PetscObject)fe); CHKERRQ(ierr);
  ierr = DMCreateDS(dm); CHKERRQ(ierr);

  // Add Dirichlet (Essential) boundary
  if (appCtx->testMode) {
    // -- Test mode - box mesh
    PetscBool hasLabel;
    PetscInt marker_ids[1] = {1};
    DMHasLabel(dm, "marker", &hasLabel);
    if (!hasLabel) {
      ierr = CreateBCLabel(dm, "marker"); CHKERRQ(ierr);
    }
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL,
                         (void(*)(void))boundaryOptions[appCtx->boundaryChoice],
                         1, marker_ids, NULL); CHKERRQ(ierr);
  } else {
    // -- ExodusII mesh
    ierr = DMGetLabelIdIS(dm, name, &faceSetIS); CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceSetIS,&numFaceSets); CHKERRQ(ierr);
    ierr = ISGetIndices(faceSetIS, &faceSetIds); CHKERRQ(ierr);

    for (PetscInt faceSet = 0; faceSet < numFaceSets; faceSet++) {
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, NULL, "Face Sets", 0, 0, NULL,
                           (void(*)(void))boundaryOptions[appCtx->boundaryChoice],
                           1, &faceSetIds[faceSet],
                           (void *)(&faceSetIds[faceSet])); CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(faceSetIS, &faceSetIds); CHKERRQ(ierr);
    ierr = ISDestroy(&faceSetIS); CHKERRQ(ierr);
  }
  ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  // Cleanup
  ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};
