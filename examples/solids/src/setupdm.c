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

/// @file
/// DM setup for solid mechanics example using PETSc

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
PetscErrorCode CreateDistributedDM(MPI_Comm comm, AppCtx app_ctx, DM *dm) {
  PetscErrorCode  ierr;
  const char      *filename = app_ctx->mesh_file;
  PetscBool       interpolate = PETSC_TRUE;
  DM              distributed_mesh = NULL;
  PetscPartitioner part;

  PetscFunctionBeginUser;

  if (!*filename) {
    PetscInt dim = 3, faces[3] = {3, 3, 3};
    ierr = PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces",
                                   faces, &dim, NULL); CHKERRQ(ierr);
    ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, NULL,
                               NULL, NULL, interpolate, dm); CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm); CHKERRQ(ierr);
  }

  // Distribute DM in parallel
  ierr = DMPlexGetPartitioner(*dm, &part); CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
  ierr = DMPlexDistribute(*dm, 0, NULL, &distributed_mesh); CHKERRQ(ierr);
  if (distributed_mesh) {
    ierr = DMDestroy(dm); CHKERRQ(ierr);
    *dm  = distributed_mesh;
  }
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view"); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// Create FE by degree
PetscErrorCode PetscFECreateByDegree(DM dm, PetscInt dim, PetscInt Nc,
                                     PetscBool isSimplex, const char prefix[],
                                     PetscInt order, PetscFE *fem) {
  PetscQuadrature q, fq;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscInt        quadPointsPerEdge;
  PetscBool       tensor = isSimplex ? PETSC_FALSE : PETSC_TRUE;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create space */
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) dm), &P); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) P, prefix); CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetTensor(P, tensor); CHKERRQ(ierr);
  ierr = PetscSpaceSetFromOptions(P); CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, Nc); CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(P, dim); CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(P, order, order); CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P); CHKERRQ(ierr);
  ierr = PetscSpacePolynomialGetTensor(P, &tensor); CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) dm), &Q);
  CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(Q,PETSCDUALSPACELAGRANGE); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) Q, prefix); CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, isSimplex, &K); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K); CHKERRQ(ierr);
  ierr = DMDestroy(&K); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(Q, Nc); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order); CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(Q, tensor); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(Q); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q); CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PetscObjectComm((PetscObject) dm), fem); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *fem, prefix); CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(*fem); CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*fem, P); CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*fem, Q); CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*fem, Nc); CHKERRQ(ierr);
  ierr = PetscFESetUp(*fem); CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P); CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q); CHKERRQ(ierr);
  /* Create quadrature */
  quadPointsPerEdge = PetscMax(order + 1,1);
  if (isSimplex) {
    ierr = PetscDTStroudConicalQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0,
                                          &q); CHKERRQ(ierr);
    ierr = PetscDTStroudConicalQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0,
                                          &fq); CHKERRQ(ierr);
  } else {
    ierr = PetscDTGaussTensorQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0,
                                        &q); CHKERRQ(ierr);
    ierr = PetscDTGaussTensorQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0,
                                        &fq); CHKERRQ(ierr);
  }
  ierr = PetscFESetQuadrature(*fem, q); CHKERRQ(ierr);
  ierr = PetscFESetFaceQuadrature(*fem, fq); CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q); CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fq); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Setup DM with FE space of appropriate degree
PetscErrorCode SetupDMByDegree(DM dm, AppCtx app_ctx, PetscInt order,
                               PetscBool boundary, PetscInt num_comp_u) {
  PetscErrorCode  ierr;
  PetscInt        dim;
  PetscFE         fe;
  IS              face_set_is;         // Index Set for Face Sets
  const char      *name = "Face Sets"; // PETSc internal requirement
  PetscInt        num_face_sets;       // Number of FaceSets in face_set_is
  const PetscInt  *face_set_ids;       // id of each FaceSet

  PetscFunctionBeginUser;

  // Setup DM
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ierr = PetscFECreateByDegree(dm, dim, num_comp_u, PETSC_FALSE, NULL, order,
                               &fe);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMAddField(dm, NULL, (PetscObject)fe); CHKERRQ(ierr);
  ierr = DMCreateDS(dm); CHKERRQ(ierr);

  // Add Dirichlet (Essential) boundary
  if (boundary) {
    if (app_ctx->forcing_choice == FORCE_MMS) {
      if (app_ctx->test_mode) {
        // -- Test mode - box mesh
        PetscBool has_label;
        PetscInt marker_ids[1] = {1};
        ierr = DMHasLabel(dm, "marker", &has_label); CHKERRQ(ierr);
        if (!has_label) {
          ierr = CreateBCLabel(dm, "marker"); CHKERRQ(ierr);
        }
        DMLabel label;
        ierr = DMGetLabel(dm, "marker", &label); CHKERRQ(ierr);
        ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "mms", label, "marker", 1, marker_ids,
                             0, 0, NULL, (void(*)(void))BCMMS, NULL, NULL, NULL);
        CHKERRQ(ierr);
      } else {
        // -- ExodusII mesh with MMS
        ierr = DMGetLabelIdIS(dm, name, &face_set_is); CHKERRQ(ierr);
        ierr = ISGetSize(face_set_is,&num_face_sets); CHKERRQ(ierr);
        ierr = ISGetIndices(face_set_is, &face_set_ids); CHKERRQ(ierr);
        DMLabel label;
        ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
        ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "mms", label, "Face Sets",
                             num_face_sets, face_set_ids, 0, 0, NULL,
                             (void(*)(void))BCMMS, NULL, NULL, NULL);
        CHKERRQ(ierr);
        ierr = ISRestoreIndices(face_set_is, &face_set_ids); CHKERRQ(ierr);
        ierr = ISDestroy(&face_set_is); CHKERRQ(ierr);
      }
    } else {
      // -- ExodusII mesh with user specified BCs
      // -- Clamp BCs
      DMLabel label;
      ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
      for (PetscInt i = 0; i < app_ctx->bc_clamp_count; i++) {
        char bcName[25];
        snprintf(bcName, sizeof bcName, "clamp_%d", app_ctx->bc_clamp_faces[i]);
        ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, bcName, label, "Face Sets", 1,
                             &app_ctx->bc_clamp_faces[i], 0, 0,
                             NULL, (void(*)(void))BCClamp, NULL,
                             (void *)&app_ctx->bc_clamp_max[i], NULL);
        CHKERRQ(ierr);
      }
    }
  }
  ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  // Cleanup
  ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};
