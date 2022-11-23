// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// DM setup for solid mechanics example using PETSc

#include "../include/setup-dm.h"

#include "../include/boundary.h"

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
PetscErrorCode CreateBCLabel(DM dm, const char name[]) {
  DMLabel label;

  PetscFunctionBeginUser;

  PetscCall(DMCreateLabel(dm, name));
  PetscCall(DMGetLabel(dm, name, &label));
  PetscCall(DMPlexMarkBoundaryFaces(dm, 1, label));

  PetscFunctionReturn(0);
};

// Read mesh and distribute DM in parallel
PetscErrorCode CreateDistributedDM(MPI_Comm comm, AppCtx app_ctx, DM *dm) {
  const char      *filename         = app_ctx->mesh_file;
  PetscBool        interpolate      = PETSC_TRUE;
  DM               distributed_mesh = NULL;
  PetscPartitioner part;

  PetscFunctionBeginUser;

  if (!*filename) {
    PetscInt dim = 3, faces[3] = {3, 3, 3};
    PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &dim, NULL));
    if (!dim) dim = 3;
    PetscCall(DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, NULL, NULL, NULL, interpolate, dm));
  } else {
    PetscCall(DMPlexCreateFromFile(comm, filename, NULL, interpolate, dm));
  }

  // Distribute DM in parallel
  PetscCall(DMPlexGetPartitioner(*dm, &part));
  PetscCall(PetscPartitionerSetFromOptions(part));
  PetscCall(DMPlexDistribute(*dm, 0, NULL, &distributed_mesh));
  if (distributed_mesh) {
    PetscCall(DMDestroy(dm));
    *dm = distributed_mesh;
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  PetscFunctionReturn(0);
};

// Setup DM with FE space of appropriate degree
PetscErrorCode SetupDMByDegree(DM dm, AppCtx app_ctx, PetscInt order, PetscBool boundary, PetscInt num_comp_u) {
  MPI_Comm        comm;
  PetscInt        dim;
  PetscFE         fe;
  IS              face_set_is;         // Index Set for Face Sets
  const char     *name = "Face Sets";  // PETSc internal requirement
  PetscInt        num_face_sets;       // Number of FaceSets in face_set_is
  const PetscInt *face_set_ids;        // id of each FaceSet

  PetscFunctionBeginUser;

  // Setup DM
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(PetscFECreateLagrange(comm, dim, num_comp_u, PETSC_FALSE, order, order, &fe));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  {
    /* create FE field for coordinates */
    PetscFE  fe_coords;
    PetscInt num_comp_coord;
    PetscCall(DMGetCoordinateDim(dm, &num_comp_coord));
    PetscCall(PetscFECreateLagrange(comm, dim, num_comp_coord, PETSC_FALSE, 1, 1, &fe_coords));
    PetscCall(DMProjectCoordinates(dm, fe_coords));
    PetscCall(PetscFEDestroy(&fe_coords));
  }

  // Add Dirichlet (Essential) boundary
  if (boundary) {
    if (app_ctx->forcing_choice == FORCE_MMS) {
      if (app_ctx->test_mode) {
        // -- Test mode - box mesh
        PetscBool has_label;
        PetscInt  marker_ids[1] = {1};
        PetscCall(DMHasLabel(dm, "marker", &has_label));
        if (!has_label) {
          PetscCall(CreateBCLabel(dm, "marker"));
        }
        DMLabel label;
        PetscCall(DMGetLabel(dm, "marker", &label));
        PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "mms", label, 1, marker_ids, 0, 0, NULL, (void (*)(void))BCMMS, NULL, NULL, NULL));
      } else {
        // -- ExodusII mesh with MMS
        PetscCall(DMGetLabelIdIS(dm, name, &face_set_is));
        PetscCall(ISGetSize(face_set_is, &num_face_sets));
        PetscCall(ISGetIndices(face_set_is, &face_set_ids));
        DMLabel label;
        PetscCall(DMGetLabel(dm, "Face Sets", &label));
        PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "mms", label, num_face_sets, face_set_ids, 0, 0, NULL, (void (*)(void))BCMMS, NULL, NULL, NULL));
        PetscCall(ISRestoreIndices(face_set_is, &face_set_ids));
        PetscCall(ISDestroy(&face_set_is));
      }
    } else {
      // -- Mesh with user specified BCs
      DMLabel label;
      PetscCall(DMGetLabel(dm, "Face Sets", &label));
      // -- Clamp BCs
      for (PetscInt i = 0; i < app_ctx->bc_clamp_count; i++) {
        char bcName[25];
        snprintf(bcName, sizeof bcName, "clamp_%" PetscInt_FMT, app_ctx->bc_clamp_faces[i]);
        PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, bcName, label, 1, &app_ctx->bc_clamp_faces[i], 0, 0, NULL, (void (*)(void))BCClamp, NULL,
                                (void *)&app_ctx->bc_clamp_max[i], NULL));
      }
    }
  }
  PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));

  // Cleanup
  PetscCall(PetscFEDestroy(&fe));

  PetscFunctionReturn(0);
};
