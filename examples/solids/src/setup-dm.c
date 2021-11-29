/// @file
/// DM setup for solid mechanics example using PETSc

#include "../include/boundary.h"
#include "../include/setup-dm.h"
#include "../include/petsc-macros.h"

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
    if (!dim) dim = 3;
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

// Setup DM with FE space of appropriate degree
PetscErrorCode SetupDMByDegree(DM dm, AppCtx app_ctx, PetscInt order,
                               PetscBool boundary, PetscInt num_comp_u) {
  PetscErrorCode  ierr;
  MPI_Comm        comm;
  PetscInt        dim;
  PetscFE         fe;
  IS              face_set_is;         // Index Set for Face Sets
  const char      *name = "Face Sets"; // PETSc internal requirement
  PetscInt        num_face_sets;       // Number of FaceSets in face_set_is
  const PetscInt  *face_set_ids;       // id of each FaceSet

  PetscFunctionBeginUser;

  // Setup DM
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);
  ierr = PetscFECreateLagrange(comm, dim, num_comp_u, PETSC_FALSE, order, order,
                               &fe); CHKERRQ(ierr);
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
