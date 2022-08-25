#include "../include/petscutils.h"

// -----------------------------------------------------------------------------
// Convert PETSc MemType to libCEED MemType
// -----------------------------------------------------------------------------
CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

// -----------------------------------------------------------------------------
// Apply 3D Kershaw mesh transformation
// -----------------------------------------------------------------------------
// Transition from a value of "a" for x=0, to a value of "b" for x=1.  Optionally
// smooth -- see the commented versions at the end.
static double step(const double a, const double b, double x) {
  if (x <= 0) return a;
  if (x >= 1) return b;
  return a + (b - a) * (x);
}

// 1D transformation at the right boundary
static double right(const double eps, const double x) { return (x <= 0.5) ? (2 - eps) * x : 1 + eps * (x - 1); }

// 1D transformation at the left boundary
static double left(const double eps, const double x) { return 1 - right(eps, 1 - x); }

// Apply 3D Kershaw mesh transformation
// The eps parameters are in (0, 1]
// Uniform mesh is recovered for eps=1
PetscErrorCode Kershaw(DM dm_orig, PetscScalar eps) {
  Vec          coord;
  PetscInt     ncoord;
  PetscScalar *c;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinatesLocal(dm_orig, &coord));
  PetscCall(VecGetLocalSize(coord, &ncoord));
  PetscCall(VecGetArray(coord, &c));

  for (PetscInt i = 0; i < ncoord; i += 3) {
    PetscScalar x = c[i], y = c[i + 1], z = c[i + 2];
    PetscInt    layer  = x * 6;
    PetscScalar lambda = (x - layer / 6.0) * 6;
    c[i]               = x;

    switch (layer) {
      case 0:
        c[i + 1] = left(eps, y);
        c[i + 2] = left(eps, z);
        break;
      case 1:
      case 4:
        c[i + 1] = step(left(eps, y), right(eps, y), lambda);
        c[i + 2] = step(left(eps, z), right(eps, z), lambda);
        break;
      case 2:
        c[i + 1] = step(right(eps, y), left(eps, y), lambda / 2);
        c[i + 2] = step(right(eps, z), left(eps, z), lambda / 2);
        break;
      case 3:
        c[i + 1] = step(right(eps, y), left(eps, y), (1 + lambda) / 2);
        c[i + 2] = step(right(eps, z), left(eps, z), (1 + lambda) / 2);
        break;
      default:
        c[i + 1] = right(eps, y);
        c[i + 2] = right(eps, z);
    }
  }
  PetscCall(VecRestoreArray(coord, &c));
  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Create BC label
// -----------------------------------------------------------------------------
static PetscErrorCode CreateBCLabel(DM dm, const char name[]) {
  DMLabel label;

  PetscFunctionBeginUser;

  PetscCall(DMCreateLabel(dm, name));
  PetscCall(DMGetLabel(dm, name, &label));
  PetscCall(DMPlexMarkBoundaryFaces(dm, 1, label));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function sets up a DM for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupDMByDegree(DM dm, PetscInt degree, PetscInt num_comp_u, PetscInt dim, bool enforce_bc, BCFunction bc_func) {
  PetscInt marker_ids[1] = {1};
  PetscFE  fe;
  MPI_Comm comm;

  PetscFunctionBeginUser;

  // Setup FE
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(PetscFECreateLagrange(comm, dim, num_comp_u, PETSC_FALSE, degree, degree, &fe));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
  {
    /* create FE field for coordinates */
    PetscFE  fe_coords;
    PetscInt num_comp_coord;
    PetscCall(DMGetCoordinateDim(dm, &num_comp_coord));
    PetscCall(PetscFECreateLagrange(comm, dim, num_comp_coord, PETSC_FALSE, 1, 1, &fe_coords));
    PetscCall(DMProjectCoordinates(dm, fe_coords));
    PetscCall(PetscFEDestroy(&fe_coords));
  }

  // Setup DM
  PetscCall(DMCreateDS(dm));
  if (enforce_bc) {
    PetscBool has_label;
    DMHasLabel(dm, "marker", &has_label);
    if (!has_label) {
      CreateBCLabel(dm, "marker");
    }
    DMLabel label;
    PetscCall(DMGetLabel(dm, "marker", &label));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, marker_ids, 0, 0, NULL, (void (*)(void))bc_func, NULL, NULL, NULL));
  }
  PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
  PetscCall(PetscFEDestroy(&fe));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Utility function - essential BC dofs are encoded in closure indices as -(i+1)
// -----------------------------------------------------------------------------
PetscInt Involute(PetscInt i) { return i >= 0 ? i : -(i + 1); };

// -----------------------------------------------------------------------------
// Get CEED restriction data from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr) {
  PetscInt num_elem, elem_size, num_dof, num_comp, *elem_restr_offsets;

  PetscFunctionBeginUser;

  PetscCall(DMPlexGetLocalOffsets(dm, domain_label, value, height, 0, &num_elem, &elem_size, &num_comp, &num_dof, &elem_restr_offsets));

  CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp, 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES, elem_restr_offsets, elem_restr);
  PetscCall(PetscFree(elem_restr_offsets));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
