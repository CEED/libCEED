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
  PetscCall(DMPlexMarkBoundaryFaces(dm, PETSC_DETERMINE, label));
  PetscCall(DMPlexLabelComplete(dm, label));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function sets up a DM for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupDMByDegree(DM dm, PetscInt p_degree, PetscInt q_extra, PetscInt num_comp_u, PetscInt dim, bool enforce_bc) {
  PetscInt  marker_ids[1] = {1};
  PetscInt  q_degree      = p_degree + q_extra;
  PetscFE   fe;
  MPI_Comm  comm;
  PetscBool is_simplex = PETSC_TRUE;

  PetscFunctionBeginUser;

  // Check if simplex or tensor-product mesh
  PetscCall(DMPlexIsSimplex(dm, &is_simplex));
  // Setup FE
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(PetscFECreateLagrange(comm, dim, num_comp_u, is_simplex, p_degree, q_degree, &fe));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));

  {
    // create FE field for coordinates
    PetscFE  fe_coords;
    PetscInt num_comp_coord;
    PetscCall(DMGetCoordinateDim(dm, &num_comp_coord));
    PetscCall(PetscFECreateLagrange(comm, dim, num_comp_coord, is_simplex, 1, q_degree, &fe_coords));
    PetscCall(DMProjectCoordinates(dm, fe_coords));
    PetscCall(PetscFEDestroy(&fe_coords));
  }

  // Setup Dirichlet BC
  // Note bp1, bp2 are projection and we don't need to apply BC
  // For bp3,bp4, the target function is zero on the boundaries
  // So we pass bcFunc = NULL in DMAddBoundary function
  if (enforce_bc) {
    PetscBool has_label;
    DMHasLabel(dm, "marker", &has_label);
    if (!has_label) {
      CreateBCLabel(dm, "marker");
    }
    DMLabel label;
    PetscCall(DMGetLabel(dm, "marker", &label));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, marker_ids, 0, 0, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMSetOptionsPrefix(dm, "final_"));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  }

  if (!is_simplex) {
    DM dm_coord;
    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
    PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
  }
  PetscCall(PetscFEDestroy(&fe));

  PetscFunctionReturn(0);
};

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
// Utility function - convert from DMPolytopeType to CeedElemTopology
// -----------------------------------------------------------------------------
CeedElemTopology ElemTopologyP2C(DMPolytopeType cell_type) {
  switch (cell_type) {
    case DM_POLYTOPE_TRIANGLE:
      return CEED_TOPOLOGY_TRIANGLE;
    case DM_POLYTOPE_QUADRILATERAL:
      return CEED_TOPOLOGY_QUAD;
    case DM_POLYTOPE_TETRAHEDRON:
      return CEED_TOPOLOGY_TET;
    case DM_POLYTOPE_HEXAHEDRON:
      return CEED_TOPOLOGY_HEX;
    default:
      return 0;
  }
}

// -----------------------------------------------------------------------------
// Convert DM field to DS field
// -----------------------------------------------------------------------------
PetscErrorCode DMFieldToDSField(DM dm, DMLabel domain_label, PetscInt dm_field, PetscInt *ds_field) {
  PetscDS         ds;
  IS              field_is;
  const PetscInt *fields;
  PetscInt        num_fields;

  PetscFunctionBeginUser;

  // Translate dm_field to ds_field
  PetscCall(DMGetRegionDS(dm, domain_label, &field_is, &ds));
  PetscCall(ISGetIndices(field_is, &fields));
  PetscCall(ISGetSize(field_is, &num_fields));
  for (PetscInt i = 0; i < num_fields; i++) {
    if (dm_field == fields[i]) {
      *ds_field = i;
      break;
    }
  }
  PetscCall(ISRestoreIndices(field_is, &fields));

  if (*ds_field == -1) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Could not find dm_field %" PetscInt_FMT " in DS", dm_field);

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Create libCEED Basis from PetscTabulation
// -----------------------------------------------------------------------------
PetscErrorCode BasisCreateFromTabulation(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt face, PetscFE fe,
                                         PetscTabulation basis_tabulation, PetscQuadrature quadrature, CeedBasis *basis) {
  PetscInt           first_point;
  PetscInt           ids[1] = {label_value};
  DMLabel            depth_label;
  DMPolytopeType     cell_type;
  CeedElemTopology   elem_topo;
  PetscScalar       *q_points, *interp, *grad;
  const PetscScalar *q_weights;
  PetscDualSpace     dual_space;
  PetscInt           num_dual_basis_vectors;
  PetscInt           dim, num_comp, P, Q;

  PetscFunctionBeginUser;

  // General basis information
  PetscCall(PetscFEGetSpatialDimension(fe, &dim));
  PetscCall(PetscFEGetNumComponents(fe, &num_comp));
  PetscCall(PetscFEGetDualSpace(fe, &dual_space));
  PetscCall(PetscDualSpaceGetDimension(dual_space, &num_dual_basis_vectors));
  P = num_dual_basis_vectors / num_comp;

  // Use depth label if no domain label present
  if (!domain_label) {
    PetscInt depth;

    PetscCall(DMPlexGetDepth(dm, &depth));
    PetscCall(DMPlexGetDepthLabel(dm, &depth_label));
    ids[0] = depth - height;
  }

  // Get cell interp, grad, and quadrature data
  PetscCall(DMGetFirstLabeledPoint(dm, dm, domain_label ? domain_label : depth_label, 1, ids, height, &first_point, NULL));
  PetscCall(DMPlexGetCellType(dm, first_point, &cell_type));
  elem_topo = ElemTopologyP2C(cell_type);
  if (!elem_topo) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "DMPlex topology not supported");
  {
    size_t             q_points_size;
    const PetscScalar *q_points_petsc;
    PetscInt           q_dim;

    PetscCall(PetscQuadratureGetData(quadrature, &q_dim, NULL, &Q, &q_points_petsc, &q_weights));
    q_points_size = Q * dim * sizeof(CeedScalar);
    PetscCall(PetscCalloc(q_points_size, &q_points));
    for (PetscInt q = 0; q < Q; q++) {
      for (PetscInt d = 0; d < q_dim; d++) q_points[q * dim + d] = q_points_petsc[q * q_dim + d];
    }
  }

  // Convert to libCEED orientation
  {
    PetscBool       is_simplex  = PETSC_FALSE;
    IS              permutation = NULL;
    const PetscInt *permutation_indices;

    PetscCall(DMPlexIsSimplex(dm, &is_simplex));
    if (!is_simplex) {
      PetscSection section;

      // -- Get permutation
      PetscCall(DMGetLocalSection(dm, &section));
      PetscCall(PetscSectionGetClosurePermutation(section, (PetscObject)dm, dim, num_comp * P, &permutation));
      PetscCall(ISGetIndices(permutation, &permutation_indices));
    }

    // -- Copy interp, grad matrices
    PetscCall(PetscCalloc(P * Q * sizeof(CeedScalar), &interp));
    PetscCall(PetscCalloc(P * Q * dim * sizeof(CeedScalar), &grad));
    const CeedInt c = 0;
    for (CeedInt q = 0; q < Q; q++) {
      for (CeedInt p_ceed = 0; p_ceed < P; p_ceed++) {
        CeedInt p_petsc = is_simplex ? (p_ceed * num_comp) : permutation_indices[p_ceed * num_comp];

        interp[q * P + p_ceed] = basis_tabulation->T[0][((face * Q + q) * P * num_comp + p_petsc) * num_comp + c];
        for (CeedInt d = 0; d < dim; d++) {
          grad[(d * Q + q) * P + p_ceed] = basis_tabulation->T[1][(((face * Q + q) * P * num_comp + p_petsc) * num_comp + c) * dim + d];
        }
      }
    }

    // -- Cleanup
    if (permutation) PetscCall(ISRestoreIndices(permutation, &permutation_indices));
    PetscCall(ISDestroy(&permutation));
  }

  // Finally, create libCEED basis
  CeedBasisCreateH1(ceed, elem_topo, num_comp, P, Q, interp, grad, q_points, q_weights, basis);
  PetscCall(PetscFree(q_points));
  PetscCall(PetscFree(interp));
  PetscCall(PetscFree(grad));

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Get CEED Basis from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label, CeedInt label_value, CeedInt height, CeedInt dm_field, BPData bp_data,
                                   CeedBasis *basis) {
  PetscDS         ds;
  PetscFE         fe;
  PetscQuadrature quadrature;
  PetscBool       is_simplex = PETSC_TRUE;
  PetscInt        ds_field   = -1;

  PetscFunctionBeginUser;

  // Get element information
  PetscCall(DMGetRegionDS(dm, domain_label, NULL, &ds));
  PetscCall(DMFieldToDSField(dm, domain_label, dm_field, &ds_field));
  PetscCall(PetscDSGetDiscretization(ds, ds_field, (PetscObject *)&fe));
  PetscCall(PetscFEGetHeightSubspace(fe, height, &fe));
  PetscCall(PetscFEGetQuadrature(fe, &quadrature));

  // Check if simplex or tensor-product mesh
  PetscCall(DMPlexIsSimplex(dm, &is_simplex));

  // Build libCEED basis
  if (is_simplex) {
    PetscTabulation basis_tabulation;
    PetscInt        num_derivatives = 1, face = 0;

    PetscCall(PetscFEGetCellTabulation(fe, num_derivatives, &basis_tabulation));
    PetscCall(BasisCreateFromTabulation(ceed, dm, domain_label, label_value, height, face, fe, basis_tabulation, quadrature, basis));
  } else {
    PetscDualSpace dual_space;
    PetscInt       num_dual_basis_vectors;
    PetscInt       dim, num_comp, P, Q;

    PetscCall(PetscFEGetSpatialDimension(fe, &dim));
    PetscCall(PetscFEGetNumComponents(fe, &num_comp));
    PetscCall(PetscFEGetDualSpace(fe, &dual_space));
    PetscCall(PetscDualSpaceGetDimension(dual_space, &num_dual_basis_vectors));
    P = num_dual_basis_vectors / num_comp;
    PetscCall(PetscQuadratureGetData(quadrature, NULL, NULL, &Q, NULL, NULL));

    CeedInt P_1d = (CeedInt)round(pow(P, 1.0 / dim));
    CeedInt Q_1d = (CeedInt)round(pow(Q, 1.0 / dim));

    CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, P_1d, Q_1d, bp_data.q_mode, basis);
  }

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

// Utility function, compute three factors of an integer
static void Split3(PetscInt size, PetscInt m[3], bool reverse) {
  for (PetscInt d = 0, size_left = size; d < 3; d++) {
    PetscInt try = (PetscInt)PetscCeilReal(PetscPowReal(size_left, 1. / (3 - d)));
    while (try * (size_left / try) != size_left) try++;
    m[reverse ? 2 - d : d] = try;
    size_left /= try;
  }
}

static int Max3(const PetscInt a[3]) { return PetscMax(a[0], PetscMax(a[1], a[2])); }

static int Min3(const PetscInt a[3]) { return PetscMin(a[0], PetscMin(a[1], a[2])); }

// -----------------------------------------------------------------------------
// Create distribute dm
// -----------------------------------------------------------------------------
PetscErrorCode CreateDistributedDM(RunParams rp, DM *dm) {
  PetscFunctionBeginUser;
  // Setup DM
  if (rp->read_mesh) {
    PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, rp->filename, NULL, PETSC_TRUE, dm));
  } else {
    if (rp->user_l_nodes) {
      // Find a nicely composite number of elements no less than global nodes
      PetscMPIInt size;
      PetscCall(MPI_Comm_size(rp->comm, &size));
      for (PetscInt g_elem = PetscMax(1, size * rp->local_nodes / PetscPowInt(rp->degree, rp->dim));; g_elem++) {
        Split3(g_elem, rp->mesh_elem, true);
        if (Max3(rp->mesh_elem) / Min3(rp->mesh_elem) <= 2) break;
      }
    }

    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, rp->dim, rp->simplex, rp->mesh_elem, NULL, NULL, NULL, PETSC_TRUE, dm));
  }

  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
