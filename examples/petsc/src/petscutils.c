#include "../include/petscutils.h"

// -----------------------------------------------------------------------------
// Convert PETSc MemType to libCEED MemType
// -----------------------------------------------------------------------------
CeedMemType MemTypeP2C(PetscMemType mem_type) {
  return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}

// -----------------------------------------------------------------------------
// Apply 3D Kershaw mesh transformation
// -----------------------------------------------------------------------------
// Transition from a value of "a" for x=0, to a value of "b" for x=1.  Optionally
// smooth -- see the commented versions at the end.
static double step(const double a, const double b, double x) {
  if (x <= 0) return a;
  if (x >= 1) return b;
  return a + (b-a) * (x);
}

// 1D transformation at the right boundary
static double right(const double eps, const double x) {
  return (x <= 0.5) ? (2-eps) * x : 1 + eps*(x-1);
}

// 1D transformation at the left boundary
static double left(const double eps, const double x) {
  return 1-right(eps,1-x);
}

// Apply 3D Kershaw mesh transformation
// The eps parameters are in (0, 1]
// Uniform mesh is recovered for eps=1
PetscErrorCode Kershaw(DM dm_orig, PetscScalar eps) {
  PetscErrorCode ierr;
  Vec coord;
  PetscInt ncoord;
  PetscScalar *c;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinatesLocal(dm_orig, &coord); CHKERRQ(ierr);
  ierr = VecGetLocalSize(coord, &ncoord); CHKERRQ(ierr);
  ierr = VecGetArray(coord, &c); CHKERRQ(ierr);

  for (PetscInt i = 0; i < ncoord; i += 3) {
    PetscScalar x = c[i], y = c[i+1], z = c[i+2];
    PetscInt layer = x*6;
    PetscScalar lambda = (x-layer/6.0)*6;
    c[i] = x;

    switch (layer) {
    case 0:
      c[i+1] = left(eps, y);
      c[i+2] = left(eps, z);
      break;
    case 1:
    case 4:
      c[i+1] = step(left(eps, y), right(eps, y), lambda);
      c[i+2] = step(left(eps, z), right(eps, z), lambda);
      break;
    case 2:
      c[i+1] = step(right(eps, y), left(eps, y), lambda/2);
      c[i+2] = step(right(eps, z), left(eps, z), lambda/2);
      break;
    case 3:
      c[i+1] = step(right(eps, y), left(eps, y), (1+lambda)/2);
      c[i+2] = step(right(eps, z), left(eps, z), (1+lambda)/2);
      break;
    default:
      c[i+1] = right(eps, y);
      c[i+2] = right(eps, z);
    }
  }
  ierr = VecRestoreArray(coord, &c); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Create BC label
// -----------------------------------------------------------------------------
static PetscErrorCode CreateBCLabel(DM dm, const char name[]) {
  int ierr;
  DMLabel label;

  PetscFunctionBeginUser;

  ierr = DMCreateLabel(dm, name); CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label); CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, 1, label); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function sets up a DM for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupDMByDegree(DM dm, PetscInt degree, PetscInt num_comp_u,
                               PetscInt dim, bool enforce_bc, BCFunction bc_func) {
  PetscInt ierr, marker_ids[1] = {1};
  PetscFE fe;
  MPI_Comm comm;
  PetscBool      is_simplex = PETSC_TRUE;

  PetscFunctionBeginUser;

  // Check if simplex or tensor-product mesh
  ierr = DMPlexIsSimplex(dm, &is_simplex); CHKERRQ(ierr);
  // Setup FE
  ierr = PetscObjectGetComm((PetscObject)dm, &comm); CHKERRQ(ierr);
  ierr = PetscFECreateLagrange(comm, dim, num_comp_u, is_simplex, degree, degree,
                               &fe); CHKERRQ(ierr);
  ierr = DMAddField(dm, NULL, (PetscObject)fe); CHKERRQ(ierr);
  ierr = DMCreateDS(dm); CHKERRQ(ierr);
  {
    DM             dm_coord;
    PetscDS        ds_coord;
    PetscFE        fe_coord_current, fe_coord_new;
    PetscDualSpace fe_coord_dual_space;
    PetscInt       fe_coord_order, num_comp_coord;

    ierr = DMGetCoordinateDM(dm, &dm_coord); CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dm, &num_comp_coord); CHKERRQ(ierr);
    ierr = DMGetRegionDS(dm_coord, NULL, NULL, &ds_coord); CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(ds_coord, 0, (PetscObject *)&fe_coord_current); CHKERRQ(ierr);
    ierr = PetscFEGetDualSpace(fe_coord_current, &fe_coord_dual_space); CHKERRQ(ierr);
    ierr = PetscDualSpaceGetOrder(fe_coord_dual_space, &fe_coord_order); CHKERRQ(ierr);

    // Create FE for coordinates
    ierr = PetscFECreateLagrange(comm, dim, num_comp_coord, is_simplex, fe_coord_order, degree, &fe_coord_new);
    CHKERRQ(ierr);
    ierr = DMProjectCoordinates(dm, fe_coord_new); CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe_coord_new); CHKERRQ(ierr);
  }
  /*
  {
    // create FE field for coordinates
    PetscFE fe_coords;
    PetscInt num_comp_coord;
    ierr = DMGetCoordinateDim(dm, &num_comp_coord); CHKERRQ(ierr);
    ierr = PetscFECreateLagrange(comm, dim, num_comp_coord, is_simplex, 1, degree,
                                 &fe_coords); CHKERRQ(ierr);
    ierr = DMProjectCoordinates(dm, fe_coords); CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe_coords); CHKERRQ(ierr);
  }
  */
  // Setup DM
  //ierr = DMCreateDS(dm); CHKERRQ(ierr);
  if (enforce_bc) {
    PetscBool has_label;
    DMHasLabel(dm, "marker", &has_label);
    if (!has_label) {CreateBCLabel(dm, "marker");}
    DMLabel label;
    ierr = DMGetLabel(dm, "marker", &label); CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1,
                         marker_ids, 0, 0, NULL, (void(*)(void))bc_func,
                         NULL, NULL, NULL); CHKERRQ(ierr);
  }

  if (!is_simplex) {
    DM dm_coord;
    ierr = DMGetCoordinateDM(dm, &dm_coord); CHKERRQ(ierr);
    ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL); CHKERRQ(ierr);
    ierr = DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL); CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Utility function - essential BC dofs are encoded in closure indices as -(i+1)
// -----------------------------------------------------------------------------
PetscInt Involute(PetscInt i) {
  return i >= 0 ? i : -(i + 1);
};

// -----------------------------------------------------------------------------
// Get CEED restriction data from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height,
    DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr) {
  PetscInt num_elem, elem_size, num_dof, num_comp, *elem_restr_offsets;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = DMPlexGetLocalOffsets(dm, domain_label, value, height, 0, &num_elem,
                               &elem_size, &num_comp, &num_dof, &elem_restr_offsets);
  CHKERRQ(ierr);

  CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp,
                            1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES,
                            elem_restr_offsets, elem_restr);
  ierr = PetscFree(elem_restr_offsets); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Utility function - convert from DMPolytopeType to CeedElemTopology
// -----------------------------------------------------------------------------
static inline CeedElemTopology ElemTopologyP2C(DMPolytopeType cell_type) {
  switch (cell_type) {
  case DM_POLYTOPE_TRIANGLE:      return CEED_TOPOLOGY_TRIANGLE;
  case DM_POLYTOPE_QUADRILATERAL: return CEED_TOPOLOGY_QUAD;
  case DM_POLYTOPE_TETRAHEDRON:   return CEED_TOPOLOGY_TET;
  case DM_POLYTOPE_HEXAHEDRON:    return CEED_TOPOLOGY_HEX;
  default:                        return 0;
  }
}

// -----------------------------------------------------------------------------
// Get CEED Basis from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label,
                                   CeedInt label_value, CeedInt height,
                                   CeedInt dm_field, CeedBasis *basis) {
  PetscErrorCode   ierr;
  PetscDS          ds;
  PetscFE          fe;
  PetscQuadrature  quadrature;
  PetscBool        is_simplex = PETSC_TRUE;
  PetscInt         dim, ds_field = -1, num_comp, P, Q;

  PetscFunctionBeginUser;

  // Get basis information
  {
    IS             field_is;
    const PetscInt *fields;
    PetscInt       num_fields;

    ierr = DMGetRegionDS(dm, domain_label, &field_is, &ds); CHKERRQ(ierr);
    // Translate dm_field to ds_field
    ierr = ISGetIndices(field_is, &fields); CHKERRQ(ierr);
    ierr = ISGetSize(field_is, &num_fields); CHKERRQ(ierr);
    for (PetscInt i = 0; i < num_fields; i++) {
      if (dm_field == fields[i]) {
        ds_field = i;
        break;
      }
    }
    ierr = ISRestoreIndices(field_is, &fields); CHKERRQ(ierr);
  }
  if (ds_field == -1) {
    // LCOV_EXCL_START
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP,
             "Could not find dm_field %D in DS", dm_field);
    // LCOV_EXCL_STOP
  }

  // Get element information
  {
    PetscDualSpace dual_space;
    PetscInt       num_dual_basis_vectors;

    ierr = PetscDSGetDiscretization(ds, ds_field, (PetscObject *)&fe);
    CHKERRQ(ierr);
    ierr = PetscFEGetHeightSubspace(fe, height, &fe); CHKERRQ(ierr);
    ierr = PetscFEGetSpatialDimension(fe, &dim); CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe, &num_comp); CHKERRQ(ierr);
    ierr = PetscFEGetDualSpace(fe, &dual_space); CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(dual_space, &num_dual_basis_vectors);
    CHKERRQ(ierr);
    P = num_dual_basis_vectors / num_comp;
    ierr = PetscFEGetQuadrature(fe, &quadrature); CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(quadrature, NULL, NULL, &Q, NULL, NULL);
    CHKERRQ(ierr);
  }

  // Check if simplex or tensor-product mesh
  ierr = DMPlexIsSimplex(dm, &is_simplex); CHKERRQ(ierr);
  // Build libCEED basis
  if (is_simplex) {
    PetscInt          num_derivatives = 1, first_point;
    PetscInt          ids[1] = {label_value};
    PetscTabulation   basis_tabulation;
    const PetscScalar *q_points, *q_weights;
    DMLabel           depth_label;
    DMPolytopeType    cell_type;
    CeedElemTopology  elem_topo;
    PetscScalar       *interp, *grad;

    // Use depth label if no domain label present
    if (!domain_label) {
      PetscInt depth;

      ierr = DMPlexGetDepth(dm, &depth); CHKERRQ(ierr);
      ierr = DMPlexGetDepthLabel(dm, &depth_label); CHKERRQ(ierr);
      ids[0] = depth - height;
    }
    // Get cell interp, grad, and quadrature data
    ierr = PetscFEGetCellTabulation(fe, num_derivatives, &basis_tabulation);
    CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(quadrature, NULL, NULL, NULL, &q_points,
                                  &q_weights); CHKERRQ(ierr);
    ierr = DMGetFirstLabeledPoint(dm, dm, domain_label ? domain_label : depth_label,
                                  1, ids, height, &first_point, NULL);
    CHKERRQ(ierr);
    ierr = DMPlexGetCellType(dm, first_point, &cell_type); CHKERRQ(ierr);
    elem_topo = ElemTopologyP2C(cell_type);
    if (!elem_topo) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP,
                              "DMPlex topology not supported");
    // Convert to libCEED orientation
    ierr = PetscCalloc(P * Q * sizeof(PetscScalar), &interp); CHKERRQ(ierr);
    ierr = PetscCalloc(P * Q * dim * sizeof(PetscScalar), &grad); CHKERRQ(ierr);
    const CeedInt c = 0;
    for (CeedInt q = 0; q < Q; q++) {
      for (CeedInt p = 0; p < P; p++) {
        interp[q*P + p] = basis_tabulation->T[0][(q*P + p)*num_comp*num_comp + c];
        for (CeedInt d = 0; d < dim; d++) {
          grad[(d*Q + q)*P + p] = basis_tabulation->T[1][((q*P + p)*num_comp*num_comp + c)
                                  *dim + d];
        }
      }
    }
    // Finaly, create libCEED basis
    ierr = CeedBasisCreateH1(ceed, elem_topo, num_comp, P, Q, interp, grad,
                             q_points, q_weights, basis);
    CHKERRQ(ierr);
    ierr = PetscFree(interp); CHKERRQ(ierr);
    ierr = PetscFree(grad); CHKERRQ(ierr);
  } else {
    CeedInt P_1d = (CeedInt) round(pow(P, 1.0 / dim));
    CeedInt Q_1d = (CeedInt) round(pow(Q, 1.0 / dim));

    ierr = CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, P_1d, Q_1d,
                                           CEED_GAUSS, basis);
    CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
