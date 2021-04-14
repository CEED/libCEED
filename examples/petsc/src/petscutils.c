#include "../include/petscutils.h"

// -----------------------------------------------------------------------------
// Convert PETSc MemType to libCEED MemType
// -----------------------------------------------------------------------------
CeedMemType MemTypeP2C(PetscMemType mem_type) {
  return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}

// -----------------------------------------------------------------------------
// Utility function taken from petsc/src/dm/impls/plex/examples/tutorials/ex7.c
// -----------------------------------------------------------------------------
PetscErrorCode ProjectToUnitSphere(DM dm) {
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       Nv, v, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &Nv); CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &dim); CHKERRQ(ierr);
  Nv  /= dim;
  ierr = VecGetArray(coordinates, &coords); CHKERRQ(ierr);
  for (v = 0; v < Nv; ++v) {
    PetscReal r = 0.0;

    for (d = 0; d < dim; ++d) r += PetscSqr(PetscRealPart(coords[v*dim+d]));
    r = PetscSqrtReal(r);
    for (d = 0; d < dim; ++d) coords[v*dim+d] /= r;
  }
  ierr = VecRestoreArray(coordinates, &coords); CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

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
// PETSc FE Boilerplate
// -----------------------------------------------------------------------------
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
};

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
  ierr = DMPlexLabelComplete(dm, label); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// This function sets up a DM for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupDMByDegree(DM dm, PetscInt degree, PetscInt num_comp_u,
                               PetscInt dim, bool enforce_bc, BCFunction bc_func) {
  PetscInt ierr, marker_ids[1] = {1};
  PetscFE fe;

  PetscFunctionBeginUser;

  // Setup FE
  ierr = PetscFECreateByDegree(dm, dim, num_comp_u, PETSC_FALSE, NULL, degree,
                               &fe);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMAddField(dm, NULL, (PetscObject)fe); CHKERRQ(ierr);

  // Setup DM
  ierr = DMCreateDS(dm); CHKERRQ(ierr);
  if (enforce_bc) {
    PetscBool has_label;
    DMHasLabel(dm, "marker", &has_label);
    if (!has_label) {CreateBCLabel(dm, "marker");}
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL,
                         (void(*)(void))bc_func, NULL, 1, marker_ids, NULL);
    CHKERRQ(ierr);
  }
  ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
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
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt topo_dim, CeedInt height, DMLabel domain_label, CeedInt value,
    CeedElemRestriction *elem_restr) {
  PetscSection section;
  PetscInt p, num_elem, num_dof, *elem_restr_offsets, e_offset, num_fields, dim,
           depth;
  DMLabel depth_label;
  IS depth_is, iter_is;
  Vec U_loc;
  const PetscInt *iter_indices;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &num_fields); CHKERRQ(ierr);
  PetscInt num_comp[num_fields], field_off[num_fields+1];
  field_off[0] = 0;
  for (PetscInt f = 0; f < num_fields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &num_comp[f]); CHKERRQ(ierr);
    field_off[f+1] = field_off[f] + num_comp[f];
  }

  ierr = DMPlexGetDepth(dm, &depth); CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depth_label); CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depth_label, depth - height, &depth_is);
  CHKERRQ(ierr);
  if (domain_label) {
    IS domain_is;
    ierr = DMLabelGetStratumIS(domain_label, value, &domain_is); CHKERRQ(ierr);
    if (domain_is) { // domain_is is non-empty
      ierr = ISIntersect(depth_is, domain_is, &iter_is); CHKERRQ(ierr);
      ierr = ISDestroy(&domain_is); CHKERRQ(ierr);
    } else { // domain_is is NULL (empty)
      iter_is = NULL;
    }
    ierr = ISDestroy(&depth_is); CHKERRQ(ierr);
  } else {
    iter_is = depth_is;
  }
  if (iter_is) {
    ierr = ISGetLocalSize(iter_is, &num_elem); CHKERRQ(ierr);
    ierr = ISGetIndices(iter_is, &iter_indices); CHKERRQ(ierr);
  } else {
    num_elem = 0;
    iter_indices = NULL;
  }
  ierr = PetscMalloc1(num_elem*PetscPowInt(P, topo_dim), &elem_restr_offsets);
  CHKERRQ(ierr);
  for (p = 0, e_offset = 0; p < num_elem; p++) {
    PetscInt c = iter_indices[p];
    PetscInt num_indices, *indices, num_nodes;
    ierr = DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE,
                                   &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
    bool flip = false;
    if (height > 0) {
      PetscInt num_cells, num_faces, start = -1;
      const PetscInt *orients, *faces, *cells;
      ierr = DMPlexGetSupport(dm, c, &cells); CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, c, &num_cells); CHKERRQ(ierr);
      if (num_cells != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                                     "Expected one cell in support of exterior face, but got %D cells",
                                     num_cells);
      ierr = DMPlexGetCone(dm, cells[0], &faces); CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, cells[0], &num_faces); CHKERRQ(ierr);
      for (PetscInt i=0; i<num_faces; i++) {if (faces[i] == c) start = i;}
      if (start < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT,
                                "Could not find face %D in cone of its support",
                                c);
      ierr = DMPlexGetConeOrientation(dm, cells[0], &orients); CHKERRQ(ierr);
      if (orients[start] < 0) flip = true;
    }
    if (num_indices % field_off[num_fields]) SETERRQ1(PETSC_COMM_SELF,
          PETSC_ERR_ARG_INCOMP, "Number of closure indices not compatible with Cell %D",
          c);
    num_nodes = num_indices / field_off[num_fields];
    for (PetscInt i = 0; i < num_nodes; i++) {
      PetscInt ii = i;
      if (flip) {
        if (P == num_nodes) ii = num_nodes - 1 - i;
        else if (P*P == num_nodes) {
          PetscInt row = i / P, col = i % P;
          ii = row + col * P;
        } else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP,
                          "No support for flipping point with %D nodes != P (%D) or P^2",
                          num_nodes, P);
      }
      // Check that indices are blocked by node and thus can be coalesced as a single field with
      // field_off[num_fields] = sum(num_comp) components.
      for (PetscInt f = 0; f < num_fields; f++) {
        for (PetscInt j = 0; j < num_comp[f]; j++) {
          if (Involute(indices[field_off[f]*num_nodes + ii*num_comp[f] + j])
              != Involute(indices[ii*num_comp[0]]) + field_off[f] + j)
            SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                     "Cell %D closure indices not interlaced for node %D field %D component %D",
                     c, ii, f, j);
        }
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = Involute(indices[ii*num_comp[0]]);
      elem_restr_offsets[e_offset++] = loc;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE,
                                       &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }
  if (e_offset != num_elem*PetscPowInt(P, topo_dim))
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB,
             "ElemRestriction of size (%D,%D) initialized %D nodes", num_elem,
             PetscPowInt(P, topo_dim),e_offset);
  if (iter_is) {
    ierr = ISRestoreIndices(iter_is, &iter_indices); CHKERRQ(ierr);
  }
  ierr = ISDestroy(&iter_is); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &U_loc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(U_loc, &num_dof); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &U_loc); CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, num_elem, PetscPowInt(P, topo_dim),
                            field_off[num_fields], 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES,
                            elem_restr_offsets, elem_restr);
  ierr = PetscFree(elem_restr_offsets); CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
