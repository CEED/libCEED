#include "../include/setup-libceed.h"
#include "../include/petsc-macros.h"
#include "../basis/quad.h"

// -----------------------------------------------------------------------------
// Convert PETSc MemType to libCEED MemType
// -----------------------------------------------------------------------------
CeedMemType MemTypeP2C(PetscMemType mem_type) {
  return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}
// -----------------------------------------------------------------------------
// Destroy libCEED objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedDataDestroy(CeedData ceed_data) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Vectors
  CeedVectorDestroy(&ceed_data->x_ceed);
  CeedVectorDestroy(&ceed_data->y_ceed);
  // Restrictions
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_x);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_u);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_geo_data_i);
  // Bases
  CeedBasisDestroy(&ceed_data->basis_x);
  CeedBasisDestroy(&ceed_data->basis_u);
  // QFunctions
  CeedQFunctionDestroy(&ceed_data->qf_residual);
  // Operators
  CeedOperatorDestroy(&ceed_data->op_residual);
  ierr = PetscFree(ceed_data); CHKERRQ(ierr);

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
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm,
    CeedInt height, DMLabel domain_label, CeedInt value, CeedInt P,
    CeedElemRestriction *elem_restr) {
  PetscSection section;
  PetscInt p, num_elem, num_dof, *restr_indices, elem_offset, num_fields,
           dim, depth;
  Vec U_loc;
  DMLabel depth_label;
  IS depth_is, iter_is;
  const PetscInt *iter_indices;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &num_fields); CHKERRQ(ierr);
  PetscInt num_comp[num_fields], field_offsets[num_fields+1];
  field_offsets[0] = 0;
  for (PetscInt f = 0; f < num_fields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &num_comp[f]); CHKERRQ(ierr);
    field_offsets[f+1] = field_offsets[f] + num_comp[f];
  }

  ierr = DMPlexGetDepth(dm, &depth); CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depth_label); CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depth_label, depth - height, &depth_is);
  CHKERRQ(ierr);
  if (domain_label) {
    IS domain_is;
    ierr = DMLabelGetStratumIS(domain_label, value, &domain_is); CHKERRQ(ierr);
    if (domain_is) { // domainIS is non-empty
      ierr = ISIntersect(depth_is, domain_is, &iter_is); CHKERRQ(ierr);
      ierr = ISDestroy(&domain_is); CHKERRQ(ierr);
    } else { // domainIS is NULL (empty)
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
  ierr = PetscMalloc1(num_elem*PetscPowInt(P, dim), &restr_indices);
  CHKERRQ(ierr);
  for (p = 0, elem_offset = 0; p < num_elem; p++) {
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
    if (num_indices % field_offsets[num_fields]) SETERRQ1(PETSC_COMM_SELF,
          PETSC_ERR_ARG_INCOMP, "Number of closure indices not compatible with Cell %D",
          c);
    num_nodes = num_indices / field_offsets[num_fields];
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
      // field_offsets[num_fields] = sum(num_comp) components.
      for (PetscInt f = 0; f < num_fields; f++) {
        for (PetscInt j = 0; j < num_comp[f]; j++) {
          if (Involute(indices[field_offsets[f]*num_nodes + ii*num_comp[f] + j])
              != Involute(indices[ii*num_comp[0]]) + field_offsets[f] + j)
            SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                     "Cell %D closure indices not interlaced for node %D field %D component %D",
                     c, ii, f, j);
        }
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = Involute(indices[ii*num_comp[0]]);
      restr_indices[elem_offset++] = loc;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE,
                                       &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }
  if (elem_offset != num_elem*PetscPowInt(P, dim))
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB,
             "ElemRestriction of size (%D,%D) initialized %D nodes", num_elem,
             PetscPowInt(P, dim),elem_offset);
  if (iter_is) {
    ierr = ISRestoreIndices(iter_is, &iter_indices); CHKERRQ(ierr);
  }
  ierr = ISDestroy(&iter_is); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &U_loc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(U_loc, &num_dof); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &U_loc); CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, num_elem, PetscPowInt(P, dim),
                            field_offsets[num_fields], 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES,
                            restr_indices, elem_restr);
  ierr = PetscFree(restr_indices); CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Get Oriented CEED restriction data from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateRestrictionFromPlexOriented(Ceed ceed, DM dm,
    CeedInt height, DMLabel domain_label, CeedInt value, CeedInt P,
    CeedElemRestriction *elem_restr_oriented) {
  PetscSection section;
  PetscInt p, num_elem, num_dof, *restr_indices, elem_offset, num_fields,
           dim, depth;
  Vec U_loc;
  DMLabel depth_label;
  IS depth_is, iter_is;
  const PetscInt *iter_indices;
  PetscErrorCode ierr;
  const PetscInt *ornt; // this is for orientation of dof

  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &num_fields); CHKERRQ(ierr);
  PetscInt num_comp[num_fields], field_offsets[num_fields+1];
  field_offsets[0] = 0;
  for (PetscInt f = 0; f < num_fields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &num_comp[f]); CHKERRQ(ierr);
    field_offsets[f+1] = field_offsets[f] + num_comp[f];
  }

  ierr = DMPlexGetDepth(dm, &depth); CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depth_label); CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depth_label, depth - height, &depth_is);
  CHKERRQ(ierr);
  if (domain_label) {
    IS domain_is;
    ierr = DMLabelGetStratumIS(domain_label, value, &domain_is); CHKERRQ(ierr);
    if (domain_is) { // domainIS is non-empty
      ierr = ISIntersect(depth_is, domain_is, &iter_is); CHKERRQ(ierr);
      ierr = ISDestroy(&domain_is); CHKERRQ(ierr);
    } else { // domainIS is NULL (empty)
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
  ierr = PetscMalloc1(num_elem*dim*PetscPowInt(P, dim),
                      &restr_indices); CHKERRQ(ierr);
  bool *orient_dof; // to flip the dof
  ierr = PetscMalloc1(num_elem*dim*PetscPowInt(P, dim), &orient_dof);
  CHKERRQ(ierr);
  for (p = 0, elem_offset = 0; p < num_elem; p++) {
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
    if (num_indices % field_offsets[num_fields]) SETERRQ1(PETSC_COMM_SELF,
          PETSC_ERR_ARG_INCOMP, "Number of closure indices not compatible with Cell %D",
          c);
    num_nodes = num_indices / field_offsets[num_fields];
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
      // field_offsets[num_fields] = sum(num_comp) components.
      for (PetscInt f = 0; f < num_fields; f++) {
        for (PetscInt j = 0; j < num_comp[f]; j++) {
          if (Involute(indices[field_offsets[f]*num_nodes + ii*num_comp[f] + j])
              != Involute(indices[ii*num_comp[0]]) + field_offsets[f] + j)
            SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                     "Cell %D closure indices not interlaced for node %D field %D component %D",
                     c, ii, f, j);
        }
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = Involute(indices[ii*num_comp[0]]);
      restr_indices[elem_offset] = loc;
      // Set orientation
      ierr = DMPlexGetConeOrientation(dm, p, &ornt); CHKERRQ(ierr);
      orient_dof[elem_offset] = ornt[i] < 0;
      elem_offset++;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE,
                                       &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }
  if (elem_offset != num_elem*dim*PetscPowInt(P, dim))
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB,
             "ElemRestriction of size (%D,%D) initialized %D nodes", num_elem,
             PetscPowInt(P, dim),elem_offset);
  if (iter_is) {
    ierr = ISRestoreIndices(iter_is, &iter_indices); CHKERRQ(ierr);
  }
  ierr = ISDestroy(&iter_is); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &U_loc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(U_loc, &num_dof); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &U_loc); CHKERRQ(ierr);
  // dof per element in Hdiv is dim*P^dim, for linear element P=2
  CeedElemRestrictionCreateOriented(ceed, num_elem, dim*PetscPowInt(P, dim),
                                    field_offsets[num_fields],
                                    1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES,
                                    restr_indices, orient_dof, elem_restr_oriented);
  ierr = PetscFree(restr_indices); CHKERRQ(ierr);
  ierr = PetscFree(orient_dof); CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Set up libCEED on the fine grid for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupLibceed(DM dm, Ceed ceed, AppCtx app_ctx,
                            ProblemData *problem_data, PetscInt U_g_size, PetscInt U_loc_size,
                            CeedData ceed_data, CeedVector rhs_ceed) {
  int           ierr;
  CeedInt       P = app_ctx->degree + 1;
  CeedInt       Q = P + 1 + app_ctx->q_extra; // Number of quadratures in 1D
  CeedInt       num_qpts = Q*Q; // Number of quadratures per element
  CeedInt       dim, num_comp_x, num_comp_u;
  CeedInt       geo_data_size = problem_data->geo_data_size;
  CeedInt       elem_node = problem_data->elem_node;
  DM            dm_coord;
  Vec           coords;
  PetscInt      c_start, c_end, num_elem;
  const PetscScalar *coordArray;
  CeedVector    x_coord;
  CeedQFunction qf_setup_rhs, qf_residual;
  CeedOperator  op_setup_rhs, op_residual;

  PetscFunctionBeginUser;
  // ---------------------------------------------------------------------------
  // libCEED bases:Hdiv basis_u and Lagrange basis_x
  // ---------------------------------------------------------------------------
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  num_comp_x = dim;
  num_comp_u = 1;   // one vector dof
  CeedInt       elem_dof = dim*elem_node; // dof per element
  CeedScalar    q_ref[dim*num_qpts], q_weights[num_qpts];
  CeedScalar    div[elem_dof*num_qpts], interp[dim*elem_dof*num_qpts];
  QuadBasis(Q, q_ref, q_weights, interp, div);
  CeedBasisCreateHdiv(ceed, CEED_QUAD, num_comp_u, elem_node, num_qpts,
                      interp, div, q_ref, q_weights, &ceed_data->basis_u);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q,
                                  problem_data->quadrature_mode, &ceed_data->basis_x);
  // ---------------------------------------------------------------------------
  // libCEED restrictions
  // ---------------------------------------------------------------------------
  ierr = DMGetCoordinateDM(dm, &dm_coord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
  CeedInt height = 0; // No BC
  DMLabel domain_label = 0;
  PetscInt value = 0;
  // -- Coordinate restriction
  ierr = CreateRestrictionFromPlex(ceed, dm_coord, height, domain_label,
                                   value, 2, &ceed_data->elem_restr_x); CHKERRQ(ierr);
  // -- Solution restriction
  ierr = CreateRestrictionFromPlexOriented(ceed, dm, height, domain_label,
         value, P, &ceed_data->elem_restr_u); CHKERRQ(ierr);
  // ---- Geometric ceed_data restriction
  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  num_elem = c_end - c_start;

  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, geo_data_size,
                                   num_elem*num_qpts*geo_data_size,
                                   CEED_STRIDES_BACKEND, &ceed_data->elem_restr_geo_data_i);

  // ---------------------------------------------------------------------------
  // Element coordinates
  // ---------------------------------------------------------------------------
  ierr = DMGetCoordinatesLocal(dm, &coords); CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords, &coordArray); CHKERRQ(ierr);

  CeedElemRestrictionCreateVector(ceed_data->elem_restr_x, &x_coord, NULL);
  CeedVectorSetArray(x_coord, CEED_MEM_HOST, CEED_COPY_VALUES,
                     (PetscScalar *)coordArray);
  ierr = VecRestoreArrayRead(coords, &coordArray); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Setup RHS and true solution
  // ---------------------------------------------------------------------------
  // Create the q-function that sets up the RHS and true solution
  CeedQFunctionCreateInterior(ceed, 1, problem_data->setup_rhs,
                              problem_data->setup_rhs_loc, &qf_setup_rhs);
  CeedQFunctionAddInput(qf_setup_rhs, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup_rhs, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup_rhs, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup_rhs, "rhs", dim, CEED_EVAL_INTERP);
  // Create the operator that builds the RHS and true solution
  CeedOperatorCreate(ceed, qf_setup_rhs, NULL, NULL, &op_setup_rhs);
  CeedOperatorSetField(op_setup_rhs, "x", ceed_data->elem_restr_x,
                       ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_rhs, "weight", CEED_ELEMRESTRICTION_NONE,
                       ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_rhs, "dx", ceed_data->elem_restr_x,
                       ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_rhs, "rhs", ceed_data->elem_restr_u,
                       ceed_data->basis_u, CEED_VECTOR_ACTIVE);

  // Setup RHS and target
  CeedOperatorApply(op_setup_rhs, x_coord, rhs_ceed, CEED_REQUEST_IMMEDIATE);


  // ---------------------------------------------------------------------------
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  // -- Operator action variables
  CeedVectorCreate(ceed, U_loc_size, &ceed_data->x_ceed);
  CeedVectorCreate(ceed, U_loc_size, &ceed_data->y_ceed);

  // Local residual evaluator
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the residual of the PDE.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data->residual,
                              problem_data->residual_loc, &qf_residual);
  CeedQFunctionAddInput(qf_residual, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_residual, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_residual, "u", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_residual, "v", dim, CEED_EVAL_INTERP);

  // -- Operator
  CeedOperatorCreate(ceed, qf_residual, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_residual);
  CeedOperatorSetField(op_residual, "weight", CEED_ELEMRESTRICTION_NONE,
                       ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_residual, "dx", ceed_data->elem_restr_x,
                       ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_residual, "u", ceed_data->elem_restr_u,
                       ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_residual, "v", ceed_data->elem_restr_u,
                       ceed_data->basis_u, CEED_VECTOR_ACTIVE);

  // -- Save libCEED data
  ceed_data->qf_residual = qf_residual;
  ceed_data->op_residual = op_residual;

  // Cleanup
  CeedQFunctionDestroy(&qf_setup_rhs);
  CeedOperatorDestroy(&op_setup_rhs);
  CeedVectorDestroy(&x_coord);

  PetscFunctionReturn(0);
};
// -----------------------------------------------------------------------------