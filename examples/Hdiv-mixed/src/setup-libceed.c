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
  CeedVectorDestroy(&ceed_data->geo_data);
  // Restrictions
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_x);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_u);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_geo_data_i);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_u_i);
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
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt topo_dim, CeedElemRestriction *elem_restr) {
  PetscSection section;
  PetscInt p, num_elem, num_dof, *elem_restr_offsets, e_offset, num_fields,
           c_start, c_end;
  Vec U_loc;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &num_fields); CHKERRQ(ierr);
  PetscInt num_comp[num_fields], field_off[num_fields+1];
  field_off[0] = 0;
  for (PetscInt f = 0; f < num_fields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &num_comp[f]); CHKERRQ(ierr);
    field_off[f+1] = field_off[f] + num_comp[f];
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  num_elem = c_end - c_start;
  ierr = PetscMalloc1(num_elem*PetscPowInt(P, topo_dim), &elem_restr_offsets);
  CHKERRQ(ierr);
  CHKERRQ(ierr);
  for (p = 0, e_offset = 0; p < num_elem; p++) {
    PetscInt num_indices, *indices, num_nodes;
    ierr = DMPlexGetClosureIndices(dm, section, section, p, PETSC_TRUE,
                                   &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
    num_nodes = num_indices / field_off[num_fields];
    for (PetscInt i = 0; i < num_nodes; i++) {
      PetscInt ii = i;
      // Check that indices are blocked by node and thus can be coalesced as a single field with
      // field_off[num_fields] = sum(num_comp) components.
      for (PetscInt f = 0; f < num_fields; f++) {
        for (PetscInt j = 0; j < num_comp[f]; j++) {
          if (Involute(indices[field_off[f]*num_nodes + ii*num_comp[f] + j])
              != Involute(indices[ii*num_comp[0]]) + field_off[f] + j)
            SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                     "Cell %D closure indices not interlaced for node %D field %D component %D",
                     p, ii, f, j);
        }
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = Involute(indices[ii*num_comp[0]]);
      elem_restr_offsets[e_offset++] = loc;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, p, PETSC_TRUE,
                                       &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }
  if (e_offset != num_elem*PetscPowInt(P, topo_dim))
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB,
             "ElemRestriction of size (%D,%D) initialized %D nodes", num_elem,
             PetscPowInt(P, topo_dim),e_offset);

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
// Get Oriented CEED restriction data from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateRestrictionFromPlexOriented(Ceed ceed, DM dm, CeedInt P,
    CeedInt topo_dim, CeedElemRestriction *elem_restr_oriented) {
  PetscSection section;
  PetscInt p, num_elem, num_dof, *elem_restr_offsets, e_offset, num_fields,
           c_start, c_end;
  Vec U_loc;
  PetscErrorCode ierr;
  const PetscInt *ornt;

  PetscFunctionBeginUser;
  ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &num_fields); CHKERRQ(ierr);
  PetscInt num_comp[num_fields], field_off[num_fields+1];
  field_off[0] = 0;
  for (PetscInt f = 0; f < num_fields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &num_comp[f]); CHKERRQ(ierr);
    field_off[f+1] = field_off[f] + num_comp[f];
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  num_elem = c_end - c_start;
  ierr = PetscMalloc1(num_elem*topo_dim*PetscPowInt(P, topo_dim),
                      &elem_restr_offsets); // doesn't alocate as many entries
  CHKERRQ(ierr);
  bool *orient;
  ierr = PetscMalloc1(num_elem*PetscPowInt(P, topo_dim), &orient);
  CHKERRQ(ierr);
  for (p = 0, e_offset = 0; p < num_elem; p++) {
    PetscInt num_indices, *indices, num_nodes;
    ierr = DMPlexGetClosureIndices(dm, section, section, p, PETSC_TRUE,
                                   &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
    num_nodes = num_indices /
                field_off[num_fields]; // 8 / 2, but I think there are really 8 nodes
    for (PetscInt i = 0; i < num_nodes; i++) {
      PetscInt ii = i;
      // Check that indices are blocked by node and thus can be coalesced as a single field with
      // field_off[num_fields] = sum(num_comp) components.
      for (PetscInt f = 0; f < num_fields; f++) {
        for (PetscInt j = 0; j < num_comp[f]; j++) {
          if (Involute(indices[field_off[f]*num_nodes + ii*num_comp[f] + j])
              != Involute(indices[ii*num_comp[0]]) + field_off[f] + j)
            SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                     "Cell %D closure indices not interlaced for node %D field %D component %D",
                     p, ii, f, j);
        }
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = Involute(indices[ii*num_comp[0]]);
      elem_restr_offsets[e_offset] = loc; // Are we getting two nodes per edge? yes,
      // Set orientation
      ierr = DMPlexGetConeOrientation(dm, p, &ornt); CHKERRQ(ierr);
      orient[e_offset] = ornt[i] < 0;
      e_offset++;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, p, PETSC_TRUE,
                                       &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }
  if (e_offset != num_elem*topo_dim*PetscPowInt(P,
      topo_dim)) // this probably needs to be like this
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB,
             "ElemRestriction of size (%D,%D) initialized %D nodes", num_elem,
             PetscPowInt(P, topo_dim),e_offset);

  ierr = DMGetLocalVector(dm, &U_loc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(U_loc, &num_dof); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &U_loc); CHKERRQ(ierr);
  // dof per element in Hdiv is dim*P^dim, for linear element P=2
  CeedElemRestrictionCreateOriented(ceed, num_elem, topo_dim*PetscPowInt(P,
                                    topo_dim), // as we're using here
                                    field_off[num_fields],
                                    1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES,
                                    elem_restr_offsets, orient, elem_restr_oriented);
  ierr = PetscFree(elem_restr_offsets); CHKERRQ(ierr);
  ierr = PetscFree(orient); CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Set up libCEED on the fine grid for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupLibceed(DM dm, Ceed ceed, AppCtx app_ctx,
                            ProblemData *problem_data,
                            PetscInt U_g_size, PetscInt U_loc_size,
                            CeedData ceed_data, CeedVector rhs_ceed,
                            CeedVector *target) {
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
  CeedQFunction qf_setup_geo, qf_residual;
  CeedOperator  op_setup_geo, op_residual;

  PetscFunctionBeginUser;
  // ---------------------------------------------------------------------------
  // libCEED bases:Hdiv basis_u and Lagrange basis_x
  // ---------------------------------------------------------------------------
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  num_comp_x = dim;
  num_comp_u = dim;
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
  // -- Coordinate restriction
  ierr = CreateRestrictionFromPlex(ceed, dm_coord, 2, dim,
                                   &ceed_data->elem_restr_x);
  CHKERRQ(ierr);
  // -- Solution restriction
  ierr = CreateRestrictionFromPlexOriented(ceed, dm, P, dim,
         &ceed_data->elem_restr_u);
  // ---- Geometric ceed_data restriction
  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  num_elem = c_end - c_start;
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, geo_data_size,
                                   num_elem*num_qpts*geo_data_size,
                                   CEED_STRIDES_BACKEND, &ceed_data->elem_restr_geo_data_i);
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, num_comp_u,
                                   num_comp_u*num_elem*num_qpts,
                                   CEED_STRIDES_BACKEND, &ceed_data->elem_restr_u_i);
  printf("----elem_restr_x:\n");
  CeedElemRestrictionView(ceed_data->elem_restr_x, stdout);
  printf("----elem_restr_u:\n");
  CeedElemRestrictionView(ceed_data->elem_restr_u, stdout);
  //CeedElemRestrictionView(ceed_data->elem_restr_geo_data_i, stdout);
  //CeedElemRestrictionView(ceed_data->elem_restr_u_i, stdout);

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
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  // -- Operator action variables
  CeedVectorCreate(ceed, U_loc_size, &ceed_data->x_ceed);
  CeedVectorCreate(ceed, U_loc_size, &ceed_data->y_ceed);
  // -- Geometric data vector
  CeedVectorCreate(ceed, num_elem*num_qpts*geo_data_size,
                   &ceed_data->geo_data);

  // ---------------------------------------------------------------------------
  // Geometric factor computation
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the quadrature data
  //   geo_data returns dXdx_i,j and w * det.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data->setup_geo,
                              problem_data->setup_geo_loc, &qf_setup_geo);
  CeedQFunctionAddInput(qf_setup_geo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup_geo, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup_geo, "geo_data", geo_data_size, CEED_EVAL_NONE);
  // -- Operator
  CeedOperatorCreate(ceed, qf_setup_geo, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_setup_geo);
  CeedOperatorSetField(op_setup_geo, "weight", CEED_ELEMRESTRICTION_NONE,
                       ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_geo, "dx", ceed_data->elem_restr_x,
                       ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "geo_data",
                       ceed_data->elem_restr_geo_data_i,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  // -- Compute the quadrature data
  CeedOperatorApply(op_setup_geo, x_coord, ceed_data->geo_data,
                    CEED_REQUEST_IMMEDIATE);
  // -- Cleanup
  CeedQFunctionDestroy(&qf_setup_geo);
  CeedOperatorDestroy(&op_setup_geo);

  // ---------------------------------------------------------------------------
  // Local residual evaluator
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the residual of the PDE.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data->residual,
                              problem_data->residual_loc, &qf_residual);
  CeedQFunctionAddInput(qf_residual, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_residual, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_residual, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_residual, "v", num_comp_u, CEED_EVAL_INTERP);

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
  // ---------------------------------------------------------------------------
  // Setup RHS and true solution
  // ---------------------------------------------------------------------------
  CeedQFunction qf_setup_rhs;
  CeedOperator op_setup_rhs;
  CeedVectorCreate(ceed, num_elem*num_qpts*num_comp_u, target);
  // Create the q-function that sets up the RHS and true solution
  CeedQFunctionCreateInterior(ceed, 1, problem_data->setup_rhs,
                              problem_data->setup_rhs_loc, &qf_setup_rhs);
  CeedQFunctionAddInput(qf_setup_rhs, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup_rhs, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup_rhs, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup_rhs, "true_soln", num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup_rhs, "rhs", num_comp_u, CEED_EVAL_INTERP);
  // Create the operator that builds the RHS and true solution
  CeedOperatorCreate(ceed, qf_setup_rhs, NULL, NULL, &op_setup_rhs);
  CeedOperatorSetField(op_setup_rhs, "x", ceed_data->elem_restr_x,
                       ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_rhs, "weight", CEED_ELEMRESTRICTION_NONE,
                       ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_rhs, "dx", ceed_data->elem_restr_x,
                       ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_rhs, "true_soln", ceed_data->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, *target);
  CeedOperatorSetField(op_setup_rhs, "rhs", ceed_data->elem_restr_u,
                       ceed_data->basis_u, CEED_VECTOR_ACTIVE);

  // Setup RHS and target
  CeedOperatorApply(op_setup_rhs, x_coord, rhs_ceed, CEED_REQUEST_IMMEDIATE);

  // Cleanup
  CeedQFunctionDestroy(&qf_setup_rhs);
  CeedOperatorDestroy(&op_setup_rhs);
  CeedVectorDestroy(&x_coord);

  PetscFunctionReturn(0);
};
// -----------------------------------------------------------------------------