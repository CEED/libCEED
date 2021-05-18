
#include "../navierstokes.h"

// -----------------------------------------------------------------------------
// libCEED Functions
// -----------------------------------------------------------------------------
// Utility function - essential BC dofs are encoded in closure indices as -(i+1).
PetscInt Involute(PetscInt i) {
  return i >= 0 ? i : -(i+1);
}

// Utility function to create local CEED restriction
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt height, DMLabel domain_label,
    CeedInt value, CeedElemRestriction *elem_restr) {
  PetscSection   section;
  PetscInt       p, num_elem, num_dofs, *elem_restrict, elem_offset, num_fields,
                 dim, depth;
  DMLabel        depth_label;
  IS             depth_IS, iter_IS;
  Vec            U_loc;
  const PetscInt *iter_indices;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &num_fields); CHKERRQ(ierr);
  PetscInt num_comp[num_fields], field_off[num_fields+1];
  field_off[0] = 0;
  for (PetscInt f=0; f<num_fields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &num_comp[f]); CHKERRQ(ierr);
    field_off[f+1] = field_off[f] + num_comp[f];
  }

  ierr = DMPlexGetDepth(dm, &depth); CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depth_label); CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depth_label, depth - height, &depth_IS);
  CHKERRQ(ierr);
  if (domain_label) {
    IS domain_IS;
    ierr = DMLabelGetStratumIS(domain_label, value, &domain_IS); CHKERRQ(ierr);
    if (domain_IS) { // domain_IS is non-empty
      ierr = ISIntersect(depth_IS, domain_IS, &iter_IS); CHKERRQ(ierr);
      ierr = ISDestroy(&domain_IS); CHKERRQ(ierr);
    } else { // domain_IS is NULL (empty)
      iter_IS = NULL;
    }
    ierr = ISDestroy(&depth_IS); CHKERRQ(ierr);
  } else {
    iter_IS = depth_IS;
  }
  if (iter_IS) {
    ierr = ISGetLocalSize(iter_IS, &num_elem); CHKERRQ(ierr);
    ierr = ISGetIndices(iter_IS, &iter_indices); CHKERRQ(ierr);
  } else {
    num_elem = 0;
    iter_indices = NULL;
  }
  ierr = PetscMalloc1(num_elem*PetscPowInt(P, dim), &elem_restrict);
  CHKERRQ(ierr);
  for (p=0, elem_offset=0; p<num_elem; p++) {
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
    for (PetscInt i=0; i<num_nodes; i++) {
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
      for (PetscInt f=0; f<num_fields; f++) {
        for (PetscInt j=0; j<num_comp[f]; j++) {
          if (Involute(indices[field_off[f]*num_nodes + ii*num_comp[f] + j])
              != Involute(indices[ii*num_comp[0]]) + field_off[f] + j)
            SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                     "Cell %D closure indices not interlaced for node %D field %D component %D",
                     c, ii, f, j);
        }
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = Involute(indices[ii*num_comp[0]]);
      elem_restrict[elem_offset++] = loc;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE,
                                       &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }
  if (elem_offset != num_elem*PetscPowInt(P, dim))
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB,
             "ElemRestriction of size (%D,%D) initialized %D nodes", num_elem,
             PetscPowInt(P, dim),elem_offset);
  if (iter_IS) {
    ierr = ISRestoreIndices(iter_IS, &iter_indices); CHKERRQ(ierr);
  }
  ierr = ISDestroy(&iter_IS); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &U_loc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(U_loc, &num_dofs); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &U_loc); CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, num_elem, PetscPowInt(P, dim),
                            field_off[num_fields],
                            1, num_dofs, CEED_MEM_HOST, CEED_COPY_VALUES, elem_restrict,
                            elem_restr);
  ierr = PetscFree(elem_restrict); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height,
                                       DMLabel domain_label, PetscInt value,
                                       CeedInt P, CeedInt Q, CeedInt q_data_size,
                                       CeedElemRestriction *elem_restr_q,
                                       CeedElemRestriction *elem_restr_x,
                                       CeedElemRestriction *elem_restr_qd_i) {

  DM             dm_coord;
  CeedInt        dim, loc_num_elem;
  CeedInt        Q_dim;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  Q_dim = CeedIntPow(Q, dim);
  ierr = DMGetCoordinateDM(dm, &dm_coord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dm, P, height, domain_label, value,
                                   elem_restr_q);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dm_coord, 2, height, domain_label, value,
                                   elem_restr_x);
  CHKERRQ(ierr);
  CeedElemRestrictionGetNumElements(*elem_restr_q, &loc_num_elem);
  CeedElemRestrictionCreateStrided(ceed, loc_num_elem, Q_dim,
                                   q_data_size, q_data_size*loc_num_elem*Q_dim,
                                   CEED_STRIDES_BACKEND, elem_restr_qd_i);
  PetscFunctionReturn(0);
}

// Utility function to create CEED Composite Operator for the entire domain
PetscErrorCode CreateOperatorForDomain(Ceed ceed, DM dm, SimpleBC bc,
                                       CeedData ceed_data, CeedVector x_coord, Physics phys,
                                       CeedOperator op_apply_vol, CeedInt height,
                                       CeedInt P_sur, CeedInt Q_sur, CeedInt q_data_size_sur,
                                       CeedOperator *op_apply) {
  CeedInt        dim, num_face;
  DMLabel        domain_label;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Create Composite Operaters
  CeedCompositeOperatorCreate(ceed, op_apply);

  // --Apply Sub-Operator for the volume
  CeedCompositeOperatorAddSub(*op_apply, op_apply_vol);

  // -- Create Sub-Operator for in/outflow BCs
  if (phys->has_neumann == PETSC_TRUE) {
    // --- Setup
    ierr = DMGetLabel(dm, "Face Sets", &domain_label); CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
    if (dim == 2) num_face = 4;
    if (dim == 3) num_face = 6;

    // --- Get number of quadrature points for the boundaries
    CeedInt num_qpts_sur;
    CeedBasisGetNumQuadraturePoints(ceed_data->basis_q_sur, &num_qpts_sur);

    // --- Create Sub-Operator for each face
    for (CeedInt i=0; i<num_face; i++) {
      CeedVector          q_data_sur;
      CeedOperator        op_setup_sur, op_apply_sur;
      CeedElemRestriction elem_restr_x_sur, elem_restr_q_sur,
                          elem_restr_qd_i_sur;

      // ---- CEED Restriction
      ierr = GetRestrictionForDomain(ceed, dm, height, domain_label, i+1, P_sur,
                                     Q_sur, q_data_size_sur, &elem_restr_q_sur,
                                     &elem_restr_x_sur, &elem_restr_qd_i_sur);
      CHKERRQ(ierr);

      // ---- CEED Vector
      PetscInt loc_num_elem_sur;
      CeedElemRestrictionGetNumElements(elem_restr_q_sur, &loc_num_elem_sur);
      CeedVectorCreate(ceed, q_data_size_sur*loc_num_elem_sur*num_qpts_sur,
                       &q_data_sur);

      // ---- CEED Operator
      // ----- CEED Operator for Setup (geometric factors)
      CeedOperatorCreate(ceed, ceed_data->qf_setup_sur, NULL, NULL, &op_setup_sur);
      CeedOperatorSetField(op_setup_sur, "dx", elem_restr_x_sur,
                           ceed_data->basis_x_sur,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_setup_sur, "weight", CEED_ELEMRESTRICTION_NONE,
                           ceed_data->basis_x_sur, CEED_VECTOR_NONE);
      CeedOperatorSetField(op_setup_sur, "q_data_sur", elem_restr_qd_i_sur,
                           CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

      // ----- CEED Operator for Physics
      CeedOperatorCreate(ceed, ceed_data->qf_apply_sur, NULL, NULL, &op_apply_sur);
      CeedOperatorSetField(op_apply_sur, "q", elem_restr_q_sur,
                           ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_apply_sur, "q_data_sur", elem_restr_qd_i_sur,
                           CEED_BASIS_COLLOCATED, q_data_sur);
      CeedOperatorSetField(op_apply_sur, "x", elem_restr_x_sur,
                           ceed_data->basis_x_sur, x_coord);
      CeedOperatorSetField(op_apply_sur, "v", elem_restr_q_sur,
                           ceed_data->basis_q_sur, CEED_VECTOR_ACTIVE);

      // ----- Apply CEED operator for Setup
      CeedOperatorApply(op_setup_sur, x_coord, q_data_sur, CEED_REQUEST_IMMEDIATE);

      // ----- Apply Sub-Operator for the Boundary
      CeedCompositeOperatorAddSub(*op_apply, op_apply_sur);

      // ----- Cleanup
      CeedVectorDestroy(&q_data_sur);
      CeedElemRestrictionDestroy(&elem_restr_q_sur);
      CeedElemRestrictionDestroy(&elem_restr_x_sur);
      CeedElemRestrictionDestroy(&elem_restr_qd_i_sur);
      CeedOperatorDestroy(&op_setup_sur);
      CeedOperatorDestroy(&op_apply_sur);
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode SetupLibceed(Ceed ceed, CeedData ceed_data, DM dm, User user,
                            AppCtx app_ctx, ProblemData *problem, SimpleBC bc) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // *****************************************************************************
  // Set up CEED objects for the interior domain (volume)
  // *****************************************************************************
  const PetscInt num_comp_q      = 5;
  const CeedInt  dim             = problem->dim,
                 num_comp_x      = problem->dim,
                 q_data_size_vol = problem->q_data_size_vol,
                 P               = app_ctx->degree + 1,
                 Q               = P + app_ctx->q_extra;

  // -----------------------------------------------------------------------------
  // CEED Bases
  // -----------------------------------------------------------------------------
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_q, P, Q, CEED_GAUSS,
                                  &ceed_data->basis_q);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q, CEED_GAUSS,
                                  &ceed_data->basis_x);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, P,
                                  CEED_GAUSS_LOBATTO, &ceed_data->basis_xc);

  // -----------------------------------------------------------------------------
  // CEED Restrictions
  // -----------------------------------------------------------------------------
  // -- Create restriction
  ierr = GetRestrictionForDomain(ceed, dm, 0, 0, 0, P, Q,
                                 q_data_size_vol, &ceed_data->elem_restr_q, &ceed_data->elem_restr_x,
                                 &ceed_data->elem_restr_qd_i); CHKERRQ(ierr);
  // -- Create E vectors
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->q_ceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->q_dot_ceed,
                                  NULL);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->g_ceed, NULL);

  // -----------------------------------------------------------------------------
  // CEED QFunctions
  // -----------------------------------------------------------------------------
  // -- Create QFunction for quadrature data
  CeedQFunctionCreateInterior(ceed, 1, problem->setup_vol, problem->setup_vol_loc,
                              &ceed_data->qf_setup_vol);
  CeedQFunctionAddInput(ceed_data->qf_setup_vol, "dx", num_comp_x*dim,
                        CEED_EVAL_GRAD);
  CeedQFunctionAddInput(ceed_data->qf_setup_vol, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(ceed_data->qf_setup_vol, "q_data", q_data_size_vol,
                         CEED_EVAL_NONE);

  // -- Create QFunction for ICs
  CeedQFunctionCreateInterior(ceed, 1, problem->ics, problem->ics_loc,
                              &ceed_data->qf_ics);
  CeedQFunctionAddInput(ceed_data->qf_ics, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(ceed_data->qf_ics, "q0", num_comp_q, CEED_EVAL_NONE);

  // -- Create QFunction for RHS
  if (problem->apply_vol_rhs) {
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_rhs,
                                problem->apply_vol_rhs_loc, &ceed_data->qf_rhs_vol);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "q", num_comp_q, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "dq", num_comp_q*dim,
                          CEED_EVAL_GRAD);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "q_data", q_data_size_vol,
                          CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_rhs_vol, "v", num_comp_q,
                           CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_rhs_vol, "dv", num_comp_q*dim,
                           CEED_EVAL_GRAD);
  }

  // -- Create QFunction for IFunction
  if (problem->apply_vol_ifunction) {
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_vol_ifunction,
                                problem->apply_vol_ifunction_loc, &ceed_data->qf_ifunction_vol);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "q", num_comp_q,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "dq", num_comp_q*dim,
                          CEED_EVAL_GRAD);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "q_dot", num_comp_q,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "q_data", q_data_size_vol,
                          CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "x", num_comp_x,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "v", num_comp_q,
                           CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "dv", num_comp_q*dim,
                           CEED_EVAL_GRAD);
  }

  // ---------------------------------------------------------------------------
  // Element coordinates
  // ---------------------------------------------------------------------------
  // -- Create CEED vector
  CeedVector x_coord;
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_x, &x_coord, NULL);

  // -- Copy PETSc vector in CEED vector
  Vec               X_loc;
  const PetscScalar *X_loc_array;
  ierr = DMGetCoordinatesLocal(dm, &X_loc); CHKERRQ(ierr);
  ierr = VecGetArrayRead(X_loc, &X_loc_array); CHKERRQ(ierr);
  CeedVectorSetArray(x_coord, CEED_MEM_HOST, CEED_COPY_VALUES,
                     (PetscScalar *)X_loc_array);
  ierr = VecRestoreArrayRead(X_loc, &X_loc_array); CHKERRQ(ierr);

  // -----------------------------------------------------------------------------
  // CEED vectors
  // -----------------------------------------------------------------------------
  // -- Create CEED vector for geometric data
  CeedInt  num_qpts_vol;
  PetscInt loc_num_elem_vol;
  CeedBasisGetNumQuadraturePoints(ceed_data->basis_q, &num_qpts_vol);
  CeedElemRestrictionGetNumElements(ceed_data->elem_restr_q, &loc_num_elem_vol);
  CeedVectorCreate(ceed, q_data_size_vol*loc_num_elem_vol*num_qpts_vol,
                   &ceed_data->q_data);

  // -----------------------------------------------------------------------------
  // CEED Operators
  // -----------------------------------------------------------------------------
  // -- Create CEED operator for quadrature data
  CeedOperatorCreate(ceed, ceed_data->qf_setup_vol, NULL, NULL,
                     &ceed_data->op_setup_vol);
  CeedOperatorSetField(ceed_data->op_setup_vol, "dx", ceed_data->elem_restr_x,
                       ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(ceed_data->op_setup_vol, "weight",
                       CEED_ELEMRESTRICTION_NONE,
                       ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(ceed_data->op_setup_vol, "q_data",
                       ceed_data->elem_restr_qd_i,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // -- Create CEED operator for ICs
  CeedOperatorCreate(ceed, ceed_data->qf_ics, NULL, NULL, &ceed_data->op_ics);
  CeedOperatorSetField(ceed_data->op_ics, "x", ceed_data->elem_restr_x,
                       ceed_data->basis_xc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(ceed_data->op_ics, "q0", ceed_data->elem_restr_q,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create CEED operator for RHS
  if (ceed_data->qf_rhs_vol) {
    CeedOperator op;
    CeedOperatorCreate(ceed, ceed_data->qf_rhs_vol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "q_data", ceed_data->elem_restr_qd_i,
                         CEED_BASIS_COLLOCATED,
                         ceed_data->q_data);
    CeedOperatorSetField(op, "x", ceed_data->elem_restr_x, ceed_data->basis_x,
                         x_coord);
    CeedOperatorSetField(op, "v", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    user->op_rhs_vol = op;
  }

  // -- CEED operator for IFunction
  if (ceed_data->qf_ifunction_vol) {
    CeedOperator op;
    CeedOperatorCreate(ceed, ceed_data->qf_ifunction_vol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "q_dot", ceed_data->elem_restr_q, ceed_data->basis_q,
                         user->q_dot_ceed);
    CeedOperatorSetField(op, "q_data", ceed_data->elem_restr_qd_i,
                         CEED_BASIS_COLLOCATED,
                         ceed_data->q_data);
    CeedOperatorSetField(op, "x", ceed_data->elem_restr_x, ceed_data->basis_x,
                         x_coord);
    CeedOperatorSetField(op, "v", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    user->op_ifunction_vol = op;
  }

  // *****************************************************************************
  // Set up CEED objects for the exterior domain (surface)
  // *****************************************************************************
  CeedInt height  = 1,
          dim_sur = dim - height,
          P_sur   = app_ctx->degree + 1,
          Q_sur   = P_sur + app_ctx->q_extra;
  const CeedInt q_data_size_sur = problem->q_data_size_sur;

  // -----------------------------------------------------------------------------
  // CEED Bases
  // -----------------------------------------------------------------------------
  CeedBasisCreateTensorH1Lagrange(ceed, dim_sur, num_comp_q, P_sur, Q_sur,
                                  CEED_GAUSS, &ceed_data->basis_q_sur);

  CeedBasisCreateTensorH1Lagrange(ceed, dim_sur, num_comp_x, 2, Q_sur, CEED_GAUSS,
                                  &ceed_data->basis_x_sur);

  CeedBasisCreateTensorH1Lagrange(ceed, dim_sur, num_comp_x, 2, P_sur,
                                  CEED_GAUSS_LOBATTO, &ceed_data->basis_xc_sur);

  // -----------------------------------------------------------------------------
  // CEED QFunctions
  // -----------------------------------------------------------------------------
  // -- Create QFunction for quadrature data
  CeedQFunctionCreateInterior(ceed, 1, problem->setup_sur, problem->setup_sur_loc,
                              &ceed_data->qf_setup_sur);
  CeedQFunctionAddInput(ceed_data->qf_setup_sur, "dx", num_comp_x*dim_sur,
                        CEED_EVAL_GRAD);
  CeedQFunctionAddInput(ceed_data->qf_setup_sur, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(ceed_data->qf_setup_sur, "q_data_sur", q_data_size_sur,
                         CEED_EVAL_NONE);

  // -- Creat QFunction for the physics on the boundaries
  if (problem->apply_sur) {
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_sur, problem->apply_sur_loc,
                                &ceed_data->qf_apply_sur);
    CeedQFunctionAddInput(ceed_data->qf_apply_sur, "q", num_comp_q,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_apply_sur, "q_data_sur", q_data_size_sur,
                          CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_apply_sur, "x", num_comp_x,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_apply_sur, "v", num_comp_q,
                           CEED_EVAL_INTERP);
  }

  // *****************************************************************************
  // CEED Operator Apply
  // *****************************************************************************
  // -- Apply CEED Operator for the geometric data
  CeedOperatorApply(ceed_data->op_setup_vol, x_coord, ceed_data->q_data,
                    CEED_REQUEST_IMMEDIATE);

  // -- Create and apply CEED Composite Operator for the entire domain
  if (!user->phys->implicit) { // RHS
    ierr = CreateOperatorForDomain(ceed, dm, bc, ceed_data, x_coord, user->phys,
                                   user->op_rhs_vol, height, P_sur, Q_sur,
                                   q_data_size_sur, &user->op_rhs); CHKERRQ(ierr);
  } else { // IFunction
    ierr = CreateOperatorForDomain(ceed, dm, bc, ceed_data, x_coord, user->phys,
                                   user->op_ifunction_vol, height, P_sur, Q_sur,
                                   q_data_size_sur, &user->op_ifunction); CHKERRQ(ierr);
  }

  // Cleanup
  CeedVectorDestroy(&x_coord);

  PetscFunctionReturn(0);
}

// Set up contex for QFunctions
PetscErrorCode SetupContextForProblems(Ceed ceed, CeedData ceed_data,
                                       AppCtx app_ctx, SetupContext setup_ctx, Physics phys) {
  PetscFunctionBeginUser;

  // ICs
  CeedQFunctionContextCreate(ceed, &ceed_data->setup_context);
  CeedQFunctionContextSetData(ceed_data->setup_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof *setup_ctx, setup_ctx);
  if (ceed_data->qf_ics && strcmp(app_ctx->problem_name, "euler_vortex") != 0)
    CeedQFunctionSetContext(ceed_data->qf_ics, ceed_data->setup_context);

  // DENSITY_CURRENT
  if (strcmp(app_ctx->problem_name, "density_current") == 0) {
    CeedQFunctionContextCreate(ceed, &ceed_data->dc_context);
    CeedQFunctionContextSetData(ceed_data->dc_context, CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                sizeof phys->dc_ctx, phys->dc_ctx);
    if (ceed_data->qf_rhs_vol)
      CeedQFunctionSetContext(ceed_data->qf_rhs_vol, ceed_data->dc_context);
    if (ceed_data->qf_ifunction_vol)
      CeedQFunctionSetContext(ceed_data->qf_ifunction_vol, ceed_data->dc_context);

    // EULER_VORTEX
  } else if (strcmp(app_ctx->problem_name, "euler_vortex") == 0) {
    CeedQFunctionContextCreate(ceed, &ceed_data->euler_context);
    CeedQFunctionContextSetData(ceed_data->euler_context, CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                sizeof phys->euler_ctx, phys->euler_ctx);
    if (ceed_data->qf_ics)
      CeedQFunctionSetContext(ceed_data->qf_ics, ceed_data->euler_context);
    if (ceed_data->qf_apply_sur)
      CeedQFunctionSetContext(ceed_data->qf_apply_sur, ceed_data->euler_context);

    // ADVECTION and ADVECTION2D
  } else {
    CeedQFunctionContextCreate(ceed, &ceed_data->advection_context);
    CeedQFunctionContextSetData(ceed_data->advection_context, CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                sizeof phys->advection_ctx, phys->advection_ctx);
    if (ceed_data->qf_rhs_vol)
      CeedQFunctionSetContext(ceed_data->qf_rhs_vol, ceed_data->advection_context);
    if (ceed_data->qf_ifunction_vol)
      CeedQFunctionSetContext(ceed_data->qf_ifunction_vol,
                              ceed_data->advection_context);
    if (ceed_data->qf_apply_sur)
      CeedQFunctionSetContext(ceed_data->qf_apply_sur, ceed_data->advection_context);
  }

  PetscFunctionReturn(0);
}
