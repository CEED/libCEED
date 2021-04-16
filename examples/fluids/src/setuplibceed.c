
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
  PetscSection section;
  PetscInt p, Nelem, Ndof, *erestrict, eoffset, nfields, dim, depth;
  DMLabel depthLabel;
  IS depthIS, iterIS;
  Vec Uloc;
  const PetscInt *iterIndices;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &nfields); CHKERRQ(ierr);
  PetscInt ncomp[nfields], fieldoff[nfields+1];
  fieldoff[0] = 0;
  for (PetscInt f=0; f<nfields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &ncomp[f]); CHKERRQ(ierr);
    fieldoff[f+1] = fieldoff[f] + ncomp[f];
  }

  ierr = DMPlexGetDepth(dm, &depth); CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel); CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel, depth - height, &depthIS); CHKERRQ(ierr);
  if (domain_label) {
    IS domainIS;
    ierr = DMLabelGetStratumIS(domain_label, value, &domainIS); CHKERRQ(ierr);
    if (domainIS) { // domainIS is non-empty
      ierr = ISIntersect(depthIS, domainIS, &iterIS); CHKERRQ(ierr);
      ierr = ISDestroy(&domainIS); CHKERRQ(ierr);
    } else { // domainIS is NULL (empty)
      iterIS = NULL;
    }
    ierr = ISDestroy(&depthIS); CHKERRQ(ierr);
  } else {
    iterIS = depthIS;
  }
  if (iterIS) {
    ierr = ISGetLocalSize(iterIS, &Nelem); CHKERRQ(ierr);
    ierr = ISGetIndices(iterIS, &iterIndices); CHKERRQ(ierr);
  } else {
    Nelem = 0;
    iterIndices = NULL;
  }
  ierr = PetscMalloc1(Nelem*PetscPowInt(P, dim), &erestrict); CHKERRQ(ierr);
  for (p=0,eoffset=0; p<Nelem; p++) {
    PetscInt c = iterIndices[p];
    PetscInt numindices, *indices, nnodes;
    ierr = DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE,
                                   &numindices, &indices, NULL, NULL);
    CHKERRQ(ierr);
    bool flip = false;
    if (height > 0) {
      PetscInt numCells, numFaces, start = -1;
      const PetscInt *orients, *faces, *cells;
      ierr = DMPlexGetSupport(dm, c, &cells); CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, c, &numCells); CHKERRQ(ierr);
      if (numCells != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                                    "Expected one cell in support of exterior face, but got %D cells",
                                    numCells);
      ierr = DMPlexGetCone(dm, cells[0], &faces); CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, cells[0], &numFaces); CHKERRQ(ierr);
      for (PetscInt i=0; i<numFaces; i++) {if (faces[i] == c) start = i;}
      if (start < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT,
                                "Could not find face %D in cone of its support",
                                c);
      ierr = DMPlexGetConeOrientation(dm, cells[0], &orients); CHKERRQ(ierr);
      if (orients[start] < 0) flip = true;
    }
    if (numindices % fieldoff[nfields]) SETERRQ1(PETSC_COMM_SELF,
          PETSC_ERR_ARG_INCOMP, "Number of closure indices not compatible with Cell %D",
          c);
    nnodes = numindices / fieldoff[nfields];
    for (PetscInt i=0; i<nnodes; i++) {
      PetscInt ii = i;
      if (flip) {
        if (P == nnodes) ii = nnodes - 1 - i;
        else if (P*P == nnodes) {
          PetscInt row = i / P, col = i % P;
          ii = row + col * P;
        } else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP,
                          "No support for flipping point with %D nodes != P (%D) or P^2",
                          nnodes, P);
      }
      // Check that indices are blocked by node and thus can be coalesced as a single field with
      // fieldoff[nfields] = sum(ncomp) components.
      for (PetscInt f=0; f<nfields; f++) {
        for (PetscInt j=0; j<ncomp[f]; j++) {
          if (Involute(indices[fieldoff[f]*nnodes + ii*ncomp[f] + j])
              != Involute(indices[ii*ncomp[0]]) + fieldoff[f] + j)
            SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                     "Cell %D closure indices not interlaced for node %D field %D component %D",
                     c, ii, f, j);
        }
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = Involute(indices[ii*ncomp[0]]);
      erestrict[eoffset++] = loc;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE,
                                       &numindices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }
  if (eoffset != Nelem*PetscPowInt(P, dim))
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB,
             "ElemRestriction of size (%D,%D) initialized %D nodes", Nelem,
             PetscPowInt(P, dim),eoffset);
  if (iterIS) {
    ierr = ISRestoreIndices(iterIS, &iterIndices); CHKERRQ(ierr);
  }
  ierr = ISDestroy(&iterIS); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &Uloc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Uloc, &Ndof); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Uloc); CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, Nelem, PetscPowInt(P, dim), fieldoff[nfields],
                            1, Ndof, CEED_MEM_HOST, CEED_COPY_VALUES, erestrict,
                            elem_restr);
  ierr = PetscFree(erestrict); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height,
                                       DMLabel domain_label, PetscInt value,
                                       CeedInt P, CeedInt Q, CeedInt q_data_size,
                                       CeedElemRestriction *elem_restr_q,
                                       CeedElemRestriction *elem_restr_x,
                                       CeedElemRestriction *elem_restr_qd_i) {

  DM dmcoord;
  CeedInt dim, localNelem;
  CeedInt Qdim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  Qdim = CeedIntPow(Q, dim);
  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dm, P, height, domain_label, value,
                                   elem_restr_q);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dmcoord, 2, height, domain_label, value,
                                   elem_restr_x);
  CHKERRQ(ierr);
  CeedElemRestrictionGetNumElements(*elem_restr_q, &localNelem);
  CeedElemRestrictionCreateStrided(ceed, localNelem, Qdim,
                                   q_data_size, q_data_size*localNelem*Qdim,
                                   CEED_STRIDES_BACKEND, elem_restr_qd_i);
  PetscFunctionReturn(0);
}

// Utility function to create CEED Composite Operator for the entire domain
PetscErrorCode CreateOperatorForDomain(Ceed ceed, DM dm, SimpleBC bc,
                                       Physics phys, CeedOperator op_apply_vol, CeedQFunction qf_apply_sur,
                                       CeedQFunction qf_setup_sur, CeedInt height, CeedInt P_Sur, CeedInt Q_sur,
                                       CeedInt q_data_size_sur, CeedInt num_qpts_sur, CeedBasis basis_x_sur,
                                       CeedBasis basis_q_sur, CeedOperator *op_apply) {
  CeedInt dim, nFace;
  PetscInt lsize;
  Vec X_loc;
  CeedVector x_corners;
  DMLabel domain_label;
  PetscScalar *x;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  // Composite Operaters
  CeedCompositeOperatorCreate(ceed, op_apply);
  // --Apply a Sub-Operator for the volume
  CeedCompositeOperatorAddSub(*op_apply, op_apply_vol);

  // Required data for in/outflow BCs
  ierr = DMGetCoordinatesLocal(dm, &X_loc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(X_loc, &lsize); CHKERRQ(ierr);
  ierr = CeedVectorCreate(ceed, lsize, &x_corners); CHKERRQ(ierr);
  ierr = VecGetArray(X_loc, &x); CHKERRQ(ierr);
  CeedVectorSetArray(x_corners, CEED_MEM_HOST, CEED_USE_POINTER, x);
  ierr = DMGetLabel(dm, "Face Sets", &domain_label); CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  if (phys->has_neumann == PETSC_TRUE) {
    if (phys->wind_type == ADVECTION_WIND_TRANSLATION)
      bc->num_wall = bc->num_slip[0] = bc->num_slip[1] = bc->num_slip[2] = 0;

    // Set number of faces
    if (dim == 2) nFace = 4;
    if (dim == 3) nFace = 6;

    // Create CEED Operator for each boundary face
    PetscInt localNelemSur[6];
    CeedVector q_dataSur[6];
    CeedOperator op_setup_sur[6], op_apply_sur[6];
    CeedElemRestriction elem_restr_xSur[6], elem_restr_qSur[6],
                        elem_restr_qd_iSur[6];

    for (CeedInt i=0; i<nFace; i++) {
      ierr = GetRestrictionForDomain(ceed, dm, height, domain_label, i+1, P_Sur,
                                     Q_sur, q_data_size_sur, &elem_restr_qSur[i],
                                     &elem_restr_xSur[i], &elem_restr_qd_iSur[i]);
      CHKERRQ(ierr);
      // Create the CEED vectors that will be needed in Boundary setup
      CeedElemRestrictionGetNumElements(elem_restr_qSur[i], &localNelemSur[i]);
      CeedVectorCreate(ceed, q_data_size_sur*localNelemSur[i]*num_qpts_sur,
                       &q_dataSur[i]);
      // Create the operator that builds the quadrature data for the Boundary operator
      CeedOperatorCreate(ceed, qf_setup_sur, NULL, NULL, &op_setup_sur[i]);
      CeedOperatorSetField(op_setup_sur[i], "dx", elem_restr_xSur[i], basis_x_sur,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_setup_sur[i], "weight", CEED_ELEMRESTRICTION_NONE,
                           basis_x_sur, CEED_VECTOR_NONE);
      CeedOperatorSetField(op_setup_sur[i], "q_dataSur", elem_restr_qd_iSur[i],
                           CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
      // Create Boundary operator
      CeedOperatorCreate(ceed, qf_apply_sur, NULL, NULL, &op_apply_sur[i]);
      CeedOperatorSetField(op_apply_sur[i], "q", elem_restr_qSur[i], basis_q_sur,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_apply_sur[i], "q_dataSur", elem_restr_qd_iSur[i],
                           CEED_BASIS_COLLOCATED, q_dataSur[i]);
      CeedOperatorSetField(op_apply_sur[i], "x", elem_restr_xSur[i], basis_x_sur,
                           x_corners);
      CeedOperatorSetField(op_apply_sur[i], "v", elem_restr_qSur[i], basis_q_sur,
                           CEED_VECTOR_ACTIVE);
      // Apply CEED operator for Boundary setup
      CeedOperatorApply(op_setup_sur[i], x_corners, q_dataSur[i],
                        CEED_REQUEST_IMMEDIATE);
      // --Apply Sub-Operator for the Boundary
      CeedCompositeOperatorAddSub(*op_apply, op_apply_sur[i]);
    }
    CeedVectorDestroy(&x_corners);
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
  const PetscInt num_comp_q = 5;
  const CeedInt  dim = problem->dim,
                 ncompx = problem->dim,
                 q_data_size_vol = problem->q_data_size_vol,
                 P = app_ctx->degree + 1,
                 numQ = P + app_ctx->q_extra;

  // -----------------------------------------------------------------------------
  // CEED Bases
  // -----------------------------------------------------------------------------
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_q, P, numQ, CEED_GAUSS,
                                  &ceed_data->basis_q);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, numQ, CEED_GAUSS,
                                  &ceed_data->basis_x);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, P,
                                  CEED_GAUSS_LOBATTO, &ceed_data->basis_xc);

  // -----------------------------------------------------------------------------
  // CEED Restrictions
  // -----------------------------------------------------------------------------
  // -- Create restriction
  ierr = GetRestrictionForDomain(ceed, dm, 0, 0, 0, P, numQ,
                                 q_data_size_vol, &ceed_data->elem_restr_q, &ceed_data->elem_restr_x,
                                 &ceed_data->elem_restr_qd_i); CHKERRQ(ierr);
  // -- Create E vectors
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &ceed_data->q0_ceed,
                                  NULL);
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
  CeedQFunctionAddInput(ceed_data->qf_setup_vol, "dx", ncompx*dim,
                        CEED_EVAL_GRAD);
  CeedQFunctionAddInput(ceed_data->qf_setup_vol, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(ceed_data->qf_setup_vol, "q_data", q_data_size_vol,
                         CEED_EVAL_NONE);

  // -- Create QFunction for ICs
  CeedQFunctionCreateInterior(ceed, 1, problem->ics, problem->ics_loc,
                              &ceed_data->qf_ics);
  CeedQFunctionAddInput(ceed_data->qf_ics, "x", ncompx, CEED_EVAL_INTERP);
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
    CeedQFunctionAddInput(ceed_data->qf_rhs_vol, "x", ncompx, CEED_EVAL_INTERP);
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
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "qdot", num_comp_q,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "q_data", q_data_size_vol,
                          CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_ifunction_vol, "x", ncompx,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "v", num_comp_q,
                           CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_ifunction_vol, "dv", num_comp_q*dim,
                           CEED_EVAL_GRAD);
  }

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

  // -- Create CEED operator for quadrature data ICs
  CeedOperatorCreate(ceed, ceed_data->qf_ics, NULL, NULL, &ceed_data->op_ics);
  CeedOperatorSetField(ceed_data->op_ics, "x", ceed_data->elem_restr_x,
                       ceed_data->basis_xc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(ceed_data->op_ics, "q0", ceed_data->elem_restr_q,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // -- Create CEED vector for quadrature data which is used in RHS or IFunction
  CeedInt  NqptsVol;
  PetscInt localNelemVol;
  CeedBasisGetNumQuadraturePoints(ceed_data->basis_q, &NqptsVol);
  CeedElemRestrictionGetNumElements(ceed_data->elem_restr_q, &localNelemVol);
  CeedVectorCreate(ceed, q_data_size_vol*localNelemVol*NqptsVol,
                   &ceed_data->q_data);

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
                         ceed_data->x_corners);
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
    CeedOperatorSetField(op, "qdot", ceed_data->elem_restr_q, ceed_data->basis_q,
                         user->q_dot_ceed);
    CeedOperatorSetField(op, "q_data", ceed_data->elem_restr_qd_i,
                         CEED_BASIS_COLLOCATED,
                         ceed_data->q_data);
    CeedOperatorSetField(op, "x", ceed_data->elem_restr_x, ceed_data->basis_x,
                         ceed_data->x_corners);
    CeedOperatorSetField(op, "v", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", ceed_data->elem_restr_q, ceed_data->basis_q,
                         CEED_VECTOR_ACTIVE);
    user->op_ifunction_vol = op;
  }

  // *****************************************************************************
  // Set up CEED objects for the in/outflow boundaries (surface)
  // *****************************************************************************
  CeedInt height = 1,
          dimSur = dim - height,
          P_Sur = app_ctx->degree + 1,  // todo: change it to q_extra_sur
          Q_sur = P_Sur + app_ctx->q_extra_sur,
          num_qpts_sur;
  const CeedInt q_data_size_sur = problem->q_data_size_sur;

  // -----------------------------------------------------------------------------
  // CEED Bases
  // -----------------------------------------------------------------------------
  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, num_comp_q, P_Sur, Q_sur,
                                  CEED_GAUSS, &ceed_data->basis_q_sur);

  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, ncompx, 2, Q_sur, CEED_GAUSS,
                                  &ceed_data->basis_x_sur);

  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, ncompx, 2, P_Sur,
                                  CEED_GAUSS_LOBATTO, &ceed_data->basis_xc_sur);

  // -----------------------------------------------------------------------------
  // CEED QFunctions
  // -----------------------------------------------------------------------------
  // -- Create QFunction for quadrature data
  CeedQFunctionCreateInterior(ceed, 1, problem->setup_sur, problem->setup_sur_loc,
                              &ceed_data->qf_setup_sur);
  CeedQFunctionAddInput(ceed_data->qf_setup_sur, "dx", ncompx*dimSur,
                        CEED_EVAL_GRAD);
  CeedQFunctionAddInput(ceed_data->qf_setup_sur, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(ceed_data->qf_setup_sur, "q_dataSur", q_data_size_sur,
                         CEED_EVAL_NONE);

  // -- Creat QFunction for the physics on the boundaries
  if (problem->apply_sur) {
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_sur, problem->apply_sur_loc,
                                &ceed_data->qf_apply_sur);
    CeedQFunctionAddInput(ceed_data->qf_apply_sur, "q", num_comp_q,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_apply_sur, "q_dataSur", q_data_size_sur,
                          CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_apply_sur, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_apply_sur, "v", num_comp_q,
                           CEED_EVAL_INTERP);
  }

  // *****************************************************************************
  // CEED Operator Apply
  // *****************************************************************************
  // -- Get the number of quadrature points for the boundaries
  //   todo: this can go inside CreateOperatorForDomain()
  CeedBasisGetNumQuadraturePoints(ceed_data->basis_q_sur, &num_qpts_sur);

  // -- Create and apply CEED Composite Operator for the entire domain
  if (!user->phys->implicit) { // RHS
    ierr = CreateOperatorForDomain(ceed, dm, bc, user->phys,
                                   user->op_rhs_vol,
                                   ceed_data->qf_apply_sur, ceed_data->qf_setup_sur,
                                   height, P_Sur, Q_sur, q_data_size_sur,
                                   num_qpts_sur, ceed_data->basis_x_sur, ceed_data->basis_q_sur,
                                   &user->op_rhs); CHKERRQ(ierr);
  } else { // IFunction
    ierr = CreateOperatorForDomain(ceed, dm, bc, user->phys,
                                   user->op_ifunction_vol,
                                   ceed_data->qf_apply_sur, ceed_data->qf_setup_sur,
                                   height, P_Sur, Q_sur, q_data_size_sur,
                                   num_qpts_sur, ceed_data->basis_x_sur, ceed_data->basis_q_sur,
                                   &user->op_ifunction); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// Set up contex for QFunctions
PetscErrorCode SetupContextForProblems(Ceed ceed, CeedData ceed_data,
                                       AppCtx app_ctx, SetupContext setup_ctx, Physics phys) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  CeedQFunctionContextCreate(ceed, &ceed_data->setup_context);
  CeedQFunctionContextSetData(ceed_data->setup_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof *setup_ctx, setup_ctx);

  CeedQFunctionContextCreate(ceed, &ceed_data->dc_context);
  CeedQFunctionContextSetData(ceed_data->dc_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof phys->dc_ctx, phys->dc_ctx);

  CeedQFunctionContextCreate(ceed, &ceed_data->euler_context);
  CeedQFunctionContextSetData(ceed_data->euler_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof phys->euler_ctx, phys->euler_ctx);

  CeedQFunctionContextCreate(ceed, &ceed_data->advection_context);
  CeedQFunctionContextSetData(ceed_data->advection_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof phys->advection_ctx, phys->advection_ctx);

  if (ceed_data->qf_ics && strcmp(app_ctx->problem_name, "euler_vortex") != 0)
    CeedQFunctionSetContext(ceed_data->qf_ics, ceed_data->setup_context);

  if (strcmp(app_ctx->problem_name, "density_current") == 0) {
    if (ceed_data->qf_rhs_vol)
      CeedQFunctionSetContext(ceed_data->qf_rhs_vol, ceed_data->dc_context);

    if (ceed_data->qf_ifunction_vol)
      CeedQFunctionSetContext(ceed_data->qf_ifunction_vol, ceed_data->dc_context);

  } else if (strcmp(app_ctx->problem_name, "euler_vortex") == 0) {
    if (ceed_data->qf_ics)
      CeedQFunctionSetContext(ceed_data->qf_ics, ceed_data->euler_context);

    if (ceed_data->qf_rhs_vol)
      CeedQFunctionSetContext(ceed_data->qf_rhs_vol, ceed_data->euler_context);

    if (ceed_data->qf_ifunction_vol)
      CeedQFunctionSetContext(ceed_data->qf_ifunction_vol, ceed_data->euler_context);

    if (ceed_data->qf_apply_sur)
      CeedQFunctionSetContext(ceed_data->qf_apply_sur, ceed_data->euler_context);
  } else {
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
