
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
    CeedInt height, DMLabel domainLabel,
    CeedInt value, CeedElemRestriction *Erestrict) {
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
  if (domainLabel) {
    IS domainIS;
    ierr = DMLabelGetStratumIS(domainLabel, value, &domainIS); CHKERRQ(ierr);
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
                            Erestrict);
  ierr = PetscFree(erestrict); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height,
                                       DMLabel domainLabel, PetscInt value,
                                       CeedInt P, CeedInt Q, CeedInt qdatasize,
                                       CeedElemRestriction *restrictq,
                                       CeedElemRestriction *restrictx,
                                       CeedElemRestriction *restrictqdi) {

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
  ierr = CreateRestrictionFromPlex(ceed, dm, P, height, domainLabel, value,
                                   restrictq);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dmcoord, 2, height, domainLabel, value,
                                   restrictx);
  CHKERRQ(ierr);
  CeedElemRestrictionGetNumElements(*restrictq, &localNelem);
  CeedElemRestrictionCreateStrided(ceed, localNelem, Qdim,
                                   qdatasize, qdatasize*localNelem*Qdim,
                                   CEED_STRIDES_BACKEND, restrictqdi);
  PetscFunctionReturn(0);
}

// Utility function to create CEED Composite Operator for the entire domain
PetscErrorCode CreateOperatorForDomain(Ceed ceed, DM dm, SimpleBC bc,
                                       Physics phys, CeedOperator op_applyVol, CeedQFunction qf_applySur,
                                       CeedQFunction qf_setupSur, CeedInt height, CeedInt numP_Sur, CeedInt numQ_Sur,
                                       CeedInt qdatasizeSur, CeedInt NqptsSur, CeedBasis basisxSur,
                                       CeedBasis basisqSur, CeedOperator *op_apply) {

  CeedInt dim, nFace;
  PetscInt lsize;
  Vec Xloc;
  CeedVector xcorners;
  DMLabel domainLabel;
  PetscScalar *x;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  // Composite Operaters
  CeedCompositeOperatorCreate(ceed, op_apply);
  // --Apply a Sub-Operator for the volume
  CeedCompositeOperatorAddSub(*op_apply, op_applyVol);

  // Required data for in/outflow BCs
  ierr = DMGetCoordinatesLocal(dm, &Xloc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Xloc, &lsize); CHKERRQ(ierr);
  ierr = CeedVectorCreate(ceed, lsize, &xcorners); CHKERRQ(ierr);
  ierr = VecGetArray(Xloc, &x); CHKERRQ(ierr);
  CeedVectorSetArray(xcorners, CEED_MEM_HOST, CEED_USE_POINTER, x);
  ierr = DMGetLabel(dm, "Face Sets", &domainLabel); CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  if (phys->has_neumann == PETSC_TRUE) {
    if (phys->wind_type == ADVECTION_WIND_TRANSLATION)
      bc->nwall = bc->nslip[0] = bc->nslip[1] = bc->nslip[2] = 0;

    // Set number of faces
    if (dim == 2) nFace = 4;
    if (dim == 3) nFace = 6;

    // Create CEED Operator for each boundary face
    PetscInt localNelemSur[6];
    CeedVector qdataSur[6];
    CeedOperator op_setupSur[6], op_applySur[6];
    CeedElemRestriction restrictxSur[6], restrictqSur[6], restrictqdiSur[6];

    for (CeedInt i=0; i<nFace; i++) {
      ierr = GetRestrictionForDomain(ceed, dm, height, domainLabel, i+1, numP_Sur,
                                     numQ_Sur, qdatasizeSur, &restrictqSur[i],
                                     &restrictxSur[i], &restrictqdiSur[i]);
      CHKERRQ(ierr);
      // Create the CEED vectors that will be needed in Boundary setup
      CeedElemRestrictionGetNumElements(restrictqSur[i], &localNelemSur[i]);
      CeedVectorCreate(ceed, qdatasizeSur*localNelemSur[i]*NqptsSur,
                       &qdataSur[i]);
      // Create the operator that builds the quadrature data for the Boundary operator
      CeedOperatorCreate(ceed, qf_setupSur, NULL, NULL, &op_setupSur[i]);
      CeedOperatorSetField(op_setupSur[i], "dx", restrictxSur[i], basisxSur,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_setupSur[i], "weight", CEED_ELEMRESTRICTION_NONE,
                           basisxSur, CEED_VECTOR_NONE);
      CeedOperatorSetField(op_setupSur[i], "qdataSur", restrictqdiSur[i],
                           CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
      // Create Boundary operator
      CeedOperatorCreate(ceed, qf_applySur, NULL, NULL, &op_applySur[i]);
      CeedOperatorSetField(op_applySur[i], "q", restrictqSur[i], basisqSur,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_applySur[i], "qdataSur", restrictqdiSur[i],
                           CEED_BASIS_COLLOCATED, qdataSur[i]);
      CeedOperatorSetField(op_applySur[i], "x", restrictxSur[i], basisxSur,
                           xcorners);
      CeedOperatorSetField(op_applySur[i], "v", restrictqSur[i], basisqSur,
                           CEED_VECTOR_ACTIVE);
      // Apply CEED operator for Boundary setup
      CeedOperatorApply(op_setupSur[i], xcorners, qdataSur[i],
                        CEED_REQUEST_IMMEDIATE);
      // --Apply Sub-Operator for the Boundary
      CeedCompositeOperatorAddSub(*op_apply, op_applySur[i]);
    }
    CeedVectorDestroy(&xcorners);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SetupLibceed(Ceed ceed, CeedData ceed_data, DM dm, User user,
                            AppCtx app_ctx, problemData *problem, SimpleBC bc) {

  PetscErrorCode ierr;

  DM dmcoord;
  const PetscInt ncompq = 5;
  PetscInt localNelemVol;

  const CeedInt dim = problem->dim,
                ncompx = problem->dim,
                qdatasizeVol = problem->qdatasizeVol,
                numP = app_ctx->degree + 1,
                numQ = numP + app_ctx->q_extra;

  // CEED Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompq, numP, numQ, CEED_GAUSS,
                                  &ceed_data->basisq);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, numQ, CEED_GAUSS,
                                  &ceed_data->basisx);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, numP,
                                  CEED_GAUSS_LOBATTO, &ceed_data->basisxc);

  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  // CEED Restrictions
  ierr = GetRestrictionForDomain(ceed, dm, 0, 0, 0, numP, numQ,
                                 qdatasizeVol, &ceed_data->restrictq, &ceed_data->restrictx,
                                 &ceed_data->restrictqdi); CHKERRQ(ierr);

  // Create the CEED vectors that will be needed in setup
  CeedInt NqptsVol;
  CeedBasisGetNumQuadraturePoints(ceed_data->basisq, &NqptsVol);
  CeedElemRestrictionGetNumElements(ceed_data->restrictq, &localNelemVol);
  CeedVectorCreate(ceed, qdatasizeVol*localNelemVol*NqptsVol, &ceed_data->qdata);
  CeedElemRestrictionCreateVector(ceed_data->restrictq, &ceed_data->q0ceed, NULL);

  // Create the Q-function that builds the quadrature data for the NS operator
  CeedQFunctionCreateInterior(ceed, 1, problem->setupVol, problem->setupVol_loc,
                              &ceed_data->qf_setupVol);
  CeedQFunctionAddInput(ceed_data->qf_setupVol, "dx", ncompx*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(ceed_data->qf_setupVol, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(ceed_data->qf_setupVol, "qdata", qdatasizeVol,
                         CEED_EVAL_NONE);

  // Create the Q-function that sets the ICs of the operator
  CeedQFunctionCreateInterior(ceed, 1, problem->ics, problem->ics_loc,
                              &ceed_data->qf_ics);
  CeedQFunctionAddInput(ceed_data->qf_ics, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(ceed_data->qf_ics, "q0", ncompq, CEED_EVAL_NONE);

  if (problem->applyVol_rhs) { // Create the Q-function that defines the action of the RHS operator
    CeedQFunctionCreateInterior(ceed, 1, problem->applyVol_rhs,
                                problem->applyVol_rhs_loc, &ceed_data->qf_rhsVol);
    CeedQFunctionAddInput(ceed_data->qf_rhsVol, "q", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_rhsVol, "dq", ncompq*dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(ceed_data->qf_rhsVol, "qdata", qdatasizeVol,
                          CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_rhsVol, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_rhsVol, "v", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_rhsVol, "dv", ncompq*dim, CEED_EVAL_GRAD);
  }

  if (problem->applyVol_ifunction) { // Create the Q-function that defines the action of the IFunction
    CeedQFunctionCreateInterior(ceed, 1, problem->applyVol_ifunction,
                                problem->applyVol_ifunction_loc, &ceed_data->qf_ifunctionVol);
    CeedQFunctionAddInput(ceed_data->qf_ifunctionVol, "q", ncompq,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_ifunctionVol, "dq", ncompq*dim,
                          CEED_EVAL_GRAD);
    CeedQFunctionAddInput(ceed_data->qf_ifunctionVol, "qdot", ncompq,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_ifunctionVol, "qdata", qdatasizeVol,
                          CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_ifunctionVol, "x", ncompx,
                          CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_ifunctionVol, "v", ncompq,
                           CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_ifunctionVol, "dv", ncompq*dim,
                           CEED_EVAL_GRAD);
  }

  // Create the operator that builds the quadrature data for the NS operator
  CeedOperatorCreate(ceed, ceed_data->qf_setupVol, NULL, NULL,
                     &ceed_data->op_setupVol);
  CeedOperatorSetField(ceed_data->op_setupVol, "dx", ceed_data->restrictx,
                       ceed_data->basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(ceed_data->op_setupVol, "weight",
                       CEED_ELEMRESTRICTION_NONE,
                       ceed_data->basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(ceed_data->op_setupVol, "qdata", ceed_data->restrictqdi,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the operator that sets the ICs
  CeedOperatorCreate(ceed, ceed_data->qf_ics, NULL, NULL, &ceed_data->op_ics);
  CeedOperatorSetField(ceed_data->op_ics, "x", ceed_data->restrictx,
                       ceed_data->basisxc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(ceed_data->op_ics, "q0", ceed_data->restrictq,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedElemRestrictionCreateVector(ceed_data->restrictq, &user->qceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->restrictq, &user->qdotceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->restrictq, &user->gceed, NULL);

  if (ceed_data->qf_rhsVol) { // Create the RHS physics operator
    CeedOperator op;
    CeedOperatorCreate(ceed, ceed_data->qf_rhsVol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", ceed_data->restrictq, ceed_data->basisq,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", ceed_data->restrictq, ceed_data->basisq,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdata", ceed_data->restrictqdi, CEED_BASIS_COLLOCATED,
                         ceed_data->qdata);
    CeedOperatorSetField(op, "x", ceed_data->restrictx, ceed_data->basisx,
                         ceed_data->xcorners);
    CeedOperatorSetField(op, "v", ceed_data->restrictq, ceed_data->basisq,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", ceed_data->restrictq, ceed_data->basisq,
                         CEED_VECTOR_ACTIVE);
    user->op_rhs_vol = op;
  }

  if (ceed_data->qf_ifunctionVol) { // Create the IFunction operator
    CeedOperator op;
    CeedOperatorCreate(ceed, ceed_data->qf_ifunctionVol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", ceed_data->restrictq, ceed_data->basisq,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", ceed_data->restrictq, ceed_data->basisq,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdot", ceed_data->restrictq, ceed_data->basisq,
                         user->qdotceed);
    CeedOperatorSetField(op, "qdata", ceed_data->restrictqdi, CEED_BASIS_COLLOCATED,
                         ceed_data->qdata);
    CeedOperatorSetField(op, "x", ceed_data->restrictx, ceed_data->basisx,
                         ceed_data->xcorners);
    CeedOperatorSetField(op, "v", ceed_data->restrictq, ceed_data->basisq,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", ceed_data->restrictq, ceed_data->basisq,
                         CEED_VECTOR_ACTIVE);
    user->op_ifunction_vol = op;
  }

  // Set up CEED for the boundaries
  CeedInt height = 1,
          dimSur = dim - height,
          numP_Sur = app_ctx->degree + 1,  // todo: change it to q_extra_sur
          numQ_Sur = numP_Sur + app_ctx->q_extra_sur,
          NqptsSur;
  const CeedInt qdatasizeSur = problem->qdatasizeSur;

  // CEED bases for the boundaries
  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, ncompq, numP_Sur, numQ_Sur,
                                  CEED_GAUSS,
                                  &ceed_data->basisqSur);
  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, ncompx, 2, numQ_Sur, CEED_GAUSS,
                                  &ceed_data->basisxSur);
  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, ncompx, 2, numP_Sur,
                                  CEED_GAUSS_LOBATTO, &ceed_data->basisxcSur);
  CeedBasisGetNumQuadraturePoints(ceed_data->basisqSur, &NqptsSur);

  // Create the Q-function that builds the quadrature data for the Surface operator
  CeedQFunctionCreateInterior(ceed, 1, problem->setupSur, problem->setupSur_loc,
                              &ceed_data->qf_setupSur);
  CeedQFunctionAddInput(ceed_data->qf_setupSur, "dx", ncompx*dimSur,
                        CEED_EVAL_GRAD);
  CeedQFunctionAddInput(ceed_data->qf_setupSur, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(ceed_data->qf_setupSur, "qdataSur", qdatasizeSur,
                         CEED_EVAL_NONE);

  // Creat Q-Function for Boundaries
  if (problem->applySur) {
    CeedQFunctionCreateInterior(ceed, 1, problem->applySur,
                                problem->applySur_loc, &ceed_data->qf_applySur);
    CeedQFunctionAddInput(ceed_data->qf_applySur, "q", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(ceed_data->qf_applySur, "qdataSur", qdatasizeSur,
                          CEED_EVAL_NONE);
    CeedQFunctionAddInput(ceed_data->qf_applySur, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(ceed_data->qf_applySur, "v", ncompq, CEED_EVAL_INTERP);
  }

  // Create CEED Operator for the whole domain
  if (!user->phys->implicit)
    ierr = CreateOperatorForDomain(ceed, dm, bc, user->phys,
                                   user->op_rhs_vol,
                                   ceed_data->qf_applySur, ceed_data->qf_setupSur,
                                   height, numP_Sur, numQ_Sur, qdatasizeSur,
                                   NqptsSur, ceed_data->basisxSur, ceed_data->basisqSur,
                                   &user->op_rhs); CHKERRQ(ierr);
  if (user->phys->implicit)
    ierr = CreateOperatorForDomain(ceed, dm, bc, user->phys,
                                   user->op_ifunction_vol,
                                   ceed_data->qf_applySur, ceed_data->qf_setupSur,
                                   height, numP_Sur, numQ_Sur, qdatasizeSur,
                                   NqptsSur, ceed_data->basisxSur, ceed_data->basisqSur,
                                   &user->op_ifunction); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
