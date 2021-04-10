
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
