// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// libCEED setup for solid mechanics example using PETSc

#include "../elasticity.h"

#include "../qfunctions/common.h"            // Geometric factors
#include "../qfunctions/Linear.h"            // Linear elasticity
#include "../qfunctions/SS-NH.h"             // Hyperelasticity small strain
// Hyperelasticity finite strain
#include "../qfunctions/FSInitial-NH1.h"     // -- Initial config 1 w/ dXref_dxinit, Grad(u) storage
#include "../qfunctions/FSInitial-NH2.h"     // -- Initial config 2 w/ dXref_dxinit, Grad(u), Cinv, constant storage
#include "../qfunctions/FSCurrent-NH1.h"     // -- Current config 1 w/ dXref_dxinit, Grad(u) storage
#include "../qfunctions/FSCurrent-NH2.h"     // -- Current config 2 w/ dXref_dxcurr, tau, constant storage
#include "../qfunctions/constantForce.h"     // Constant forcing function
#include "../qfunctions/manufacturedForce.h" // Manufactured solution forcing
#include "../qfunctions/manufacturedTrue.h"  // Manufactured true solution

#if PETSC_VERSION_LT(3,14,0)
#  define DMPlexGetClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexGetClosureIndices(a,b,c,d,f,g,i)
#  define DMPlexRestoreClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexRestoreClosureIndices(a,b,c,d,f,g,i)
#endif

// -----------------------------------------------------------------------------
// Problem options
// -----------------------------------------------------------------------------
// Data specific to each problem option
problemData problemOptions[6] = {
  [ELAS_LINEAR] = {
    .qdatasize = 10, // For linear elasticity, 6 would be sufficient
    .setupgeo = SetupGeo,
    .apply = ElasLinearF,
    .jacob = ElasLineardF,
    .energy = ElasLinearEnergy,
    .diagnostic = ElasLinearDiagnostic,
    .setupgeofname = SetupGeo_loc,
    .applyfname = ElasLinearF_loc,
    .jacobfname = ElasLineardF_loc,
    .energyfname = ElasLinearEnergy_loc,
    .diagnosticfname = ElasLinearDiagnostic_loc,
    .qmode = CEED_GAUSS
  },
  [ELAS_SS_NH] = {
    .qdatasize = 10,
    .setupgeo = SetupGeo,
    .apply = ElasSSNHF,
    .jacob = ElasSSNHdF,
    .energy = ElasSSNHEnergy,
    .diagnostic = ElasSSNHDiagnostic,
    .setupgeofname = SetupGeo_loc,
    .applyfname = ElasSSNHF_loc,
    .jacobfname = ElasSSNHdF_loc,
    .energyfname = ElasSSNHEnergy_loc,
    .diagnosticfname = ElasSSNHDiagnostic_loc,
    .qmode = CEED_GAUSS
  },
  [ELAS_FSInitial_NH1] = {
    .qdatasize = 10,
    .setupgeo = SetupGeo,
    .apply = ElasFSInitialNH1F,
    .jacob = ElasFSInitialNH1dF,
    .energy = ElasFSInitialNH1Energy,
    .diagnostic = ElasFSInitialNH1Diagnostic,
    .setupgeofname = SetupGeo_loc,
    .applyfname = ElasFSInitialNH1F_loc,
    .jacobfname = ElasFSInitialNH1dF_loc,
    .energyfname = ElasFSInitialNH1Energy_loc,
    .diagnosticfname = ElasFSInitialNH1Diagnostic_loc,
    .qmode = CEED_GAUSS
  },
  [ELAS_FSInitial_NH2] = {
    .qdatasize = 10,
    .setupgeo = SetupGeo,
    .apply = ElasFSInitialNH2F,
    .jacob = ElasFSInitialNH2dF,
    .energy = ElasFSInitialNH2Energy,
    .diagnostic = ElasFSInitialNH2Diagnostic,
    .setupgeofname = SetupGeo_loc,
    .applyfname = ElasFSInitialNH2F_loc,
    .jacobfname = ElasFSInitialNH2dF_loc,
    .energyfname = ElasFSInitialNH2Energy_loc,
    .diagnosticfname = ElasFSInitialNH2Diagnostic_loc,
    .qmode = CEED_GAUSS
  },
  [ELAS_FSCurrent_NH1] = {
    .qdatasize = 10,
    .setupgeo = SetupGeo,
    .apply = ElasFSCurrentNH1F,
    .jacob = ElasFSCurrentNH1dF,
    .energy = ElasFSCurrentNH1Energy,
    .diagnostic = ElasFSCurrentNH1Diagnostic,
    .setupgeofname = SetupGeo_loc,
    .applyfname = ElasFSCurrentNH1F_loc,
    .jacobfname = ElasFSCurrentNH1dF_loc,
    .energyfname = ElasFSCurrentNH1Energy_loc,
    .diagnosticfname = ElasFSCurrentNH1Diagnostic_loc,
    .qmode = CEED_GAUSS
  },
  [ELAS_FSCurrent_NH2] = {
    .qdatasize = 10,
    .setupgeo = SetupGeo,
    .apply = ElasFSCurrentNH2F,
    .jacob = ElasFSCurrentNH2dF,
    .energy = ElasFSCurrentNH2Energy,
    .diagnostic = ElasFSCurrentNH2Diagnostic,
    .setupgeofname = SetupGeo_loc,
    .applyfname = ElasFSCurrentNH2F_loc,
    .jacobfname = ElasFSCurrentNH2dF_loc,
    .energyfname = ElasFSCurrentNH2Energy_loc,
    .diagnosticfname = ElasFSCurrentNH2Diagnostic_loc,
    .qmode = CEED_GAUSS
  }
};

// Forcing function data
forcingData forcingOptions[3] = {
  [FORCE_NONE] = {
    .setupforcing = NULL,
    .setupforcingfname = NULL
  },
  [FORCE_CONST] = {
    .setupforcing = SetupConstantForce,
    .setupforcingfname = SetupConstantForce_loc
  },
  [FORCE_MMS] = {
    .setupforcing = SetupMMSForce,
    .setupforcingfname = SetupMMSForce_loc
  }
};

// -----------------------------------------------------------------------------
// libCEED Functions
// -----------------------------------------------------------------------------
// Destroy libCEED objects
PetscErrorCode CeedDataDestroy(CeedInt level, CeedData data) {
  PetscErrorCode ierr;

  // Vectors
  CeedVectorDestroy(&data->qdata);
  CeedVectorDestroy(&data->qdataDiagnostic);
  CeedVectorDestroy(&data->gradu);
  CeedVectorDestroy(&data->Cinv);
  CeedVectorDestroy(&data->lamlogJ);
  CeedVectorDestroy(&data->dXdx);
  CeedVectorDestroy(&data->tau);
  CeedVectorDestroy(&data->Cc1);
  CeedVectorDestroy(&data->xceed);
  CeedVectorDestroy(&data->yceed);
  CeedVectorDestroy(&data->truesoln);

  // Restrictions
  CeedElemRestrictionDestroy(&data->Erestrictu);
  CeedElemRestrictionDestroy(&data->Erestrictx);
  CeedElemRestrictionDestroy(&data->ErestrictGradui);
  CeedElemRestrictionDestroy(&data->ErestrictCinv);
  CeedElemRestrictionDestroy(&data->ErestrictlamlogJ);
  CeedElemRestrictionDestroy(&data->ErestrictdXdx);
  CeedElemRestrictionDestroy(&data->Erestricttau);
  CeedElemRestrictionDestroy(&data->ErestrictCc1);
  CeedElemRestrictionDestroy(&data->Erestrictqdi);
  CeedElemRestrictionDestroy(&data->ErestrictEnergy);
  CeedElemRestrictionDestroy(&data->ErestrictDiagnostic);
  CeedElemRestrictionDestroy(&data->ErestrictqdDiagnostici);

  // Bases
  CeedBasisDestroy(&data->basisx);
  CeedBasisDestroy(&data->basisu);
  CeedBasisDestroy(&data->basisEnergy);
  CeedBasisDestroy(&data->basisDiagnostic);

  // QFunctions
  CeedQFunctionDestroy(&data->qfJacob);
  CeedQFunctionDestroy(&data->qfApply);
  CeedQFunctionDestroy(&data->qfEnergy);
  CeedQFunctionDestroy(&data->qfDiagnostic);

  // Operators
  CeedOperatorDestroy(&data->opJacob);
  CeedOperatorDestroy(&data->opApply);
  CeedOperatorDestroy(&data->opEnergy);
  CeedOperatorDestroy(&data->opDiagnostic);

  // Restriction and Prolongation data
  CeedBasisDestroy(&data->basisCtoF);
  CeedOperatorDestroy(&data->opProlong);
  CeedOperatorDestroy(&data->opRestrict);

  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// Utility function - essential BC dofs are encoded in closure indices as -(i+1)
PetscInt Involute(PetscInt i) {
  return i >= 0 ? i : -(i + 1);
};

// Utility function to create local CEED restriction from DMPlex
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt height, DMLabel domainLabel, CeedInt value,
    CeedElemRestriction *Erestrict) {

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
  for (PetscInt f = 0; f < nfields; f++) {
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
  for (p = 0, eoffset = 0; p < Nelem; p++) {
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
    for (PetscInt i = 0; i < nnodes; i++) {
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
      for (PetscInt f = 0; f < nfields; f++) {
        for (PetscInt j = 0; j < ncomp[f]; j++) {
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
};

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
  if (restrictq) {
    ierr = CreateRestrictionFromPlex(ceed, dm, P, height, domainLabel, value,
                                     restrictq); CHKERRQ(ierr);
  }
  if (restrictx) {
    ierr = CreateRestrictionFromPlex(ceed, dmcoord, 2, height, domainLabel,
                                     value, restrictx); CHKERRQ(ierr);
  }
  if (restrictqdi) {
    CeedElemRestrictionGetNumElements(*restrictq, &localNelem);
    CeedElemRestrictionCreateStrided(ceed, localNelem, Qdim,
                                     qdatasize, qdatasize*localNelem*Qdim,
                                     CEED_STRIDES_BACKEND, restrictqdi);
  }

  PetscFunctionReturn(0);
};

// Set up libCEED on the fine grid for a given degree
PetscErrorCode SetupLibceedFineLevel(DM dm, DM dmEnergy, DM dmDiagnostic,
                                     Ceed ceed, AppCtx appCtx,
                                     CeedQFunctionContext physCtx,
                                     CeedData *data, PetscInt fineLevel,
                                     PetscInt ncompu, PetscInt Ugsz,
                                     PetscInt Ulocsz, CeedVector forceCeed,
                                     CeedVector neumannCeed) {
  int           ierr;
  CeedInt       P = appCtx->levelDegrees[fineLevel] + 1;
  CeedInt       Q = appCtx->levelDegrees[fineLevel] + 1 + appCtx->qextra;
  CeedInt       dim, ncompx, ncompe = 1, ncompd = 5;
  CeedInt       nqpts;
  CeedInt       qdatasize = problemOptions[appCtx->problemChoice].qdatasize;
  problemType   problemChoice = appCtx->problemChoice;
  forcingType   forcingChoice = appCtx->forcingChoice;
  DM            dmcoord;
  Vec           coords;
  PetscInt      cStart, cEnd, nelem;
  const PetscScalar *coordArray;
  CeedVector    xcoord;
  CeedQFunction qfSetupGeo, qfApply, qfJacob, qfEnergy, qfDiagnostic;
  CeedOperator  opSetupGeo, opApply, opJacob, opEnergy, opDiagnostic;

  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ncompx = dim;

  // ---------------------------------------------------------------------------
  // libCEED restrictions
  // ---------------------------------------------------------------------------
  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  // -- Coordinate restriction
  ierr = CreateRestrictionFromPlex(ceed, dmcoord, 2, 0, 0, 0,
                                   &(data[fineLevel]->Erestrictx));
  CHKERRQ(ierr);
  // -- Solution restriction
  ierr = CreateRestrictionFromPlex(ceed, dm, P, 0, 0, 0,
                                   &data[fineLevel]->Erestrictu);
  CHKERRQ(ierr);
  // -- Energy restriction
  ierr = CreateRestrictionFromPlex(ceed, dmEnergy, P, 0, 0, 0,
                                   &data[fineLevel]->ErestrictEnergy);
  CHKERRQ(ierr);
  // -- Pressure restriction
  ierr = CreateRestrictionFromPlex(ceed, dmDiagnostic, P, 0, 0, 0,
                                   &data[fineLevel]->ErestrictDiagnostic);
  CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;

  // -- Geometric data restriction
  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, qdatasize,
                                   qdatasize*nelem*Q*Q*Q,
                                   CEED_STRIDES_BACKEND,
                                   &data[fineLevel]->Erestrictqdi);
  // -- State vector gradient restriction
  switch (problemChoice) {
  // ---- Linear Elasticity
  case ELAS_LINEAR:
    break;
  // ---- Hyperelasticity at small strain
  case ELAS_SS_NH:
    CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, dim*ncompu,
                                     dim*ncompu*nelem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fineLevel]->ErestrictGradui);
    break;
  // ---- Hyperelasticity at finite strain
  case ELAS_FSInitial_NH1:
    // ------ storage: dXdx, Grad(u)
    CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, dim*ncompu,
                                     dim*ncompu*nelem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fineLevel]->ErestrictGradui);
    break;
  case ELAS_FSInitial_NH2:
    // ------ storage: dXdx, Grad(u), Cinv, lamda*logJ
    CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, dim*ncompu,
                                     dim*ncompu*nelem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fineLevel]->ErestrictGradui);
    CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, (dim+1)*ncompu/2,
                                     (dim+1)*ncompu*nelem*Q*Q*Q/2,
                                     CEED_STRIDES_BACKEND,
                                     &data[fineLevel]->ErestrictCinv);
    CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, 1,
                                     1*nelem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fineLevel]->ErestrictlamlogJ);
    break;
  case ELAS_FSCurrent_NH1:
    // ------ storage: dXdx, Grad(u)
    CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, dim*ncompu,
                                     dim*ncompu*nelem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fineLevel]->ErestrictGradui);
    break;
  case ELAS_FSCurrent_NH2:
    // ------ storage: dXdxcur, tau, mu - lamda*logJ
    CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, dim*ncompu,
                                     dim*ncompu*nelem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fineLevel]->ErestrictdXdx);

    CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, (dim+1)*ncompu/2,
                                     (dim+1)*ncompu*nelem*Q*Q*Q/2,
                                     CEED_STRIDES_BACKEND,
                                     &data[fineLevel]->Erestricttau);
    CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, 1,
                                     1*nelem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fineLevel]->ErestrictCc1);
    break;
  }
  // -- Geometric data restriction
  CeedElemRestrictionCreateStrided(ceed, nelem, P*P*P, qdatasize,
                                   qdatasize*nelem*P*P*P,
                                   CEED_STRIDES_BACKEND,
                                   &data[fineLevel]->ErestrictqdDiagnostici);

  // ---------------------------------------------------------------------------
  // Element coordinates
  // ---------------------------------------------------------------------------
  ierr = DMGetCoordinatesLocal(dm, &coords); CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords, &coordArray); CHKERRQ(ierr);

  CeedElemRestrictionCreateVector(data[fineLevel]->Erestrictx, &xcoord, NULL);
  CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_COPY_VALUES,
                     (PetscScalar *)coordArray);
  ierr = VecRestoreArrayRead(coords, &coordArray); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // libCEED bases
  // ---------------------------------------------------------------------------
  // -- Solution basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompu, P, Q,
                                  problemOptions[problemChoice].qmode,
                                  &data[fineLevel]->basisu);
  // -- Coordinate basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, Q,
                                  problemOptions[problemChoice].qmode,
                                  &data[fineLevel]->basisx);
  // -- Energy basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompe, P, Q,
                                  problemOptions[problemChoice].qmode,
                                  &data[fineLevel]->basisEnergy);
  // -- Diagnostic output basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompu, P, P, CEED_GAUSS_LOBATTO,
                                  &data[fineLevel]->basisDiagnostic);

  // ---------------------------------------------------------------------------
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  CeedBasisGetNumQuadraturePoints(data[fineLevel]->basisu, &nqpts);
  // -- Geometric data vector
  CeedVectorCreate(ceed, qdatasize*nelem*nqpts, &data[fineLevel]->qdata);
  // -- Collocated geometric data vector
  CeedVectorCreate(ceed, qdatasize*nelem*P*P*P,
                   &data[fineLevel]->qdataDiagnostic);
  // -- State gradient vector
  switch (problemChoice) {
  case ELAS_LINEAR:
    break;
  case ELAS_SS_NH:
    CeedVectorCreate(ceed, dim*ncompu*nelem*nqpts, &data[fineLevel]->gradu);
    break;
  case ELAS_FSInitial_NH1:
    CeedVectorCreate(ceed, dim*ncompu*nelem*nqpts, &data[fineLevel]->gradu);
    break;
  case ELAS_FSInitial_NH2:
    CeedVectorCreate(ceed, dim*ncompu*nelem*nqpts, &data[fineLevel]->gradu);
    CeedVectorCreate(ceed, (dim+1)*ncompu*nelem*nqpts/2, &data[fineLevel]->Cinv);
    CeedVectorCreate(ceed, 1*nelem*nqpts, &data[fineLevel]->lamlogJ);
    break;
  case ELAS_FSCurrent_NH1:
    CeedVectorCreate(ceed, dim*ncompu*nelem*nqpts, &data[fineLevel]->gradu);
    break;
  case ELAS_FSCurrent_NH2:
    CeedVectorCreate(ceed, dim*ncompu*nelem*nqpts, &data[fineLevel]->dXdx);
    CeedVectorCreate(ceed, (dim+1)*ncompu*nelem*nqpts/2, &data[fineLevel]->tau);
    CeedVectorCreate(ceed, 1*nelem*nqpts, &data[fineLevel]->Cc1);
    break;
  }
  // -- Operator action variables
  CeedVectorCreate(ceed, Ulocsz, &data[fineLevel]->xceed);
  CeedVectorCreate(ceed, Ulocsz, &data[fineLevel]->yceed);

  // ---------------------------------------------------------------------------
  // Geometric factor computation
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the quadrature data
  //   qdata returns dXdx_i,j and w * det.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problemOptions[problemChoice].setupgeo,
                              problemOptions[problemChoice].setupgeofname,
                              &qfSetupGeo);
  CeedQFunctionAddInput(qfSetupGeo, "dx", ncompx*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qfSetupGeo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qfSetupGeo, "qdata", qdatasize, CEED_EVAL_NONE);

  // -- Operator
  CeedOperatorCreate(ceed, qfSetupGeo, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &opSetupGeo);
  CeedOperatorSetField(opSetupGeo, "dx", data[fineLevel]->Erestrictx,
                       data[fineLevel]->basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opSetupGeo, "weight", CEED_ELEMRESTRICTION_NONE,
                       data[fineLevel]->basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(opSetupGeo, "qdata", data[fineLevel]->Erestrictqdi,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // -- Compute the quadrature data
  CeedOperatorApply(opSetupGeo, xcoord, data[fineLevel]->qdata,
                    CEED_REQUEST_IMMEDIATE);

  // -- Cleanup
  CeedQFunctionDestroy(&qfSetupGeo);
  CeedOperatorDestroy(&opSetupGeo);

  // ---------------------------------------------------------------------------
  // Local residual evaluator
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the residual of the
  //   non-linear PDE.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problemOptions[problemChoice].apply,
                              problemOptions[problemChoice].applyfname,
                              &qfApply);
  CeedQFunctionAddInput(qfApply, "du", ncompu*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qfApply, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qfApply, "dv", ncompu*dim, CEED_EVAL_GRAD);
  switch (problemChoice) {
  case ELAS_LINEAR:
    break;
  case ELAS_SS_NH:
    CeedQFunctionAddOutput(qfApply, "gradu", ncompu*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSInitial_NH1:
    CeedQFunctionAddOutput(qfApply, "gradu", ncompu*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSInitial_NH2:
    CeedQFunctionAddOutput(qfApply, "gradu", ncompu*dim, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qfApply, "Cinv", ncompu*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qfApply, "lamlogJ", 1, CEED_EVAL_NONE);
    break;
  case ELAS_FSCurrent_NH1:
    CeedQFunctionAddOutput(qfApply, "gradu", ncompu*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSCurrent_NH2:
    CeedQFunctionAddOutput(qfApply, "dXdx", ncompu*dim, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qfApply, "tau", ncompu*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qfApply, "Cc1", 1, CEED_EVAL_NONE);
    break;
  }
  CeedQFunctionSetContext(qfApply, physCtx);

  // -- Operator
  CeedOperatorCreate(ceed, qfApply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &opApply);
  CeedOperatorSetField(opApply, "du", data[fineLevel]->Erestrictu,
                       data[fineLevel]->basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opApply, "qdata", data[fineLevel]->Erestrictqdi,
                       CEED_BASIS_COLLOCATED, data[fineLevel]->qdata);
  CeedOperatorSetField(opApply, "dv", data[fineLevel]->Erestrictu,
                       data[fineLevel]->basisu, CEED_VECTOR_ACTIVE);
  switch (problemChoice) {
  case ELAS_LINEAR:
    break;
  case ELAS_SS_NH:
    CeedOperatorSetField(opApply, "gradu", data[fineLevel]->ErestrictGradui,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->gradu);
    break;
  case ELAS_FSInitial_NH1:
    CeedOperatorSetField(opApply, "gradu", data[fineLevel]->ErestrictGradui,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->gradu);
    break;
  case ELAS_FSInitial_NH2:
    CeedOperatorSetField(opApply, "gradu", data[fineLevel]->ErestrictGradui,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->gradu);
    CeedOperatorSetField(opApply, "Cinv", data[fineLevel]->ErestrictCinv,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->Cinv);
    CeedOperatorSetField(opApply, "lamlogJ", data[fineLevel]->ErestrictlamlogJ,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->lamlogJ);
    break;
  case ELAS_FSCurrent_NH1:
    CeedOperatorSetField(opApply, "gradu", data[fineLevel]->ErestrictGradui,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->gradu);
    break;
  case ELAS_FSCurrent_NH2:
    CeedOperatorSetField(opApply, "dXdx", data[fineLevel]->ErestrictdXdx,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->dXdx);
    CeedOperatorSetField(opApply, "tau", data[fineLevel]->Erestricttau,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->tau);
    CeedOperatorSetField(opApply, "Cc1", data[fineLevel]->ErestrictCc1,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->Cc1);
    break;
  }
  // -- Save libCEED data
  data[fineLevel]->qfApply = qfApply;
  data[fineLevel]->opApply = opApply;

  // ---------------------------------------------------------------------------
  // Jacobian evaluator
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the action of the
  //   Jacobian for each linear solve.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problemOptions[problemChoice].jacob,
                              problemOptions[problemChoice].jacobfname,
                              &qfJacob);
  CeedQFunctionAddInput(qfJacob, "deltadu", ncompu*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qfJacob, "qdata", qdatasize, CEED_EVAL_NONE);
  switch (problemChoice) {
  case ELAS_LINEAR:
    break;
  case ELAS_SS_NH:
    CeedQFunctionAddInput(qfJacob, "gradu", ncompu*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSInitial_NH1:
    CeedQFunctionAddInput(qfJacob, "gradu", ncompu*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSInitial_NH2:
    CeedQFunctionAddInput(qfJacob, "gradu", ncompu*dim, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qfJacob, "Cinv", ncompu*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qfJacob, "lamlogJ", 1, CEED_EVAL_NONE);
    break;
  case ELAS_FSCurrent_NH1:
    CeedQFunctionAddInput(qfJacob, "gradu", ncompu*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSCurrent_NH2:
    CeedQFunctionAddInput(qfJacob, "dXdx", ncompu*dim, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qfJacob, "tau", ncompu*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qfJacob, "Cc1", 1, CEED_EVAL_NONE);
    break;
  }
  CeedQFunctionAddOutput(qfJacob, "deltadv", ncompu*dim, CEED_EVAL_GRAD);
  CeedQFunctionSetContext(qfJacob, physCtx);

  // -- Operator
  CeedOperatorCreate(ceed, qfJacob, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &opJacob);
  CeedOperatorSetField(opJacob, "deltadu", data[fineLevel]->Erestrictu,
                       data[fineLevel]->basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opJacob, "qdata", data[fineLevel]->Erestrictqdi,
                       CEED_BASIS_COLLOCATED, data[fineLevel]->qdata);
  CeedOperatorSetField(opJacob, "deltadv", data[fineLevel]->Erestrictu,
                       data[fineLevel]->basisu, CEED_VECTOR_ACTIVE);
  switch (problemChoice) {
  case ELAS_LINEAR:
    break;
  case ELAS_SS_NH:
    CeedOperatorSetField(opJacob, "gradu", data[fineLevel]->ErestrictGradui,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->gradu);
    break;
  case ELAS_FSInitial_NH1:
    CeedOperatorSetField(opJacob, "gradu", data[fineLevel]->ErestrictGradui,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->gradu);
    break;
  case ELAS_FSInitial_NH2:
    CeedOperatorSetField(opJacob, "gradu", data[fineLevel]->ErestrictGradui,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->gradu);
    CeedOperatorSetField(opJacob, "Cinv", data[fineLevel]->ErestrictCinv,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->Cinv);
    CeedOperatorSetField(opJacob, "lamlogJ", data[fineLevel]->ErestrictlamlogJ,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->lamlogJ);
    break;
  case ELAS_FSCurrent_NH1:
    CeedOperatorSetField(opJacob, "gradu", data[fineLevel]->ErestrictGradui,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->gradu);
    break;
  case ELAS_FSCurrent_NH2:
    CeedOperatorSetField(opJacob, "dXdx", data[fineLevel]->ErestrictdXdx,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->dXdx);
    CeedOperatorSetField(opJacob, "tau", data[fineLevel]->Erestricttau,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->tau);
    CeedOperatorSetField(opJacob, "Cc1", data[fineLevel]->ErestrictCc1,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->Cc1);
    break;
  }
  // -- Save libCEED data
  data[fineLevel]->qfJacob = qfJacob;
  data[fineLevel]->opJacob = opJacob;

  // ---------------------------------------------------------------------------
  // Traction boundary conditions, if needed
  // ---------------------------------------------------------------------------
  if (appCtx->bcTractionCount > 0) {
    // -- Setup
    DMLabel domainLabel;
    ierr = DMGetLabel(dm, "Face Sets", &domainLabel); CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

    // -- Basis
    CeedInt height = 1;
    CeedBasis basisuFace, basisxFace;
    CeedBasisCreateTensorH1Lagrange(ceed, dim - height, ncompu, P, Q,
                                    problemOptions[problemChoice].qmode,
                                    &basisuFace);
    CeedBasisCreateTensorH1Lagrange(ceed, dim - height, ncompx, 2, Q,
                                    problemOptions[problemChoice].qmode,
                                    &basisxFace);

    // -- QFunction
    CeedQFunction qfTraction;
    CeedQFunctionContext tractionCtx;
    CeedQFunctionCreateInterior(ceed, 1, SetupTractionBCs, SetupTractionBCs_loc,
                                &qfTraction);
    CeedQFunctionContextCreate(ceed, &tractionCtx);
    CeedQFunctionSetContext(qfTraction, tractionCtx);
    CeedQFunctionAddInput(qfTraction, "dx", ncompx*(ncompx - height),
                          CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qfTraction, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddOutput(qfTraction, "v", ncompu, CEED_EVAL_INTERP);

    // -- Compute contribution on each boundary face
    for (CeedInt i = 0; i < appCtx->bcTractionCount; i++) {
      CeedElemRestriction ErestrictuFace, ErestrictxFace;
      CeedOperator opTraction;
      CeedQFunctionContextSetData(tractionCtx, CEED_MEM_HOST, CEED_USE_POINTER,
                                  3 * sizeof(CeedScalar),
                                  appCtx->bcTractionVector[i]);

      // Setup restriction
      ierr = GetRestrictionForDomain(ceed, dm, height, domainLabel,
                                     appCtx->bcTractionFaces[i], P, Q,
                                     0, &ErestrictuFace, &ErestrictxFace, NULL);
      CHKERRQ(ierr);

      // ---- Create boundary Operator
      CeedOperatorCreate(ceed, qfTraction, NULL, NULL, &opTraction);
      CeedOperatorSetField(opTraction, "dx", ErestrictxFace, basisxFace,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(opTraction, "weight", CEED_ELEMRESTRICTION_NONE,
                           basisxFace, CEED_VECTOR_NONE);
      CeedOperatorSetField(opTraction, "v", ErestrictuFace,
                           basisuFace, CEED_VECTOR_ACTIVE);

      // ---- Compute traction on face
      CeedOperatorApplyAdd(opTraction, xcoord, neumannCeed,
                           CEED_REQUEST_IMMEDIATE);

      // ---- Cleanup
      CeedElemRestrictionDestroy(&ErestrictuFace);
      CeedElemRestrictionDestroy(&ErestrictxFace);
      CeedOperatorDestroy(&opTraction);
    }

    // -- Cleanup
    CeedBasisDestroy(&basisuFace);
    CeedBasisDestroy(&basisxFace);
    CeedQFunctionDestroy(&qfTraction);
    CeedQFunctionContextDestroy(&tractionCtx);
  }

  // ---------------------------------------------------------------------------
  // Forcing term, if needed
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the forcing term (RHS)
  //   for the non-linear PDE.
  // ---------------------------------------------------------------------------
  if (forcingChoice != FORCE_NONE) {
    CeedQFunction qfSetupForce;
    CeedOperator opSetupForce;

    // -- QFunction
    CeedQFunctionCreateInterior(ceed, 1,
                                forcingOptions[forcingChoice].setupforcing,
                                forcingOptions[forcingChoice].setupforcingfname,
                                &qfSetupForce);
    CeedQFunctionAddInput(qfSetupForce, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qfSetupForce, "qdata", qdatasize, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qfSetupForce, "force", ncompu, CEED_EVAL_INTERP);
    if (forcingChoice == FORCE_MMS) {
      CeedQFunctionSetContext(qfSetupForce, physCtx);
    } else {
      CeedQFunctionContext ctxForcing;
      CeedQFunctionContextCreate(ceed, &ctxForcing);
      CeedQFunctionContextSetData(ctxForcing, CEED_MEM_HOST, CEED_USE_POINTER,
                                  sizeof(*appCtx->forcingVector),
                                  appCtx->forcingVector);
      CeedQFunctionSetContext(qfSetupForce, ctxForcing);
      CeedQFunctionContextDestroy(&ctxForcing);
    }

    // -- Operator
    CeedOperatorCreate(ceed, qfSetupForce, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &opSetupForce);
    CeedOperatorSetField(opSetupForce, "x", data[fineLevel]->Erestrictx,
                         data[fineLevel]->basisx, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(opSetupForce, "qdata", data[fineLevel]->Erestrictqdi,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->qdata);
    CeedOperatorSetField(opSetupForce, "force", data[fineLevel]->Erestrictu,
                         data[fineLevel]->basisu, CEED_VECTOR_ACTIVE);

    // -- Compute forcing term
    CeedOperatorApply(opSetupForce, xcoord, forceCeed, CEED_REQUEST_IMMEDIATE);

    // -- Cleanup
    CeedQFunctionDestroy(&qfSetupForce);
    CeedOperatorDestroy(&opSetupForce);
  }

  // ---------------------------------------------------------------------------
  // True solution, for MMS
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the true solution at
  //   the mesh nodes for validation with the manufactured solution.
  // ---------------------------------------------------------------------------
  if (forcingChoice == FORCE_MMS) {
    CeedScalar *truearray;
    const CeedScalar *multarray;
    CeedVector multvec;
    CeedBasis basisxtrue;
    CeedQFunction qfTrue;
    CeedOperator opTrue;

    // -- Solution vector
    CeedVectorCreate(ceed, Ulocsz, &(data[fineLevel]->truesoln));

    // -- Basis
    CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, P, CEED_GAUSS_LOBATTO,
                                    &basisxtrue);

    // QFunction
    CeedQFunctionCreateInterior(ceed, 1, MMSTrueSoln, MMSTrueSoln_loc,
                                &qfTrue);
    CeedQFunctionAddInput(qfTrue, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qfTrue, "true_soln", ncompu, CEED_EVAL_NONE);

    // Operator
    CeedOperatorCreate(ceed, qfTrue, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                       &opTrue);
    CeedOperatorSetField(opTrue, "x", data[fineLevel]->Erestrictx, basisxtrue,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(opTrue, "true_soln", data[fineLevel]->Erestrictu,
                         CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

    // -- Compute true solution
    CeedOperatorApply(opTrue, xcoord, data[fineLevel]->truesoln,
                      CEED_REQUEST_IMMEDIATE);

    // -- Multiplicity calculation
    CeedElemRestrictionCreateVector(data[fineLevel]->Erestrictu, &multvec, NULL);
    CeedVectorSetValue(multvec, 0.);
    CeedElemRestrictionGetMultiplicity(data[fineLevel]->Erestrictu, multvec);

    // -- Multiplicity correction
    CeedVectorGetArray(data[fineLevel]->truesoln, CEED_MEM_HOST, &truearray);
    CeedVectorGetArrayRead(multvec, CEED_MEM_HOST, &multarray);
    for (CeedInt i = 0; i < Ulocsz; i++)
      truearray[i] /= multarray[i];
    CeedVectorRestoreArray(data[fineLevel]->truesoln, &truearray);
    CeedVectorRestoreArrayRead(multvec, &multarray);

    // -- Cleanup
    CeedVectorDestroy(&multvec);
    CeedBasisDestroy(&basisxtrue);
    CeedQFunctionDestroy(&qfTrue);
    CeedOperatorDestroy(&opTrue);
  }

  // ---------------------------------------------------------------------------
  // Local energy computation
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the strain energy
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problemOptions[problemChoice].energy,
                              problemOptions[problemChoice].energyfname,
                              &qfEnergy);
  CeedQFunctionAddInput(qfEnergy, "du", ncompu*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qfEnergy, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qfEnergy, "energy", ncompe, CEED_EVAL_INTERP);
  CeedQFunctionSetContext(qfEnergy, physCtx);

  // -- Operator
  CeedOperatorCreate(ceed, qfEnergy, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &opEnergy);
  CeedOperatorSetField(opEnergy, "du", data[fineLevel]->Erestrictu,
                       data[fineLevel]->basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opEnergy, "qdata", data[fineLevel]->Erestrictqdi,
                       CEED_BASIS_COLLOCATED, data[fineLevel]->qdata);
  CeedOperatorSetField(opEnergy, "energy", data[fineLevel]->ErestrictEnergy,
                       data[fineLevel]->basisEnergy, CEED_VECTOR_ACTIVE);
  // -- Save libCEED data
  data[fineLevel]->qfEnergy = qfEnergy;
  data[fineLevel]->opEnergy = opEnergy;

  // ---------------------------------------------------------------------------
  // Diagnostic value computation
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes nodal diagnostic quantities
  // ---------------------------------------------------------------------------
  // Geometric factors
  // -- Coordinate basis
  CeedBasis basisx;
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, Q, CEED_GAUSS_LOBATTO,
                                  &basisx);
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problemOptions[problemChoice].setupgeo,
                              problemOptions[problemChoice].setupgeofname,
                              &qfSetupGeo);
  CeedQFunctionAddInput(qfSetupGeo, "dx", ncompx*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qfSetupGeo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qfSetupGeo, "qdata", qdatasize, CEED_EVAL_NONE);

  // -- Operator
  CeedOperatorCreate(ceed, qfSetupGeo, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &opSetupGeo);
  CeedOperatorSetField(opSetupGeo, "dx", data[fineLevel]->Erestrictx,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opSetupGeo, "weight", CEED_ELEMRESTRICTION_NONE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(opSetupGeo, "qdata",
                       data[fineLevel]->ErestrictqdDiagnostici,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // -- Compute the quadrature data
  CeedOperatorApply(opSetupGeo, xcoord, data[fineLevel]->qdataDiagnostic,
                    CEED_REQUEST_IMMEDIATE);

  // -- Cleanup
  CeedBasisDestroy(&basisx);
  CeedQFunctionDestroy(&qfSetupGeo);
  CeedOperatorDestroy(&opSetupGeo);

  // Diagnostic quantities
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problemOptions[problemChoice].diagnostic,
                              problemOptions[problemChoice].diagnosticfname,
                              &qfDiagnostic);
  CeedQFunctionAddInput(qfDiagnostic, "u", ncompu, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qfDiagnostic, "du", ncompu*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qfDiagnostic, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qfDiagnostic, "diagnostic", ncompu + ncompd,
                         CEED_EVAL_NONE);
  CeedQFunctionSetContext(qfDiagnostic, physCtx);

  // -- Operator
  CeedOperatorCreate(ceed, qfDiagnostic, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &opDiagnostic);
  CeedOperatorSetField(opDiagnostic, "u", data[fineLevel]->Erestrictu,
                       data[fineLevel]->basisDiagnostic, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opDiagnostic, "du", data[fineLevel]->Erestrictu,
                       data[fineLevel]->basisDiagnostic, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opDiagnostic, "qdata",
                       data[fineLevel]->ErestrictqdDiagnostici,
                       CEED_BASIS_COLLOCATED, data[fineLevel]->qdataDiagnostic);
  CeedOperatorSetField(opDiagnostic, "diagnostic",
                       data[fineLevel]->ErestrictDiagnostic,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  // -- Save libCEED data
  data[fineLevel]->qfDiagnostic = qfDiagnostic;
  data[fineLevel]->opDiagnostic = opDiagnostic;

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------
  CeedVectorDestroy(&xcoord);

  PetscFunctionReturn(0);
};

// Set up libCEED multigrid level for a given degree
//   Prolongation and Restriction are between level and level+1
PetscErrorCode SetupLibceedLevel(DM dm, Ceed ceed, AppCtx appCtx,
                                 CeedData *data, PetscInt level,
                                 PetscInt ncompu, PetscInt Ugsz,
                                 PetscInt Ulocsz, CeedVector fineMult) {
  PetscErrorCode ierr;
  CeedInt        fineLevel = appCtx->numLevels - 1;
  CeedInt        P = appCtx->levelDegrees[level] + 1;
  CeedInt        Q = appCtx->levelDegrees[fineLevel] + 1 + appCtx->qextra;
  CeedInt        dim;
  CeedOperator   opJacob, opProlong, opRestrict;

  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // libCEED restrictions
  // ---------------------------------------------------------------------------
  // -- Solution restriction
  ierr = CreateRestrictionFromPlex(ceed, dm, P, 0, 0, 0,
                                   &data[level]->Erestrictu);
  CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // libCEED bases
  // ---------------------------------------------------------------------------
  // -- Solution basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompu, P, Q,
                                  problemOptions[appCtx->problemChoice].qmode,
                                  &data[level]->basisu);

  // ---------------------------------------------------------------------------
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  CeedVectorCreate(ceed, Ulocsz, &data[level]->xceed);
  CeedVectorCreate(ceed, Ulocsz, &data[level]->yceed);

  // ---------------------------------------------------------------------------
  // Coarse Grid, Prolongation, and Restriction Operators
  // ---------------------------------------------------------------------------
  // Create the Operators that compute the prolongation and
  //   restriction between the p-multigrid levels and the coarse grid eval.
  // ---------------------------------------------------------------------------
  CeedOperatorMultigridLevelCreate(data[level+1]->opJacob, fineMult,
                                   data[level]->Erestrictu, data[level]->basisu,
                                   &opJacob, &opProlong, &opRestrict);

  // -- Save libCEED data
  data[level]->opJacob = opJacob;
  data[level+1]->opProlong = opProlong;
  data[level+1]->opRestrict = opRestrict;

  PetscFunctionReturn(0);
};
