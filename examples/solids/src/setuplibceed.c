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
#include "../qfunctions/linElas.h"           // Linear elasticity
#include "../qfunctions/hyperSS.h"           // Hyperelasticity small strain
#include "../qfunctions/hyperFS.h"           // Hyperelasticity finite strain
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
problemData problemOptions[3] = {
  [ELAS_LIN] = {
    .qdatasize = 10, // For linear elasticity, 6 would be sufficient
    .setupgeo = SetupGeo,
    .apply = LinElasF,
    .jacob = LinElasdF,
    .energy = LinElasEnergy,
    .diagnostic = LinElasDiagnostic,
    .setupgeofname = SetupGeo_loc,
    .applyfname = LinElasF_loc,
    .jacobfname = LinElasdF_loc,
    .energyfname = LinElasEnergy_loc,
    .diagnosticfname = LinElasDiagnostic_loc,
    .qmode = CEED_GAUSS
  },
  [ELAS_HYPER_SS] = {
    .qdatasize = 10,
    .setupgeo = SetupGeo,
    .apply = HyperSSF,
    .jacob = HyperSSdF,
    .energy = HyperSSEnergy,
    .diagnostic = HyperSSDiagnostic,
    .setupgeofname = SetupGeo_loc,
    .applyfname = HyperSSF_loc,
    .jacobfname = HyperSSdF_loc,
    .energyfname = HyperSSEnergy_loc,
    .diagnosticfname = HyperSSDiagnostic_loc,
    .qmode = CEED_GAUSS
  },
  [ELAS_HYPER_FS] = {
    .qdatasize = 10,
    .setupgeo = SetupGeo,
    .apply = HyperFSF,
    .jacob = HyperFSdF,
    .energy = HyperFSEnergy,
    .diagnostic = HyperFSDiagnostic,
    .setupgeofname = SetupGeo_loc,
    .applyfname = HyperFSF_loc,
    .jacobfname = HyperFSdF_loc,
    .energyfname = HyperFSEnergy_loc,
    .diagnosticfname = HyperFSDiagnostic_loc,
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
  CeedVectorDestroy(&data->xceed);
  CeedVectorDestroy(&data->yceed);
  CeedVectorDestroy(&data->truesoln);

  // Restrictions
  CeedElemRestrictionDestroy(&data->Erestrictu);
  CeedElemRestrictionDestroy(&data->Erestrictx);
  CeedElemRestrictionDestroy(&data->ErestrictGradui);
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

// Get libCEED restriction data from DMPlex
PetscErrorCode CreateRestrictionPlex(Ceed ceed, CeedInt P, CeedInt ncomp,
                                     CeedElemRestriction *Erestrict, DM dm) {
  PetscInt ierr;
  PetscInt c, cStart, cEnd, nelem, nnodes, *erestrict, eoffset;
  PetscSection section;
  Vec Uloc;

  PetscFunctionBeginUser;

  // Get Nelem
  ierr = DMGetSection(dm, &section); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;

  // Get indices
  ierr = PetscMalloc1(nelem*P*P*P, &erestrict); CHKERRQ(ierr);
  for (c=cStart, eoffset=0; c<cEnd; c++) {
    PetscInt numindices, *indices, i;
    ierr = DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE,
                                   &numindices, &indices, NULL, NULL);
    CHKERRQ(ierr);
    for (i=0; i<numindices; i+=ncomp) {
      for (PetscInt j=0; j<ncomp; j++) {
        if (indices[i+j] != indices[i] + (PetscInt)(copysign(j, indices[i])))
          SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                   "Cell %D closure indices not interlaced", c);
      }
      // Essential boundary conditions are encoded as -(loc+1)
      PetscInt loc = indices[i] >= 0 ? indices[i] : -(indices[i] + 1);
      erestrict[eoffset++] = loc;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE,
                                       &numindices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }

  // Setup CEED restriction
  ierr = DMGetLocalVector(dm, &Uloc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Uloc, &nnodes); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &Uloc); CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, nelem, P*P*P, ncomp, 1, nnodes, CEED_MEM_HOST,
                            CEED_COPY_VALUES, erestrict, Erestrict);
  ierr = PetscFree(erestrict); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// Set up libCEED for a given degree
PetscErrorCode SetupLibceedFineLevel(DM dm, DM dmEnergy, DM dmDiagnostic,
                                     Ceed ceed, AppCtx appCtx,
                                     CeedQFunctionContext physCtx,
                                     CeedData *data, PetscInt fineLevel,
                                     PetscInt ncompu, PetscInt Ugsz,
                                     PetscInt Ulocsz, CeedVector forceCeed) {
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
  ierr = CreateRestrictionPlex(ceed, 2, ncompx, &(data[fineLevel]->Erestrictx),
                               dmcoord); CHKERRQ(ierr);
  // -- Solution restriction
  ierr = CreateRestrictionPlex(ceed, P, ncompu, &data[fineLevel]->Erestrictu,
                               dm); CHKERRQ(ierr);
  // -- Energy restriction
  ierr = CreateRestrictionPlex(ceed, P, ncompe,
                               &data[fineLevel]->ErestrictEnergy, dmEnergy);
  CHKERRQ(ierr);
  // -- Pressure restriction
  ierr = CreateRestrictionPlex(ceed, P, ncompu + ncompd,
                               &data[fineLevel]->ErestrictDiagnostic,
                               dmDiagnostic); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;

  // -- Geometric data restriction
  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, qdatasize,
                                   qdatasize*nelem*Q*Q*Q,
                                   CEED_STRIDES_BACKEND,
                                   &data[fineLevel]->Erestrictqdi);
  // -- State vector gradient restriction
  if (problemChoice != ELAS_LIN)
    CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, dim*ncompu,
                                     dim*ncompu*nelem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fineLevel]->ErestrictGradui);
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
  if (problemChoice != ELAS_LIN)
    CeedVectorCreate(ceed, dim*ncompu*nelem*nqpts, &data[fineLevel]->gradu);
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
  if (problemChoice != ELAS_LIN)
    CeedQFunctionAddOutput(qfApply, "gradu", ncompu*dim, CEED_EVAL_NONE);
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
  if (problemChoice != ELAS_LIN)
    CeedOperatorSetField(opApply, "gradu", data[fineLevel]->ErestrictGradui,
                         data[fineLevel]->basisu, data[fineLevel]->gradu);
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
  if (problemChoice != ELAS_LIN)
    CeedQFunctionAddInput(qfJacob, "gradu", ncompu*dim, CEED_EVAL_NONE);
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
  if (problemChoice != ELAS_LIN)
    CeedOperatorSetField(opJacob, "gradu", data[fineLevel]->ErestrictGradui,
                         CEED_BASIS_COLLOCATED, data[fineLevel]->gradu);

  // -- Save libCEED data
  data[fineLevel]->qfJacob = qfJacob;
  data[fineLevel]->opJacob = opJacob;

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
  ierr = CreateRestrictionPlex(ceed, P, ncompu, &data[level]->Erestrictu, dm);
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
