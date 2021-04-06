#include "../include/libceedsetup.h"
#include "../include/petscutils.h"
#include <stdio.h>

// -----------------------------------------------------------------------------
// Destroy libCEED operator objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedDataDestroy(CeedInt i, CeedData data) {
  int ierr;

  CeedVectorDestroy(&data->qdata);
  CeedVectorDestroy(&data->Xceed);
  CeedVectorDestroy(&data->Yceed);
  CeedBasisDestroy(&data->basisx);
  CeedBasisDestroy(&data->basisu);
  CeedElemRestrictionDestroy(&data->Erestrictu);
  CeedElemRestrictionDestroy(&data->Erestrictx);
  CeedElemRestrictionDestroy(&data->Erestrictui);
  CeedElemRestrictionDestroy(&data->Erestrictqdi);
  CeedQFunctionDestroy(&data->qfApply);
  CeedOperatorDestroy(&data->opApply);
  if (i > 0) {
    CeedOperatorDestroy(&data->opProlong);
    CeedBasisDestroy(&data->basisctof);
    CeedOperatorDestroy(&data->opRestrict);
  }
  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Set up libCEED for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupLibceedByDegree(DM dm, Ceed ceed, CeedInt degree,
                                    CeedInt topodim, CeedInt qextra,
                                    PetscInt ncompx, PetscInt ncompu,
                                    PetscInt gsize, PetscInt xlsize,
                                    bpData bpData, CeedData data,
                                    PetscBool setup_rhs, CeedVector rhsceed,
                                    CeedVector *target) {
  int ierr;
  DM dmcoord;
  Vec coords;
  const PetscScalar *coordArray;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictui, Erestrictqdi;
  CeedQFunction qfSetupGeo, qfApply;
  CeedOperator opSetupGeo, opApply;
  CeedVector xcoord, qdata, Xceed, Yceed;
  CeedInt P, Q, nqpts, cStart, cEnd, nelem, qdatasize = bpData.qdatasize;
  CeedScalar R = 1,                      // radius of the sphere
             l = 1.0/PetscSqrtReal(3.0); // half edge of the inscribed cube

  // CEED bases
  P = degree + 1;
  Q = P + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, topodim, ncompu, P, Q, bpData.qmode,
                                  &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, topodim, ncompx, 2, Q, bpData.qmode,
                                  &basisx);
  CeedBasisGetNumQuadraturePoints(basisu, &nqpts);

  // CEED restrictions
  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dmcoord, 2, topodim, 0, 0, 0,
                                   &Erestrictx);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dm, P, topodim, 0, 0, 0, &Erestrictu);
  CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;

  CeedElemRestrictionCreateStrided(ceed, nelem, nqpts, ncompu, ncompu*nelem*nqpts,
                                   CEED_STRIDES_BACKEND, &Erestrictui);
  CeedElemRestrictionCreateStrided(ceed, nelem, nqpts, qdatasize,
                                   qdatasize*nelem*nqpts,
                                   CEED_STRIDES_BACKEND, &Erestrictqdi);

  // Element coordinates
  ierr = DMGetCoordinatesLocal(dm, &coords); CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords, &coordArray); CHKERRQ(ierr);

  CeedElemRestrictionCreateVector(Erestrictx, &xcoord, NULL);
  CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_COPY_VALUES,
                     (PetscScalar *)coordArray);
  ierr = VecRestoreArrayRead(coords, &coordArray);

  // Create the persistent vectors that will be needed in setup and apply
  CeedVectorCreate(ceed, qdatasize*nelem*nqpts, &qdata);
  CeedVectorCreate(ceed, xlsize, &Xceed);
  CeedVectorCreate(ceed, xlsize, &Yceed);

  // Create the QFunction that builds the context data
  CeedQFunctionCreateInterior(ceed, 1, bpData.setupgeo, bpData.setupgeofname,
                              &qfSetupGeo);
  CeedQFunctionAddInput(qfSetupGeo, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qfSetupGeo, "dx", ncompx*topodim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qfSetupGeo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qfSetupGeo, "qdata", qdatasize, CEED_EVAL_NONE);

  // Create the operator that builds the quadrature data
  CeedOperatorCreate(ceed, qfSetupGeo, NULL, NULL, &opSetupGeo);
  CeedOperatorSetField(opSetupGeo, "x", Erestrictx, basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opSetupGeo, "dx", Erestrictx, basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opSetupGeo, "weight", CEED_ELEMRESTRICTION_NONE, basisx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(opSetupGeo, "qdata", Erestrictqdi,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Setup qdata
  CeedOperatorApply(opSetupGeo, xcoord, qdata, CEED_REQUEST_IMMEDIATE);

  // Set up PDE operator
  CeedInt inscale = bpData.inmode == CEED_EVAL_GRAD ? topodim : 1;
  CeedInt outscale = bpData.outmode == CEED_EVAL_GRAD ? topodim : 1;
  CeedQFunctionCreateInterior(ceed, 1, bpData.apply, bpData.applyfname, &qfApply);
  CeedQFunctionAddInput(qfApply, "u", ncompu*inscale, bpData.inmode);
  CeedQFunctionAddInput(qfApply, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qfApply, "v", ncompu*outscale, bpData.outmode);

  // Create the mass or diff operator
  CeedOperatorCreate(ceed, qfApply, NULL, NULL, &opApply);
  CeedOperatorSetField(opApply, "u", Erestrictu, basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opApply, "qdata", Erestrictqdi, CEED_BASIS_COLLOCATED,
                       qdata);
  CeedOperatorSetField(opApply, "v", Erestrictu, basisu, CEED_VECTOR_ACTIVE);

  // Set up RHS if needed
  if (setup_rhs) {
    CeedQFunction qfSetupRHS;
    CeedOperator opSetupRHS;
    CeedVectorCreate(ceed, nelem*nqpts*ncompu, target);

    // Create the q-function that sets up the RHS and true solution
    CeedQFunctionCreateInterior(ceed, 1, bpData.setuprhs, bpData.setuprhsfname,
                                &qfSetupRHS);
    CeedQFunctionAddInput(qfSetupRHS, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qfSetupRHS, "qdata", qdatasize, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qfSetupRHS, "true_soln", ncompu, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qfSetupRHS, "rhs", ncompu, CEED_EVAL_INTERP);

    // Create the operator that builds the RHS and true solution
    CeedOperatorCreate(ceed, qfSetupRHS, NULL, NULL, &opSetupRHS);
    CeedOperatorSetField(opSetupRHS, "x", Erestrictx, basisx, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(opSetupRHS, "qdata", Erestrictqdi, CEED_BASIS_COLLOCATED,
                         qdata);
    CeedOperatorSetField(opSetupRHS, "true_soln", Erestrictui,
                         CEED_BASIS_COLLOCATED, *target);
    CeedOperatorSetField(opSetupRHS, "rhs", Erestrictu, basisu,
                         CEED_VECTOR_ACTIVE);

    // Set up the libCEED context
    CeedQFunctionContext rhsSetupCtx;
    CeedQFunctionContextCreate(ceed, &rhsSetupCtx);
    CeedScalar rhsSetupData[2] = {R, l};
    CeedQFunctionContextSetData(rhsSetupCtx, CEED_MEM_HOST, CEED_COPY_VALUES,
                                sizeof rhsSetupData, &rhsSetupData);
    CeedQFunctionSetContext(qfSetupRHS, rhsSetupCtx);
    CeedQFunctionContextDestroy(&rhsSetupCtx);

    // Setup RHS and target
    CeedOperatorApply(opSetupRHS, xcoord, rhsceed, CEED_REQUEST_IMMEDIATE);

    // Cleanup
    CeedQFunctionDestroy(&qfSetupRHS);
    CeedOperatorDestroy(&opSetupRHS);
  }

  // Cleanup
  CeedQFunctionDestroy(&qfSetupGeo);
  CeedOperatorDestroy(&opSetupGeo);
  CeedVectorDestroy(&xcoord);

  // Save libCEED data required for level
  data->basisx = basisx; data->basisu = basisu;
  data->Erestrictx = Erestrictx;
  data->Erestrictu = Erestrictu;
  data->Erestrictui = Erestrictui;
  data->Erestrictqdi = Erestrictqdi;
  data->qfApply = qfApply;
  data->opApply = opApply;
  data->qdata = qdata;
  data->Xceed = Xceed;
  data->Yceed = Yceed;

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Setup libCEED level transfer operator objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedLevelTransferSetup(Ceed ceed, CeedInt numlevels,
                                      CeedInt ncompu,
                                      CeedData *data, CeedInt *leveldegrees,
                                      CeedQFunction qfrestrict, CeedQFunction qfprolong) {
  // Return early if numlevels=1
  if (numlevels==1)
    PetscFunctionReturn(0);

  // Set up each level
  for (CeedInt i=1; i<numlevels; i++) {
    // P coarse and P fine
    CeedInt Pc = leveldegrees[i-1] + 1;
    CeedInt Pf = leveldegrees[i] + 1;

    // Restriction - Fine to corse
    CeedBasis basisctof;
    CeedOperator opRestrict;

    // Basis
    CeedBasisCreateTensorH1Lagrange(ceed, 3, ncompu, Pc, Pf,
                                    CEED_GAUSS_LOBATTO, &basisctof);

    // Create the restriction operator
    CeedOperatorCreate(ceed, qfrestrict, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &opRestrict);
    CeedOperatorSetField(opRestrict, "input", data[i]->Erestrictu,
                         CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(opRestrict, "output", data[i-1]->Erestrictu,
                         basisctof, CEED_VECTOR_ACTIVE);

    // Save libCEED data required for level
    data[i]->basisctof = basisctof;
    data[i]->opRestrict = opRestrict;

    // Interpolation - Corse to fine
    CeedOperator opProlong;

    // Create the prolongation operator
    CeedOperatorCreate(ceed, qfprolong, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &opProlong);
    CeedOperatorSetField(opProlong, "input", data[i-1]->Erestrictu,
                         basisctof, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(opProlong, "output", data[i]->Erestrictu,
                         CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

    // Save libCEED data required for level
    data[i]->opProlong = opProlong;
  }

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
