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

#ifndef setuparea_h
#define setuparea_h

#include <stdbool.h>
#include <string.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>
#include <ceed.h>
#include "qfunctions/area/areacube.h"
#include "qfunctions/area/areasphere.h"

#if PETSC_VERSION_LT(3,14,0)
#  define DMPlexGetClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexGetClosureIndices(a,b,c,d,f,g,i)
#  define DMPlexRestoreClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexRestoreClosureIndices(a,b,c,d,f,g,i)
#endif

// -----------------------------------------------------------------------------
// PETSc Operator Structs
// -----------------------------------------------------------------------------

// Data for PETSc
typedef struct User_ *User;
struct User_ {
  MPI_Comm comm;
  DM dm;
  Vec Xloc, Yloc, diag;
  CeedVector xceed, yceed;
  CeedOperator op;
  Ceed ceed;
};

// -----------------------------------------------------------------------------
// libCEED Data Struct
// -----------------------------------------------------------------------------

// libCEED data struct for level
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  Ceed ceed;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictqdi;
  CeedQFunction qf_apply;
  CeedOperator op_apply, op_restrict, op_interp;
  CeedVector qdata, uceed, vceed;
};

// -----------------------------------------------------------------------------
// Problem Option Data
// -----------------------------------------------------------------------------

// Problem options
typedef enum {
  CUBE = 0, SPHERE = 1
} problemType;
static const char *const problemTypes[] = {"cube", "sphere",
                                           "problemType", "AREA", NULL
                                          };

// Problem specific data
typedef struct {
  CeedInt ncompu, ncompx, qdatasize, qextra, topodim;
  CeedQFunctionUser setupgeo, apply;
  const char *setupgeofname, *applyfname;
  CeedEvalMode inmode, outmode;
  CeedQuadMode qmode;
} problemData;

static problemData problemOptions[6] = {
  [CUBE] = {
    .ncompx = 3,
    .ncompu = 1,
    .topodim = 2,
    .qdatasize = 1,
    .qextra = 1,
    .setupgeo = SetupMassGeoCube,
    .apply = Mass,
    .setupgeofname = SetupMassGeoCube_loc,
    .applyfname = Mass_loc,
    .inmode = CEED_EVAL_INTERP,
    .outmode = CEED_EVAL_INTERP,
    .qmode = CEED_GAUSS
  },
  [SPHERE] = {
    .ncompx = 3,
    .ncompu = 1,
    .topodim = 2,
    .qdatasize = 1,
    .qextra = 1,
    .setupgeo = SetupMassGeoSphere,
    .apply = Mass,
    .setupgeofname = SetupMassGeoSphere_loc,
    .applyfname = Mass_loc,
    .inmode = CEED_EVAL_INTERP,
    .outmode = CEED_EVAL_INTERP,
    .qmode = CEED_GAUSS
  }
};

// -----------------------------------------------------------------------------
// PETSc sphere auxiliary functions
// -----------------------------------------------------------------------------

// Utility function taken from petsc/src/dm/impls/plex/examples/tutorials/ex7.c
static PetscErrorCode ProjectToUnitSphere(DM dm) {
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
}

// -----------------------------------------------------------------------------
// PETSc Finite Element space setup
// -----------------------------------------------------------------------------

// Create FE by degree
static int PetscFECreateByDegree(DM dm, PetscInt dim, PetscInt Nc,
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
}

// -----------------------------------------------------------------------------
// PETSc Setup for Level
// -----------------------------------------------------------------------------

// This function sets up a DM for a given degree
static int SetupDMByDegree(DM dm, PetscInt degree, PetscInt ncompu,
                           PetscInt dim) {
  PetscInt ierr;
  PetscFE fe;

  PetscFunctionBeginUser;

  // Setup FE
  ierr = PetscFECreateByDegree(dm, dim, ncompu, PETSC_FALSE, NULL, degree, &fe);
  CHKERRQ(ierr);
  ierr = DMAddField(dm, NULL, (PetscObject)fe); CHKERRQ(ierr);

  // Setup DM
  ierr = DMCreateDS(dm); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// libCEED Setup for Level
// -----------------------------------------------------------------------------

// Destroy libCEED operator objects
static PetscErrorCode CeedDataDestroy(CeedData data) {
  PetscInt ierr;

  CeedVectorDestroy(&data->qdata);
  CeedVectorDestroy(&data->uceed);
  CeedVectorDestroy(&data->vceed);
  CeedBasisDestroy(&data->basisx);
  CeedBasisDestroy(&data->basisu);
  CeedElemRestrictionDestroy(&data->Erestrictu);
  CeedElemRestrictionDestroy(&data->Erestrictx);
  CeedElemRestrictionDestroy(&data->Erestrictqdi);
  CeedQFunctionDestroy(&data->qf_apply);
  CeedOperatorDestroy(&data->op_apply);
  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Auxiliary function to define CEED restrictions from DMPlex data
static int CreateRestrictionPlex(Ceed ceed, CeedInt P, CeedInt ncomp,
                                 CeedElemRestriction *Erestrict, DM dm) {
  PetscInt ierr;
  PetscInt c, cStart, cEnd, nelem, nnodes, *erestrict, eoffset;
  PetscSection section;
  Vec Uloc;

  PetscFunctionBegin;

  // Get Nelem
  ierr = DMGetSection(dm, &section); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart,& cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;

  // Get indices
  ierr = PetscMalloc1(nelem*P*P, &erestrict); CHKERRQ(ierr);
  for (c=cStart, eoffset = 0; c<cEnd; c++) {
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
      // NO BC on closed surfaces
      PetscInt loc = indices[i];
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
  CeedElemRestrictionCreate(ceed, nelem, P*P, ncomp, 1, nnodes,
                            CEED_MEM_HOST, CEED_COPY_VALUES, erestrict,
                            Erestrict);
  ierr = PetscFree(erestrict); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Set up libCEED for a given degree
static int SetupLibceedByDegree(DM dm, Ceed ceed, CeedInt degree,
                                CeedInt topodim, CeedInt qextra,
                                PetscInt ncompx, PetscInt ncompu,
                                PetscInt xlsize, problemType problemChoice,
                                CeedData data) {
  int ierr;
  DM dmcoord;
  Vec coords;
  const PetscScalar *coordArray;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictqdi;
  CeedQFunction qf_setupgeo, qf_apply;
  CeedOperator op_setupgeo, op_apply;
  CeedVector xcoord, qdata, uceed, vceed;
  CeedInt P, Q, cStart, cEnd, nelem,
          qdatasize = problemOptions[problemChoice].qdatasize;

  // CEED bases
  P = degree + 1;
  Q = P + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, topodim, ncompu, P, Q,
                                  problemOptions[problemChoice].qmode, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, topodim, ncompx, 2, Q,
                                  problemOptions[problemChoice].qmode, &basisx);

  // CEED restrictions
  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  ierr = CreateRestrictionPlex(ceed, 2, ncompx, &Erestrictx, dmcoord);
  CHKERRQ(ierr);
  ierr = CreateRestrictionPlex(ceed, P, ncompu, &Erestrictu, dm); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;

  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q, qdatasize,
                                   qdatasize*nelem*Q*Q,
                                   CEED_STRIDES_BACKEND, &Erestrictqdi);

  // Element coordinates
  ierr = DMGetCoordinatesLocal(dm, &coords); CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords, &coordArray); CHKERRQ(ierr);

  CeedElemRestrictionCreateVector(Erestrictx, &xcoord, NULL);
  CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_COPY_VALUES,
                     (PetscScalar *)coordArray);
  ierr = VecRestoreArrayRead(coords, &coordArray);

  // Create the vectors that will be needed in setup and apply
  CeedInt nqpts;
  CeedBasisGetNumQuadraturePoints(basisu, &nqpts);
  CeedVectorCreate(ceed, qdatasize*nelem*nqpts, &qdata);
  CeedVectorCreate(ceed, xlsize, &uceed);
  CeedVectorCreate(ceed, xlsize, &vceed);

  // Create the Q-function that builds the operator (i.e. computes its
  // quadrature data) and set its context data
  CeedQFunctionCreateInterior(ceed, 1, problemOptions[problemChoice].setupgeo,
                              problemOptions[problemChoice].setupgeofname,
                              &qf_setupgeo);
  CeedQFunctionAddInput(qf_setupgeo, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setupgeo, "dx", ncompx*topodim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setupgeo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setupgeo, "qdata", qdatasize, CEED_EVAL_NONE);

  // Set up the mass operator
  CeedQFunctionCreateInterior(ceed, 1, problemOptions[problemChoice].apply,
                              problemOptions[problemChoice].applyfname,
                              &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", ncompu,
                        problemOptions[problemChoice].inmode);
  CeedQFunctionAddInput(qf_apply, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", ncompu,
                         problemOptions[problemChoice].outmode);

  // Create the operator that builds the quadrature data for the operator
  CeedOperatorCreate(ceed, qf_setupgeo, NULL, NULL, &op_setupgeo);
  CeedOperatorSetField(op_setupgeo, "x", Erestrictx, basisx,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setupgeo, "dx", Erestrictx, basisx,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setupgeo, "weight", CEED_ELEMRESTRICTION_NONE, basisx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setupgeo, "qdata", Erestrictqdi,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the mass or diff operator
  CeedOperatorCreate(ceed, qf_apply, NULL, NULL, &op_apply);
  CeedOperatorSetField(op_apply, "u", Erestrictu, basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata", Erestrictqdi, CEED_BASIS_COLLOCATED,
                       qdata);
  CeedOperatorSetField(op_apply, "v", Erestrictu, basisu, CEED_VECTOR_ACTIVE);

  // Setup qdata
  CeedOperatorApply(op_setupgeo, xcoord, qdata, CEED_REQUEST_IMMEDIATE);

  // Cleanup
  CeedQFunctionDestroy(&qf_setupgeo);
  CeedOperatorDestroy(&op_setupgeo);
  CeedVectorDestroy(&xcoord);

  // Save libCEED data
  data->basisx = basisx;
  data->basisu = basisu;
  data->Erestrictx = Erestrictx;
  data->Erestrictu = Erestrictu;
  data->Erestrictqdi = Erestrictqdi;
  data->qf_apply = qf_apply;
  data->op_apply = op_apply;
  data->qdata = qdata;
  data->uceed = uceed;
  data->vceed = vceed;

  PetscFunctionReturn(0);
}

#endif // setuparea_h
