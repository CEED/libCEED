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

#ifndef setup_h
#define setup_h

#include <stdbool.h>
#include <string.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>
#include <ceed.h>
#include "qfunctions/bps/common.h"
#include "qfunctions/bps/bp1.h"
#include "qfunctions/bps/bp2.h"
#include "qfunctions/bps/bp3.h"
#include "qfunctions/bps/bp4.h"

// -----------------------------------------------------------------------------
// PETSc Operator Structs
// -----------------------------------------------------------------------------

// Data for PETSc Matshell
typedef struct UserO_ *UserO;
struct UserO_ {
  MPI_Comm comm;
  DM dm;
  Vec Xloc, Yloc, diag;
  CeedVector xceed, yceed;
  CeedOperator op;
  Ceed ceed;
};

// Data for PETSc Interp/Restrict Matshells
typedef struct UserIR_ *UserIR;
struct UserIR_ {
  MPI_Comm comm;
  DM dmc, dmf;
  Vec Xloc, Yloc, mult;
  CeedVector ceedvecc, ceedvecf;
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
  CeedBasis basisx, basisu, basisctof;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi, Erestrictui,
                      Erestrictqdi;
  CeedQFunction qf_apply;
  CeedOperator op_apply, op_restrict, op_interp;
  CeedVector qdata, xceed, yceed;
};

// -----------------------------------------------------------------------------
// Command Line Options
// -----------------------------------------------------------------------------

// Coarsening options
typedef enum {
  COARSEN_UNIFORM = 0, COARSEN_LOGARITHMIC = 1
} coarsenType;
static const char *const coarsenTypes [] = {"uniform","logarithmic",
                                            "coarsenType","COARSEN",0
                                           };

// -----------------------------------------------------------------------------
// Boundary Conditions
// -----------------------------------------------------------------------------

// Diff boundary condition function
PetscErrorCode BCsDiff(PetscInt dim, PetscReal time, const PetscReal x[],
                       PetscInt ncompu, PetscScalar *u, void *ctx) {
  // *INDENT-OFF*
  #ifndef M_PI
  #define M_PI    3.14159265358979323846
  #endif
  // *INDENT-ON*
  const CeedScalar c[3] = { 0, 1., 2. };
  const CeedScalar k[3] = { 1., 2., 3. };

  PetscFunctionBeginUser;

  for (PetscInt i = 0; i < ncompu; i++)
    u[i] = sin(M_PI*(c[0] + k[0]*x[0])) *
           sin(M_PI*(c[1] + k[1]*x[1])) *
           sin(M_PI*(c[2] + k[2]*x[2]));

  PetscFunctionReturn(0);
}

// Mass boundary condition function
PetscErrorCode BCsMass(PetscInt dim, PetscReal time, const PetscReal x[],
                       PetscInt ncompu, PetscScalar *u, void *ctx) {
  PetscFunctionBeginUser;

  for (PetscInt i = 0; i < ncompu; i++)
    u[i] = PetscSqrtScalar(PetscSqr(x[0]) + PetscSqr(x[1]) +
                           PetscSqr(x[2]));

  PetscFunctionReturn(0);
}

// Create BC label
static PetscErrorCode CreateBCLabel(DM dm, const char name[]) {
  int ierr;
  DMLabel label;

  PetscFunctionBeginUser;

  ierr = DMCreateLabel(dm, name); CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label); CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, 1, label); CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm, label); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// BP Option Data
// -----------------------------------------------------------------------------

// BP options
typedef enum {
  CEED_BP1 = 0, CEED_BP2 = 1, CEED_BP3 = 2,
  CEED_BP4 = 3, CEED_BP5 = 4, CEED_BP6 = 5
} bpType;
static const char *const bpTypes[] = {"bp1","bp2","bp3","bp4","bp5","bp6",
                                      "bpType","CEED_BP",0
                                     };

// BP specific data
typedef struct {
  CeedInt ncompu, qdatasize, qextra;
  CeedQFunctionUser setupgeo, setuprhs, apply, error;
  const char *setupgeofname, *setuprhsfname, *applyfname, *errorfname;
  CeedEvalMode inmode, outmode;
  CeedQuadMode qmode;
  PetscBool enforce_bc;
  PetscErrorCode (*bcs_func)(PetscInt, PetscReal, const PetscReal *,
                             PetscInt, PetscScalar *, void *);
} bpData;

bpData bpOptions[6] = {
  [CEED_BP1] = {
    .ncompu = 1,
    .qdatasize = 1,
    .qextra = 1,
    .setupgeo = SetupMassGeo,
    .setuprhs = SetupMassRhs,
    .apply = Mass,
    .error = Error,
    .setupgeofname = SetupMassGeo_loc,
    .setuprhsfname = SetupMassRhs_loc,
    .applyfname = Mass_loc,
    .errorfname = Error_loc,
    .inmode = CEED_EVAL_INTERP,
    .outmode = CEED_EVAL_INTERP,
    .qmode = CEED_GAUSS,
    .enforce_bc = false,
    .bcs_func = BCsMass
  },
  [CEED_BP2] = {
    .ncompu = 3,
    .qdatasize = 1,
    .qextra = 1,
    .setupgeo = SetupMassGeo,
    .setuprhs = SetupMassRhs3,
    .apply = Mass3,
    .error = Error3,
    .setupgeofname = SetupMassGeo_loc,
    .setuprhsfname = SetupMassRhs3_loc,
    .applyfname = Mass3_loc,
    .errorfname = Error3_loc,
    .inmode = CEED_EVAL_INTERP,
    .outmode = CEED_EVAL_INTERP,
    .qmode = CEED_GAUSS,
    .enforce_bc = false,
    .bcs_func = BCsMass
  },
  [CEED_BP3] = {
    .ncompu = 1,
    .qdatasize = 6,
    .qextra = 1,
    .setupgeo = SetupDiffGeo,
    .setuprhs = SetupDiffRhs,
    .apply = Diff,
    .error = Error,
    .setupgeofname = SetupDiffGeo_loc,
    .setuprhsfname = SetupDiffRhs_loc,
    .applyfname = Diff_loc,
    .errorfname = Error_loc,
    .inmode = CEED_EVAL_GRAD,
    .outmode = CEED_EVAL_GRAD,
    .qmode = CEED_GAUSS,
    .enforce_bc = true,
    .bcs_func = BCsDiff
  },
  [CEED_BP4] = {
    .ncompu = 3,
    .qdatasize = 6,
    .qextra = 1,
    .setupgeo = SetupDiffGeo,
    .setuprhs = SetupDiffRhs3,
    .apply = Diff3,
    .error = Error3,
    .setupgeofname = SetupDiffGeo_loc,
    .setuprhsfname = SetupDiffRhs3_loc,
    .applyfname = Diff_loc,
    .errorfname = Error3_loc,
    .inmode = CEED_EVAL_GRAD,
    .outmode = CEED_EVAL_GRAD,
    .qmode = CEED_GAUSS,
    .enforce_bc = true,
    .bcs_func = BCsDiff
  },
  [CEED_BP5] = {
    .ncompu = 1,
    .qdatasize = 6,
    .qextra = 0,
    .setupgeo = SetupDiffGeo,
    .setuprhs = SetupDiffRhs,
    .apply = Diff,
    .error = Error,
    .setupgeofname = SetupDiffGeo_loc,
    .setuprhsfname = SetupDiffRhs_loc,
    .applyfname = Diff_loc,
    .errorfname = Error_loc,
    .inmode = CEED_EVAL_GRAD,
    .outmode = CEED_EVAL_GRAD,
    .qmode = CEED_GAUSS_LOBATTO,
    .enforce_bc = true,
    .bcs_func = BCsDiff
  },
  [CEED_BP6] = {
    .ncompu = 3,
    .qdatasize = 6,
    .qextra = 0,
    .setupgeo = SetupDiffGeo,
    .setuprhs = SetupDiffRhs3,
    .apply = Diff3,
    .error = Error3,
    .setupgeofname = SetupDiffGeo_loc,
    .setuprhsfname = SetupDiffRhs3_loc,
    .applyfname = Diff_loc,
    .errorfname = Error3_loc,
    .inmode = CEED_EVAL_GRAD,
    .outmode = CEED_EVAL_GRAD,
    .qmode = CEED_GAUSS_LOBATTO,
    .enforce_bc = true,
    .bcs_func = BCsDiff
  }
};

// -----------------------------------------------------------------------------
// PETSc FE Boilerplate
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
    ierr = PetscDTGaussJacobiQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0,
                                        &q); CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0,
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
                           bpType bpChoice) {
  PetscInt ierr, dim, marker_ids[1] = {1};
  PetscFE fe;

  PetscFunctionBeginUser;

  // Setup FE
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ierr = PetscFECreateByDegree(dm, dim, ncompu, PETSC_FALSE, NULL, degree, &fe);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMAddField(dm, NULL, (PetscObject)fe); CHKERRQ(ierr);

  // Setup DM
  ierr = DMCreateDS(dm); CHKERRQ(ierr);
  if (bpOptions[bpChoice].enforce_bc) {
    PetscBool hasLabel;
    DMHasLabel(dm, "marker", &hasLabel);
    if (!hasLabel) {CreateBCLabel(dm, "marker");}
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL,
                         (void(*)(void))bpOptions[bpChoice].bcs_func,
                         1, marker_ids, NULL);
    CHKERRQ(ierr);
  }
  ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// libCEED Setup for Level
// -----------------------------------------------------------------------------

// Destroy libCEED operator objects
static PetscErrorCode CeedDataDestroy(CeedInt i, CeedData data) {
  PetscInt ierr;

  CeedVectorDestroy(&data->qdata);
  CeedVectorDestroy(&data->xceed);
  CeedVectorDestroy(&data->yceed);
  CeedBasisDestroy(&data->basisx);
  CeedBasisDestroy(&data->basisu);
  CeedElemRestrictionDestroy(&data->Erestrictu);
  CeedElemRestrictionDestroy(&data->Erestrictx);
  CeedElemRestrictionDestroy(&data->Erestrictui);
  CeedElemRestrictionDestroy(&data->Erestrictxi);
  CeedElemRestrictionDestroy(&data->Erestrictqdi);
  CeedQFunctionDestroy(&data->qf_apply);
  CeedOperatorDestroy(&data->op_apply);
  if (i > 0) {
    CeedOperatorDestroy(&data->op_interp);
    CeedBasisDestroy(&data->basisctof);
    CeedOperatorDestroy(&data->op_restrict);
  }
  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Get CEED restriction data from DMPlex
static int CreateRestrictionPlex(Ceed ceed, CeedInt P, CeedInt ncomp,
                                 CeedElemRestriction *Erestrict, DM dm) {
  PetscInt ierr;
  PetscInt c, cStart, cEnd, nelem, nnodes, *erestrict, eoffset;
  PetscSection section;
  Vec Uloc;

  PetscFunctionBeginUser;

  // Get Nelem
  ierr = DMGetSection(dm, &section); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart,& cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;

  // Get indices
  ierr = PetscMalloc1(nelem*P*P*P, &erestrict); CHKERRQ(ierr);
  for (c=cStart, eoffset=0; c<cEnd; c++) {
    PetscInt numindices, *indices, i;
    ierr = DMPlexGetClosureIndices(dm, section, section, c, &numindices,
                                   &indices, NULL); CHKERRQ(ierr);
    for (i=0; i<numindices; i+=ncomp) {
      for (PetscInt j=0; j<ncomp; j++) {
        if (indices[i+j] != indices[i] + (PetscInt)(copysign(j, indices[i])))
          SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                   "Cell %D closure indices not interlaced", c);
      }
      // Essential boundary conditions are encoded as -(loc+1)
      PetscInt loc = indices[i] >= 0 ? indices[i] : -(indices[i] + 1);
      erestrict[eoffset++] = loc/ncomp;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, &numindices,
                                       &indices, NULL); CHKERRQ(ierr);
  }

  // Setup CEED restriction
  ierr = DMGetLocalVector(dm, &Uloc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Uloc, &nnodes); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &Uloc); CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, nelem, P*P*P, nnodes/ncomp, ncomp,
                            CEED_MEM_HOST, CEED_COPY_VALUES, erestrict,
                            Erestrict);
  ierr = PetscFree(erestrict); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Set up libCEED for a given degree
static int SetupLibceedByDegree(DM dm, Ceed ceed, CeedInt degree, CeedInt dim,
                                CeedInt qextra, PetscInt ncompu, PetscInt gsize,
                                PetscInt xlsize, bpType bpChoice, CeedData data,
                                PetscBool setup_rhs, CeedVector rhsceed,
                                CeedVector *target) {
  int ierr;
  DM dmcoord;
  PetscSection section;
  Vec coords;
  const PetscScalar *coordArray;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi,
                      Erestrictui, Erestrictqdi;
  CeedQFunction qf_setupgeo, qf_apply;
  CeedOperator op_setupgeo, op_apply;
  CeedVector xcoord, qdata, xceed, yceed;
  CeedInt qdatasize = bpOptions[bpChoice].qdatasize, ncompx = dim, P, Q,
          cStart, cEnd, nelem;

  // CEED bases
  P = degree + 1;
  Q = P + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompu, P, Q,
                                  bpOptions[bpChoice].qmode, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, Q,
                                  bpOptions[bpChoice].qmode, &basisx);

  // CEED restrictions
  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  CreateRestrictionPlex(ceed, 2, ncompx, &Erestrictx, dmcoord);
  CreateRestrictionPlex(ceed, P, ncompu, &Erestrictu, dm);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;

  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q*Q, nelem*Q*Q*Q, ncompu,
                                    &Erestrictui); CHKERRQ(ierr);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q*Q, nelem*Q*Q*Q,
                                    qdatasize, &Erestrictqdi); CHKERRQ(ierr);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q*Q, nelem*Q*Q*Q, ncompx,
                                    &Erestrictxi); CHKERRQ(ierr);

  // Element coordinates
  ierr = DMGetCoordinatesLocal(dm, &coords); CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords, &coordArray); CHKERRQ(ierr);
  ierr = DMGetSection(dmcoord, &section); CHKERRQ(ierr);

  CeedElemRestrictionCreateVector(Erestrictx, &xcoord, NULL);
  CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_COPY_VALUES,
                     (PetscScalar *)coordArray);
  ierr = VecRestoreArrayRead(coords, &coordArray); CHKERRQ(ierr);

  // Create the persistent vectors that will be needed in setup and apply
  CeedInt nqpts;
  CeedBasisGetNumQuadraturePoints(basisu, &nqpts);
  CeedVectorCreate(ceed, qdatasize*nelem*nqpts, &qdata);
  CeedVectorCreate(ceed, xlsize, &xceed);
  CeedVectorCreate(ceed, xlsize, &yceed);

  // Create the Q-function that builds the operator (i.e. computes its
  // quadrature data) and set its context data
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].setupgeo,
                              bpOptions[bpChoice].setupgeofname, &qf_setupgeo);
  CeedQFunctionAddInput(qf_setupgeo, "dx", ncompx*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setupgeo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setupgeo, "qdata", qdatasize, CEED_EVAL_NONE);

  // Set up PDE operator
  CeedInt inscale = bpOptions[bpChoice].inmode==CEED_EVAL_GRAD ? dim : 1;
  CeedInt outscale = bpOptions[bpChoice].outmode==CEED_EVAL_GRAD ? dim : 1;
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].apply,
                              bpOptions[bpChoice].applyfname, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", ncompu*inscale,
                        bpOptions[bpChoice].inmode);
  CeedQFunctionAddInput(qf_apply, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", ncompu*outscale,
                         bpOptions[bpChoice].outmode);

  // Create the operator that builds the quadrature data for the operator
  CeedOperatorCreate(ceed, qf_setupgeo, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_setupgeo);
  CeedOperatorSetField(op_setupgeo, "dx", Erestrictx, CEED_TRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setupgeo, "weight", Erestrictxi, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setupgeo, "qdata", Erestrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the operator
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_apply);
  CeedOperatorSetField(op_apply, "u", Erestrictu, CEED_TRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata", Erestrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_apply, "v", Erestrictu, CEED_TRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);

  // Setup qdata
  CeedOperatorApply(op_setupgeo, xcoord, qdata, CEED_REQUEST_IMMEDIATE);

  // Set up RHS if needed
  if (setup_rhs) {
    CeedQFunction qf_setuprhs;
    CeedOperator op_setuprhs;
    CeedVectorCreate(ceed, nelem*nqpts*ncompu, target);

    // Create the q-function that sets up the RHS and true solution
    CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].setuprhs,
                                bpOptions[bpChoice].setuprhsfname, &qf_setuprhs);
    CeedQFunctionAddInput(qf_setuprhs, "x", dim, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_setuprhs, "dx", ncompx*dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_setuprhs, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddOutput(qf_setuprhs, "true_soln", ncompu, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_setuprhs, "rhs", ncompu, CEED_EVAL_INTERP);

    // Create the operator that builds the RHS and true solution
    CeedOperatorCreate(ceed, qf_setuprhs, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &op_setuprhs);
    CeedOperatorSetField(op_setuprhs, "x", Erestrictx, CEED_TRANSPOSE,
                         basisx, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setuprhs, "dx", Erestrictx, CEED_TRANSPOSE,
                         basisx, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setuprhs, "weight", Erestrictxi, CEED_NOTRANSPOSE,
                         basisx, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_setuprhs, "true_soln", Erestrictui, CEED_NOTRANSPOSE,
                         CEED_BASIS_COLLOCATED, *target);
    CeedOperatorSetField(op_setuprhs, "rhs", Erestrictu, CEED_TRANSPOSE,
                         basisu, CEED_VECTOR_ACTIVE);

    // Setup RHS and target
    CeedOperatorApply(op_setuprhs, xcoord, rhsceed, CEED_REQUEST_IMMEDIATE);
    CeedVectorSyncArray(rhsceed, CEED_MEM_HOST);

    // Cleanup
    CeedQFunctionDestroy(&qf_setuprhs);
    CeedOperatorDestroy(&op_setuprhs);
  }

  // Cleanup
  CeedQFunctionDestroy(&qf_setupgeo);
  CeedOperatorDestroy(&op_setupgeo);
  CeedVectorDestroy(&xcoord);

  // Save libCEED data required for level
  data->basisx = basisx; data->basisu = basisu;
  data->Erestrictx = Erestrictx;
  data->Erestrictu = Erestrictu;
  data->Erestrictxi = Erestrictxi;
  data->Erestrictui = Erestrictui;
  data->Erestrictqdi = Erestrictqdi;
  data->qf_apply = qf_apply;
  data->op_apply = op_apply;
  data->qdata = qdata;
  data->xceed = xceed;
  data->yceed = yceed;

  PetscFunctionReturn(0);
}

#ifdef multigrid
// Setup libCEED level transfer operator objects
static PetscErrorCode CeedLevelTransferSetup(Ceed ceed, CeedInt numlevels,
    CeedInt ncompu, bpType bpChoice, CeedData *data, CeedInt *leveldegrees,
    CeedQFunction qf_restrict, CeedQFunction qf_prolong) {
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
    CeedOperator op_restrict;

    // Basis
    CeedBasisCreateTensorH1Lagrange(ceed, 3, ncompu, Pc, Pf,
                                    CEED_GAUSS_LOBATTO, &basisctof);

    // Create the restriction operator
    CeedOperatorCreate(ceed, qf_restrict, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &op_restrict);
    CeedOperatorSetField(op_restrict, "input", data[i]->Erestrictu,
                         CEED_NOTRANSPOSE, CEED_BASIS_COLLOCATED,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_restrict, "output", data[i-1]->Erestrictu,
                         CEED_TRANSPOSE, basisctof, CEED_VECTOR_ACTIVE);

    // Save libCEED data required for level
    data[i]->basisctof = basisctof;
    data[i]->op_restrict = op_restrict;

    // Interpolation - Corse to fine
    CeedOperator op_interp;

    // Create the prolongation operator
    CeedOperatorCreate(ceed, qf_prolong, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &op_interp);
    CeedOperatorSetField(op_interp, "input", data[i-1]->Erestrictu,
                         CEED_NOTRANSPOSE, basisctof, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_interp, "output", data[i]->Erestrictu,
                         CEED_TRANSPOSE, CEED_BASIS_COLLOCATED,
                         CEED_VECTOR_ACTIVE);

    // Save libCEED data required for level
    data[i]->op_interp = op_interp;
  }

  PetscFunctionReturn(0);
}
#endif

// -----------------------------------------------------------------------------
// Mat Shell Functions
// -----------------------------------------------------------------------------

#ifdef multigrid
// This function returns the computed diagonal of the operator
static PetscErrorCode MatGetDiag(Mat A, Vec D) {
  PetscErrorCode ierr;
  UserO user;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  ierr = VecCopy(user->diag, D); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif

// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
static PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserO user;
  PetscScalar *x, *y;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = DMGlobalToLocalBegin(user->dm, X, INSERT_VALUES, user->Xloc);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dm, X, INSERT_VALUES, user->Xloc);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Yloc); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecGetArray(user->Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorSetArray(user->yceed, CEED_MEM_HOST, CEED_USE_POINTER, y);

  // Apply CEED operator
  CeedOperatorApply(user->op, user->xceed, user->yceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorSyncArray(user->yceed, CEED_MEM_HOST);

  // Restore PETSc vectors
  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Yloc, &y); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(user->dm, user->Yloc, ADD_VALUES, Y);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dm, user->Yloc, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#ifdef multigrid
// This function uses libCEED to compute the action of the interp operator
static PetscErrorCode MatMult_Interp(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserIR user;
  PetscScalar *x, *y;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->Xloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->dmc, X, INSERT_VALUES, user->Xloc);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dmc, X, INSERT_VALUES, user->Xloc);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Yloc); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecGetArray(user->Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedvecc, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorSetArray(user->ceedvecf, CEED_MEM_HOST, CEED_USE_POINTER, y);

  // Apply CEED operator
  CeedOperatorApply(user->op, user->ceedvecc, user->ceedvecf,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorSyncArray(user->ceedvecf, CEED_MEM_HOST);

  // Restore PETSc vectors
  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Yloc, &y); CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(user->Yloc, user->Yloc, user->mult);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(user->dmf, user->Yloc, ADD_VALUES, Y);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dmf, user->Yloc, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// This function uses libCEED to compute the action of the restriction operator
static PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserIR user;
  PetscScalar *x, *y;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->Xloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->dmf, X, INSERT_VALUES, user->Xloc);
  CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dmf, X, INSERT_VALUES, user->Xloc);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Yloc); CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(user->Xloc, user->Xloc, user->mult); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecGetArray(user->Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedvecf, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorSetArray(user->ceedvecc, CEED_MEM_HOST, CEED_USE_POINTER, y);

  // Apply CEED operator
  CeedOperatorApply(user->op, user->ceedvecf, user->ceedvecc,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorSyncArray(user->ceedvecc, CEED_MEM_HOST);

  // Restore PETSc vectors
  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Yloc, &y); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(user->dmc, user->Yloc, ADD_VALUES, Y);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dmc, user->Yloc, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif

// This function calculates the error in the final solution
static PetscErrorCode ComputeErrorMax(UserO user, CeedOperator op_error,
                                      Vec X, CeedVector target,
                                      PetscReal *maxerror) {
  PetscErrorCode ierr;
  PetscScalar *x;
  CeedVector collocated_error;
  CeedInt length;

  PetscFunctionBeginUser;
  CeedVectorGetLength(target, &length);
  CeedVectorCreate(user->ceed, length, &collocated_error);

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->Xloc); CHKERRQ(ierr);

  // Setup CEED vector
  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Apply CEED operator
  CeedOperatorApply(op_error, user->xceed, collocated_error,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vector
  VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);

  // Reduce max error
  *maxerror = 0;
  const CeedScalar *e;
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &e);
  for (CeedInt i=0; i<length; i++) {
    *maxerror = PetscMax(*maxerror, PetscAbsScalar(e[i]));
  }
  CeedVectorRestoreArrayRead(collocated_error, &e);
  ierr = MPI_Allreduce(MPI_IN_PLACE, maxerror,
                       1, MPIU_REAL, MPIU_MAX, user->comm); CHKERRQ(ierr);

  // Cleanup
  CeedVectorDestroy(&collocated_error);

  PetscFunctionReturn(0);
}
#endif
