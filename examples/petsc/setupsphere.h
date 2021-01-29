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

#ifndef setupsphere_h
#define setupsphere_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>
#include <stdbool.h>
#include <string.h>
#include "qfunctions/bps/bp1sphere.h"
#include "qfunctions/bps/bp2sphere.h"
#include "qfunctions/bps/bp3sphere.h"
#include "qfunctions/bps/bp4sphere.h"
#include "qfunctions/bps/common.h"

#if PETSC_VERSION_LT(3,14,0)
#  define DMPlexGetClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexGetClosureIndices(a,b,c,d,f,g,i)
#  define DMPlexRestoreClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexRestoreClosureIndices(a,b,c,d,f,g,i)
#endif

#if PETSC_VERSION_LT(3,14,0)
#  define DMPlexCreateSphereMesh(a,b,c,d,e) DMPlexCreateSphereMesh(a,b,c,e)
#endif

// -----------------------------------------------------------------------------
// PETSc Operator Structs
// -----------------------------------------------------------------------------

// Data for PETSc
typedef struct UserO_ *UserO;
struct UserO_ {
  MPI_Comm comm;
  DM dm;
  Vec Xloc, Yloc, diag;
  CeedVector xceed, yceed;
  CeedOperator op;
  Ceed ceed;
};

// Data for PETSc Interp/Restrict operators
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
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictui, Erestrictqdi;
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

static bpData bpOptions[6] = {
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
    .qmode = CEED_GAUSS
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
    .qmode = CEED_GAUSS
  },
  [CEED_BP3] = {
    .ncompu = 1,
    .qdatasize = 4,
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
    .qmode = CEED_GAUSS
  },
  [CEED_BP4] = {
    .ncompu = 3,
    .qdatasize = 4,
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
    .qmode = CEED_GAUSS
  },
  [CEED_BP5] = {
    .ncompu = 1,
    .qdatasize = 4,
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
    .qmode = CEED_GAUSS_LOBATTO
  },
  [CEED_BP6] = {
    .ncompu = 3,
    .qdatasize = 4,
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
    .qmode = CEED_GAUSS_LOBATTO
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
        if (indices[i+j] != indices[i] + copysign(j, indices[i]))
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
  CeedElemRestrictionCreate(ceed, nelem, P*P, ncomp, 1, nnodes, CEED_MEM_HOST,
                            CEED_COPY_VALUES, erestrict, Erestrict);
  ierr = PetscFree(erestrict); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Set up libCEED for a given degree
static int SetupLibceedByDegree(DM dm, Ceed ceed, CeedInt degree,
                                CeedInt topodim, CeedInt qextra,
                                PetscInt ncompx, PetscInt ncompu,
                                PetscInt gsize, PetscInt xlsize,
                                bpType bpChoice, CeedData data,
                                PetscBool setup_rhs, CeedVector rhsceed,
                                CeedVector *target) {
  int ierr;
  DM dmcoord;
  Vec coords;
  const PetscScalar *coordArray;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictui, Erestrictqdi;
  CeedQFunction qf_setupgeo, qf_apply;
  CeedOperator op_setupgeo, op_apply;
  CeedVector xcoord, qdata, xceed, yceed;
  CeedInt P, Q, cStart, cEnd, nelem, qdatasize = bpOptions[bpChoice].qdatasize;
  CeedScalar R = 1,                      // radius of the sphere
             l = 1.0/PetscSqrtReal(3.0); // half edge of the inscribed cube

  // CEED bases
  P = degree + 1;
  Q = P + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, topodim, ncompu, P, Q,
                                  bpOptions[bpChoice].qmode, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, topodim, ncompx, 2, Q,
                                  bpOptions[bpChoice].qmode, &basisx);

  // CEED restrictions
  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  ierr = CreateRestrictionPlex(ceed, 2, ncompx, &Erestrictx, dmcoord);
  CHKERRQ(ierr);
  ierr = CreateRestrictionPlex(ceed, P, ncompu, &Erestrictu, dm); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;

  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q, ncompu, ncompu*nelem*Q*Q,
                                   CEED_STRIDES_BACKEND, &Erestrictui);
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
  CeedQFunctionAddInput(qf_setupgeo, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setupgeo, "dx", ncompx*topodim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setupgeo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setupgeo, "qdata", qdatasize, CEED_EVAL_NONE);

  // Set up PDE operator
  CeedInt inscale = bpOptions[bpChoice].inmode==CEED_EVAL_GRAD ? topodim : 1;
  CeedInt outscale = bpOptions[bpChoice].outmode==CEED_EVAL_GRAD ? topodim : 1;
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].apply,
                              bpOptions[bpChoice].applyfname, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", ncompu*inscale,
                        bpOptions[bpChoice].inmode);
  CeedQFunctionAddInput(qf_apply, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", ncompu*outscale,
                         bpOptions[bpChoice].outmode);

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

  // Set up RHS if needed
  if (setup_rhs) {
    CeedQFunction qf_setuprhs;
    CeedOperator op_setuprhs;
    CeedVectorCreate(ceed, nelem*nqpts*ncompu, target);

    // Create the q-function that sets up the RHS and true solution
    CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].setuprhs,
                                bpOptions[bpChoice].setuprhsfname, &qf_setuprhs);
    CeedQFunctionAddInput(qf_setuprhs, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_setuprhs, "qdata", qdatasize, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_setuprhs, "true_soln", ncompu, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_setuprhs, "rhs", ncompu, CEED_EVAL_INTERP);

    // Create the operator that builds the RHS and true solution
    CeedOperatorCreate(ceed, qf_setuprhs, NULL, NULL, &op_setuprhs);
    CeedOperatorSetField(op_setuprhs, "x", Erestrictx, basisx, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setuprhs, "qdata", Erestrictqdi, CEED_BASIS_COLLOCATED,
                         qdata);
    CeedOperatorSetField(op_setuprhs, "true_soln", Erestrictui,
                         CEED_BASIS_COLLOCATED,
                         *target);
    CeedOperatorSetField(op_setuprhs, "rhs", Erestrictu, basisu,
                         CEED_VECTOR_ACTIVE);

    // Set up the libCEED context
    CeedQFunctionContext rhsSetup;
    CeedQFunctionContextCreate(ceed, &rhsSetup);
    CeedScalar rhsSetupData[2] = {R, l};
    CeedQFunctionContextSetData(rhsSetup, CEED_MEM_HOST, CEED_COPY_VALUES,
                                sizeof rhsSetupData, &rhsSetupData);
    CeedQFunctionSetContext(qf_setuprhs, rhsSetup);
    CeedQFunctionContextDestroy(&rhsSetup);

    // Setup RHS and target
    CeedOperatorApply(op_setuprhs, xcoord, rhsceed, CEED_REQUEST_IMMEDIATE);
    CeedVectorTakeArray(rhsceed, CEED_MEM_HOST, NULL);

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
    CeedOperatorCreate(ceed, qf_restrict, NULL, NULL, &op_restrict);
    CeedOperatorSetField(op_restrict, "input", data[i]->Erestrictu,
                         CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_restrict, "output", data[i-1]->Erestrictu,
                         basisctof, CEED_VECTOR_ACTIVE);

    // Save libCEED data required for level
    data[i]->basisctof = basisctof;
    data[i]->op_restrict = op_restrict;

    // Interpolation - Corse to fine
    CeedOperator op_interp;

    // Create the prolongation operator
    CeedOperatorCreate(ceed, qf_prolong, NULL, NULL, &op_interp);
    CeedOperatorSetField(op_interp, "uin", data[i-1]->Erestrictu,
                         basisctof, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_interp, "uout", data[i]->Erestrictu,
                         CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

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

  // Restore PETSc vectors
  CeedVectorTakeArray(user->xceed, CEED_MEM_HOST, NULL);
  CeedVectorTakeArray(user->yceed, CEED_MEM_HOST, NULL);
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

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceedvecc, CEED_MEM_HOST, NULL);
  CeedVectorTakeArray(user->ceedvecf, CEED_MEM_HOST, NULL);
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

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceedvecf, CEED_MEM_HOST, NULL);
  CeedVectorTakeArray(user->ceedvecc, CEED_MEM_HOST, NULL);
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

#endif // setupsphere_h
