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

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>
#include <petscsys.h>
#include <stdbool.h>
#include <string.h>
#include "qfunctions/bps/bp1.h"
#include "qfunctions/bps/bp2.h"
#include "qfunctions/bps/bp3.h"
#include "qfunctions/bps/bp4.h"
#include "qfunctions/bps/common.h"

#if PETSC_VERSION_LT(3,12,0)
#ifdef PETSC_HAVE_CUDA
#include <petsccuda.h>
// Note: With PETSc prior to version 3.12.0, providing the source path to
//       include 'cublas_v2.h' will be needed to use 'petsccuda.h'.
#endif
#endif

#if PETSC_VERSION_LT(3,14,0)
#  define DMPlexGetClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexGetClosureIndices(a,b,c,d,f,g,i)
#  define DMPlexRestoreClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexRestoreClosureIndices(a,b,c,d,f,g,i)
#endif

#if PETSC_VERSION_LT(3,14,0)
#  define DMAddBoundary(a,b,c,d,e,f,g,h,i,j,k,l) DMAddBoundary(a,b,c,d,e,f,g,h,j,k,l)
#endif

static CeedMemType MemTypeP2C(PetscMemType mtype) {
  return PetscMemTypeDevice(mtype) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}
// -----------------------------------------------------------------------------
// PETSc Operator Structs
// -----------------------------------------------------------------------------

// Data for PETSc Matshell
typedef struct UserO_ *UserO;
struct UserO_ {
  MPI_Comm comm;
  DM dm;
  Vec Xloc, Yloc;
  CeedVector xceed, yceed;
  CeedOperator op;
  Ceed ceed;
};

// Data for PETSc Prolong/Restrict Matshells
typedef struct UserProlongRestr_ *UserProlongRestr;
struct UserProlongRestr_ {
  MPI_Comm comm;
  DM dmc, dmf;
  Vec locvecc, locvecf, multvec;
  CeedVector ceedvecc, ceedvecf;
  CeedOperator opprolong, oprestrict;
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
  CeedQFunction qfapply;
  CeedOperator opapply, oprestrict, opprolong;
  CeedVector qdata, xceed, yceed;
};

// -----------------------------------------------------------------------------
// Command Line Options
// -----------------------------------------------------------------------------

// MemType Options
static const char *const memTypes[] = {"host","device","memType",
                                       "CEED_MEM_",0
                                      };

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
    .enforce_bc = PETSC_FALSE,
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
    .enforce_bc = PETSC_FALSE,
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
    .enforce_bc = PETSC_TRUE,
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
    .applyfname = Diff3_loc,
    .errorfname = Error3_loc,
    .inmode = CEED_EVAL_GRAD,
    .outmode = CEED_EVAL_GRAD,
    .qmode = CEED_GAUSS,
    .enforce_bc = PETSC_TRUE,
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
    .enforce_bc = PETSC_TRUE,
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
    .applyfname = Diff3_loc,
    .errorfname = Error3_loc,
    .inmode = CEED_EVAL_GRAD,
    .outmode = CEED_EVAL_GRAD,
    .qmode = CEED_GAUSS_LOBATTO,
    .enforce_bc = PETSC_TRUE,
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
                         (void(*)(void))bpOptions[bpChoice].bcs_func, NULL,
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
  CeedElemRestrictionDestroy(&data->Erestrictqdi);
  CeedQFunctionDestroy(&data->qfapply);
  CeedOperatorDestroy(&data->opapply);
  if (i > 0) {
    CeedOperatorDestroy(&data->opprolong);
    CeedBasisDestroy(&data->basisctof);
    CeedOperatorDestroy(&data->oprestrict);
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
}

// Set up libCEED for a given degree
static int SetupLibceedByDegree(DM dm, Ceed ceed, CeedInt degree, CeedInt dim,
                                CeedInt qextra, PetscInt ncompu, PetscInt gsize,
                                PetscInt xlsize, bpType bpChoice, CeedData data,
                                PetscBool setup_rhs, CeedVector rhsceed,
                                CeedVector *target) {
  int ierr;
  DM dmcoord;
  Vec coords;
  const PetscScalar *coordArray;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictui, Erestrictqdi;
  CeedQFunction qf_setupgeo, qfapply;
  CeedOperator op_setupgeo, opapply;
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

  ierr = CreateRestrictionPlex(ceed, 2, ncompx, &Erestrictx, dmcoord);
  CHKERRQ(ierr);
  ierr = CreateRestrictionPlex(ceed, P, ncompu, &Erestrictu, dm); CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;

  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, ncompu,
                                   ncompu*nelem*Q*Q*Q,
                                   CEED_STRIDES_BACKEND, &Erestrictui);
  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q*Q, qdatasize,
                                   qdatasize*nelem*Q*Q*Q,
                                   CEED_STRIDES_BACKEND, &Erestrictqdi);

  // Element coordinates
  ierr = DMGetCoordinatesLocal(dm, &coords); CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords, &coordArray); CHKERRQ(ierr);

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
                              bpOptions[bpChoice].applyfname, &qfapply);
  CeedQFunctionAddInput(qfapply, "u", ncompu*inscale,
                        bpOptions[bpChoice].inmode);
  CeedQFunctionAddInput(qfapply, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qfapply, "v", ncompu*outscale,
                         bpOptions[bpChoice].outmode);

  // Create the operator that builds the quadrature data for the operator
  CeedOperatorCreate(ceed, qf_setupgeo, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_setupgeo);
  CeedOperatorSetField(op_setupgeo, "dx", Erestrictx, basisx,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setupgeo, "weight", CEED_ELEMRESTRICTION_NONE, basisx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setupgeo, "qdata", Erestrictqdi,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the operator
  CeedOperatorCreate(ceed, qfapply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &opapply);
  CeedOperatorSetField(opapply, "u", Erestrictu, basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opapply, "qdata", Erestrictqdi, CEED_BASIS_COLLOCATED,
                       qdata);
  CeedOperatorSetField(opapply, "v", Erestrictu, basisu, CEED_VECTOR_ACTIVE);

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
    CeedOperatorSetField(op_setuprhs, "x", Erestrictx, basisx,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setuprhs, "dx", Erestrictx, basisx,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setuprhs, "weight", CEED_ELEMRESTRICTION_NONE,
                         basisx, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_setuprhs, "true_soln", Erestrictui,
                         CEED_BASIS_COLLOCATED, *target);
    CeedOperatorSetField(op_setuprhs, "rhs", Erestrictu, basisu,
                         CEED_VECTOR_ACTIVE);

    // Setup RHS and target
    CeedOperatorApply(op_setuprhs, xcoord, rhsceed, CEED_REQUEST_IMMEDIATE);

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
  data->qfapply = qfapply;
  data->opapply = opapply;
  data->qdata = qdata;
  data->xceed = xceed;
  data->yceed = yceed;

  PetscFunctionReturn(0);
}

// Setup libCEED level transfer operator objects
#ifdef multigrid
static PetscErrorCode CeedLevelTransferSetup(Ceed ceed, CeedInt numlevels,
    CeedInt ncompu, bpType bpChoice, CeedData *data, CeedInt *leveldegrees,
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
    CeedOperator oprestrict;

    // Basis
    CeedBasisCreateTensorH1Lagrange(ceed, 3, ncompu, Pc, Pf,
                                    CEED_GAUSS_LOBATTO, &basisctof);

    // Create the restriction operator
    CeedOperatorCreate(ceed, qfrestrict, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &oprestrict);
    CeedOperatorSetField(oprestrict, "input", data[i]->Erestrictu,
                         CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(oprestrict, "output", data[i-1]->Erestrictu,
                         basisctof, CEED_VECTOR_ACTIVE);

    // Save libCEED data required for level
    data[i]->basisctof = basisctof;
    data[i]->oprestrict = oprestrict;

    // Interpolation - Corse to fine
    CeedOperator opprolong;

    // Create the prolongation operator
    CeedOperatorCreate(ceed, qfprolong, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &opprolong);
    CeedOperatorSetField(opprolong, "input", data[i-1]->Erestrictu,
                         basisctof, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(opprolong, "output", data[i]->Erestrictu,
                         CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

    // Save libCEED data required for level
    data[i]->opprolong = opprolong;
  }

  PetscFunctionReturn(0);
}
#endif

// -----------------------------------------------------------------------------
// Mat Shell Functions
// -----------------------------------------------------------------------------
// This function returns the computed diagonal of the operator
static PetscErrorCode MatGetDiag(Mat A, Vec D) {
  PetscErrorCode ierr;
  UserO user;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Compute Diagonal via libCEED
  PetscScalar *x;
  PetscMemType memtype;

  // -- Place PETSc vector in libCEED vector
  ierr = VecGetArrayAndMemType(user->Xloc, &x, &memtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, MemTypeP2C(memtype), CEED_USE_POINTER, x);

  // -- Compute Diagonal
  CeedOperatorLinearAssembleDiagonal(user->op, user->xceed,
                                     CEED_REQUEST_IMMEDIATE);

  // -- Local-to-Global
  CeedVectorTakeArray(user->xceed, MemTypeP2C(memtype), NULL);
  ierr = VecRestoreArrayAndMemType(user->Xloc, &x); CHKERRQ(ierr);
  ierr = VecZeroEntries(D); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->Xloc, ADD_VALUES, D); CHKERRQ(ierr);

  // Cleanup
  ierr = VecZeroEntries(user->Xloc); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
static PetscErrorCode ApplyLocal_Ceed(Vec X, Vec Y, UserO user) {
  PetscErrorCode ierr;
  PetscScalar *x, *y;
  PetscMemType xmemtype, ymemtype;

  PetscFunctionBeginUser;

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->Xloc); CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(user->Xloc, (const PetscScalar **)&x,
                                   &xmemtype); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->Yloc, &y, &ymemtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, MemTypeP2C(xmemtype), CEED_USE_POINTER, x);
  CeedVectorSetArray(user->yceed, MemTypeP2C(ymemtype), CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(user->op, user->xceed, user->yceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->xceed, MemTypeP2C(xmemtype), NULL);
  CeedVectorTakeArray(user->yceed, MemTypeP2C(ymemtype), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->Yloc, &y); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, user->Yloc, ADD_VALUES, Y); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// This function wraps the libCEED operator for a MatShell
static PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserO user;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // libCEED for local action of residual evaluator
  ierr = ApplyLocal_Ceed(X, Y, user); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// This function wraps the libCEED operator for a SNES residual evaluation
#ifdef multigrid
static PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx) {
  PetscErrorCode ierr;
  UserO user = (UserO)ctx;

  PetscFunctionBeginUser;

  // libCEED for local action of residual evaluator
  ierr = ApplyLocal_Ceed(X, Y, user); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};
#endif

// This function uses libCEED to compute the action of the prolongation operator
#ifdef multigrid
static PetscErrorCode MatMult_Prolong(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserProlongRestr user;
  PetscScalar *c, *f;
  PetscMemType cmemtype, fmemtype;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->locvecc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dmc, X, INSERT_VALUES, user->locvecc);
  CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(user->locvecc, (const PetscScalar **)&c,
                                   &cmemtype); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->locvecf, &f, &fmemtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedvecc, MemTypeP2C(cmemtype), CEED_USE_POINTER, c);
  CeedVectorSetArray(user->ceedvecf, MemTypeP2C(fmemtype), CEED_USE_POINTER, f);

  // Apply libCEED operator
  CeedOperatorApply(user->opprolong, user->ceedvecc, user->ceedvecf,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceedvecc, MemTypeP2C(cmemtype), NULL);
  CeedVectorTakeArray(user->ceedvecf, MemTypeP2C(fmemtype), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->locvecc, (const PetscScalar **)&c);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->locvecf, &f); CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(user->locvecf, user->locvecf, user->multvec);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dmf, user->locvecf, ADD_VALUES, Y);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif

// This function uses libCEED to compute the action of the restriction operator
#ifdef multigrid
static PetscErrorCode MatMult_Restrict(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  UserProlongRestr user;
  PetscScalar *c, *f;
  PetscMemType cmemtype, fmemtype;

  PetscFunctionBeginUser;

  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecZeroEntries(user->locvecf); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dmf, X, INSERT_VALUES, user->locvecf);
  CHKERRQ(ierr);

  // Multiplicity
  ierr = VecPointwiseMult(user->locvecf, user->locvecf, user->multvec);
  CHKERRQ(ierr);

  // Setup libCEED vectors
  ierr = VecGetArrayReadAndMemType(user->locvecf, (const PetscScalar **)&f,
                                   &fmemtype); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(user->locvecc, &c, &cmemtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->ceedvecf, MemTypeP2C(fmemtype), CEED_USE_POINTER, f);
  CeedVectorSetArray(user->ceedvecc, MemTypeP2C(cmemtype), CEED_USE_POINTER, c);

  // Apply CEED operator
  CeedOperatorApply(user->oprestrict, user->ceedvecf, user->ceedvecc,
                    CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(user->ceedvecc, MemTypeP2C(cmemtype), NULL);
  CeedVectorTakeArray(user->ceedvecf, MemTypeP2C(fmemtype), NULL);
  ierr = VecRestoreArrayReadAndMemType(user->locvecf, (const PetscScalar **)&f);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(user->locvecc, &c); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dmc, user->locvecc, ADD_VALUES, Y);
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
  PetscMemType memtype;
  CeedVector collocated_error;
  CeedInt length;

  PetscFunctionBeginUser;
  CeedVectorGetLength(target, &length);
  CeedVectorCreate(user->ceed, length, &collocated_error);

  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->Xloc); CHKERRQ(ierr);

  // Setup libCEED vector
  ierr = VecGetArrayReadAndMemType(user->Xloc, (const PetscScalar **)&x,
                                   &memtype); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, MemTypeP2C(memtype), CEED_USE_POINTER, x);

  // Apply libCEED operator
  CeedOperatorApply(op_error, user->xceed, collocated_error,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorTakeArray(user->xceed, MemTypeP2C(memtype), NULL);

  // Restore PETSc vector
  ierr = VecRestoreArrayReadAndMemType(user->Xloc, (const PetscScalar **)&x);
  CHKERRQ(ierr);

  // Reduce max error
  *maxerror = 0;
  const CeedScalar *e;
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &e);
  for (CeedInt i=0; i<length; i++) {
    *maxerror = PetscMax(*maxerror, PetscAbsScalar(e[i]));
  }
  CeedVectorRestoreArrayRead(collocated_error, &e);
  ierr = MPI_Allreduce(MPI_IN_PLACE, maxerror, 1, MPIU_REAL, MPIU_MAX,
                       user->comm); CHKERRQ(ierr);

  // Cleanup
  CeedVectorDestroy(&collocated_error);

  PetscFunctionReturn(0);
}

#endif //setup_h
