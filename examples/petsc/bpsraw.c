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

//                        libCEED + PETSc Example: CEED BPs
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps.
//
// The code is intentionally "raw", using only low-level communication
// primitives.
//
// Build with:
//
//     make bpsraw [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bpsraw -problem bp1
//     bpsraw -problem bp2 -ceed /cpu/self
//     bpsraw -problem bp3 -ceed /gpu/occa
//     bpsraw -problem bp4 -ceed /cpu/occa
//     bpsraw -problem bp5 -ceed /omp/occa
//     bpsraw -problem bp6 -ceed /ocl/occa
//
//TESTARGS -ceed {ceed_resource} -test -problem bp2 -degree 5 -qextra 5

/// @file
/// CEED BPs example using PETSc
/// See bps.c for an implementation using DMPlex unstructured grids.
const char help[] = "Solve CEED BPs using PETSc\n";

#include <stdbool.h>
#include <string.h>
#include <petscksp.h>
#include <ceed.h>
#include "qfunctions/bps/common.h"
#include "qfunctions/bps/bp1.h"
#include "qfunctions/bps/bp2.h"
#include "qfunctions/bps/bp3.h"
#include "qfunctions/bps/bp4.h"

static void Split3(PetscInt size, PetscInt m[3], bool reverse) {
  for (PetscInt d=0,sizeleft=size; d<3; d++) {
    PetscInt try = (PetscInt)PetscCeilReal(PetscPowReal(sizeleft, 1./(3 - d)));
    while (try * (sizeleft / try) != sizeleft) try++;
    m[reverse ? 2-d : d] = try;
    sizeleft /= try;
  }
}

static PetscInt Max3(const PetscInt a[3]) {
  return PetscMax(a[0], PetscMax(a[1], a[2]));
}
static PetscInt Min3(const PetscInt a[3]) {
  return PetscMin(a[0], PetscMin(a[1], a[2]));
}
static void GlobalNodes(const PetscInt p[3], const PetscInt irank[3],
                        PetscInt degree, const PetscInt melem[3],
                        PetscInt mnodes[3]) {
  for (int d=0; d<3; d++)
    mnodes[d] = degree*melem[d] + (irank[d] == p[d]-1);
}
static PetscInt GlobalStart(const PetscInt p[3], const PetscInt irank[3],
                            PetscInt degree, const PetscInt melem[3]) {
  PetscInt start = 0;
  // Dumb brute-force is easier to read
  for (PetscInt i=0; i<p[0]; i++) {
    for (PetscInt j=0; j<p[1]; j++) {
      for (PetscInt k=0; k<p[2]; k++) {
        PetscInt mnodes[3], ijkrank[] = {i,j,k};
        if (i == irank[0] && j == irank[1] && k == irank[2]) return start;
        GlobalNodes(p, ijkrank, degree, melem, mnodes);
        start += mnodes[0] * mnodes[1] * mnodes[2];
      }
    }
  }
  return -1;
}
static int CreateRestriction(Ceed ceed, const CeedInt melem[3],
                             CeedInt P, CeedInt ncomp,
                             CeedElemRestriction *Erestrict) {
  const PetscInt nelem = melem[0]*melem[1]*melem[2];
  PetscInt mnodes[3], *idx, *idxp;

  // Get indicies
  for (int d=0; d<3; d++) mnodes[d] = melem[d]*(P-1) + 1;
  idxp = idx = malloc(nelem*P*P*P*sizeof idx[0]);
  for (CeedInt i=0; i<melem[0]; i++)
    for (CeedInt j=0; j<melem[1]; j++)
      for (CeedInt k=0; k<melem[2]; k++,idxp += P*P*P)
        for (CeedInt ii=0; ii<P; ii++)
          for (CeedInt jj=0; jj<P; jj++)
            for (CeedInt kk=0; kk<P; kk++) {
              if (0) { // This is the C-style (i,j,k) ordering that I prefer
                idxp[(ii*P+jj)*P+kk] = (((i*(P-1)+ii)*mnodes[1]
                                         + (j*(P-1)+jj))*mnodes[2]
                                        + (k*(P-1)+kk));
              } else { // (k,j,i) ordering for consistency with MFEM example
                idxp[ii+P*(jj+P*kk)] = (((i*(P-1)+ii)*mnodes[1]
                                         + (j*(P-1)+jj))*mnodes[2]
                                        + (k*(P-1)+kk));
              }
            }

  // Setup CEED restriction
  CeedElemRestrictionCreate(ceed, nelem, P*P*P, mnodes[0]*mnodes[1]*mnodes[2],
                            ncomp, CEED_MEM_HOST, CEED_OWN_POINTER, idx,
                            Erestrict);

  PetscFunctionReturn(0);
}

// Data for PETSc
typedef struct User_ *User;
struct User_ {
  MPI_Comm comm;
  VecScatter ltog;              // Scatter for all entries
  VecScatter ltog0;             // Skip Dirichlet values
  VecScatter gtogD;             // global-to-global; only Dirichlet values
  Vec Xloc, Yloc;
  CeedVector xceed, yceed;
  CeedOperator op;
  CeedVector qdata;
  Ceed ceed;
};

// BP Options
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
    .qmode = CEED_GAUSS
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
    .qmode = CEED_GAUSS
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
    .qmode = CEED_GAUSS_LOBATTO
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
    .qmode = CEED_GAUSS_LOBATTO
  }
};

// This function uses libCEED to compute the action of the mass matrix
static PetscErrorCode MatMult_Mass(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  User user;
  PetscScalar *x, *y;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);
  ierr = VecScatterBegin(user->ltog, X, user->Xloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog, X, user->Xloc, INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Yloc); CHKERRQ(ierr);

  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecGetArray(user->Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorSetArray(user->yceed, CEED_MEM_HOST, CEED_USE_POINTER, y);

  CeedOperatorApply(user->op, user->xceed, user->yceed,
                    CEED_REQUEST_IMMEDIATE);
  ierr = CeedVectorSyncArray(user->yceed, CEED_MEM_HOST); CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Yloc, &y); CHKERRQ(ierr);

  if (Y) {
    ierr = VecZeroEntries(Y); CHKERRQ(ierr);
    ierr = VecScatterBegin(user->ltog, user->Yloc, Y, ADD_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(user->ltog, user->Yloc, Y, ADD_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
static PetscErrorCode MatMult_Diff(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  User user;
  PetscScalar *x, *y;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);

  // Global-to-local
  ierr = VecScatterBegin(user->ltog0, X, user->Xloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog0, X, user->Xloc, INSERT_VALUES,
                       SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Yloc); CHKERRQ(ierr);

  // Setup CEED vectors
  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecGetArray(user->Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorSetArray(user->yceed, CEED_MEM_HOST, CEED_USE_POINTER, y);

  // Apply CEED operator
  CeedOperatorApply(user->op, user->xceed, user->yceed,
                    CEED_REQUEST_IMMEDIATE);
  ierr = CeedVectorSyncArray(user->yceed, CEED_MEM_HOST); CHKERRQ(ierr);

  // Restore PETSc vectors
  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Yloc, &y); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  ierr = VecScatterBegin(user->gtogD, X, Y, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(user->gtogD, X, Y, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterBegin(user->ltog0, user->Yloc, Y, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog0, user->Yloc, Y, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// This function calculates the error in the final solution
static PetscErrorCode ComputeErrorMax(User user, CeedOperator op_error, Vec X,
                                      CeedVector target, PetscReal *maxerror) {
  PetscErrorCode ierr;
  PetscScalar *x;
  CeedVector collocated_error;
  CeedInt length;

  PetscFunctionBeginUser;
  CeedVectorGetLength(target, &length);
  CeedVectorCreate(user->ceed, length, &collocated_error);

  // Global-to-local
  ierr = VecScatterBegin(user->ltog, X, user->Xloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog, X, user->Xloc, INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);

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

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char ceedresource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt degree, qextra, localnodes, localelem, melem[3], mnodes[3], p[3],
           irank[3], lnodes[3], lsize, ncompu = 1;
  PetscScalar *r;
  PetscBool test_mode, benchmark_mode, write_solution;
  PetscMPIInt size, rank;
  Vec X, Xloc, rhs, rhsloc;
  Mat mat;
  KSP ksp;
  VecScatter ltog, ltog0, gtogD;
  User user;
  Ceed ceed;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi, Erestrictui,
                      Erestrictqdi;
  CeedQFunction qf_setupgeo, qf_setuprhs, qf_apply, qf_error;
  CeedOperator op_setupgeo, op_setuprhs, op_apply, op_error;
  CeedVector xcoord, qdata, rhsceed, target;
  CeedInt P, Q;
  const CeedInt dim = 3, ncompx = 3;
  bpType bpChoice;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL); CHKERRQ(ierr);
  bpChoice = CEED_BP1;
  ierr = PetscOptionsEnum("-problem",
                          "CEED benchmark problem to solve", NULL,
                          bpTypes, (PetscEnum)bpChoice, (PetscEnum *)&bpChoice,
                          NULL); CHKERRQ(ierr);
  ncompu = bpOptions[bpChoice].ncompu;
  test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, test_mode, &test_mode, NULL); CHKERRQ(ierr);
  benchmark_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-benchmark",
                          "Benchmarking mode (prints benchmark statistics)",
                          NULL, benchmark_mode, &benchmark_mode, NULL);
  CHKERRQ(ierr);
  write_solution = PETSC_FALSE;
  ierr = PetscOptionsBool("-write_solution",
                          "Write solution for visualization",
                          NULL, write_solution, &write_solution, NULL);
  CHKERRQ(ierr);
  degree = test_mode ? 3 : 1;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  qextra = bpOptions[bpChoice].qextra;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  localnodes = 1000;
  ierr = PetscOptionsInt("-local",
                         "Target number of locally owned nodes per process",
                         NULL, localnodes, &localnodes, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  P = degree + 1;
  Q = P + qextra;

  // Determine size of process grid
  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  Split3(size, p, false);

  // Find a nicely composite number of elements no less than localnodes
  for (localelem = PetscMax(1, localnodes / (degree*degree*degree)); ;
       localelem++) {
    Split3(localelem, melem, true);
    if (Max3(melem) / Min3(melem) <= 2) break;
  }

  // Find my location in the process grid
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  for (int d=0,rankleft=rank; d<dim; d++) {
    const int pstride[3] = {p[1] *p[2], p[2], 1};
    irank[d] = rankleft / pstride[d];
    rankleft -= irank[d] * pstride[d];
  }

  GlobalNodes(p, irank, degree, melem, mnodes);

  // Setup global vector
  ierr = VecCreate(comm, &X); CHKERRQ(ierr);
  ierr = VecSetSizes(X, mnodes[0]*mnodes[1]*mnodes[2]*ncompu, PETSC_DECIDE);
  CHKERRQ(ierr);
  ierr = VecSetUp(X); CHKERRQ(ierr);

  // Set up libCEED
  CeedInit(ceedresource, &ceed);

  // Print summary
  if (!test_mode) {
    CeedInt gsize;
    ierr = VecGetSize(X, &gsize); CHKERRQ(ierr);
    const char *usedresource;
    CeedGetResource(ceed, &usedresource);
    ierr = PetscPrintf(comm,
                       "\n-- CEED Benchmark Problem %d -- libCEED + PETSc --\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)       : %d\n"
                       "    Number of 1D Quadrature Points (q) : %d\n"
                       "    Global nodes                       : %D\n"
                       "    Process Decomposition              : %D %D %D\n"
                       "    Local Elements                     : %D = %D %D %D\n"
                       "    Owned nodes                        : %D = %D %D %D\n",
                       bpChoice+1, usedresource, P, Q,  gsize/ncompu, p[0],
                       p[1], p[2], localelem, melem[0], melem[1], melem[2],
                       mnodes[0]*mnodes[1]*mnodes[2], mnodes[0], mnodes[1],
                       mnodes[2]); CHKERRQ(ierr);
  }

  {
    lsize = 1;
    for (int d=0; d<dim; d++) {
      lnodes[d] = melem[d]*degree + 1;
      lsize *= lnodes[d];
    }
    ierr = VecCreate(PETSC_COMM_SELF, &Xloc); CHKERRQ(ierr);
    ierr = VecSetSizes(Xloc, lsize*ncompu, PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetUp(Xloc); CHKERRQ(ierr);

    // Create local-to-global scatter
    PetscInt *ltogind, *ltogind0, *locind, l0count;
    IS ltogis, ltogis0, locis;
    PetscInt gstart[2][2][2], gmnodes[2][2][2][dim];

    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
        for (int k=0; k<2; k++) {
          PetscInt ijkrank[3] = {irank[0]+i, irank[1]+j, irank[2]+k};
          gstart[i][j][k] = GlobalStart(p, ijkrank, degree, melem);
          GlobalNodes(p, ijkrank, degree, melem, gmnodes[i][j][k]);
        }
      }
    }

    ierr = PetscMalloc1(lsize, &ltogind); CHKERRQ(ierr);
    ierr = PetscMalloc1(lsize, &ltogind0); CHKERRQ(ierr);
    ierr = PetscMalloc1(lsize, &locind); CHKERRQ(ierr);
    l0count = 0;
    for (PetscInt i=0,ir,ii; ir=i>=mnodes[0], ii=i-ir*mnodes[0], i<lnodes[0]; i++)
      for (PetscInt j=0,jr,jj; jr=j>=mnodes[1], jj=j-jr*mnodes[1], j<lnodes[1]; j++)
        for (PetscInt k=0,kr,kk; kr=k>=mnodes[2], kk=k-kr*mnodes[2], k<lnodes[2]; k++) {
          PetscInt here = (i*lnodes[1]+j)*lnodes[2]+k;
          ltogind[here] =
            gstart[ir][jr][kr] + (ii*gmnodes[ir][jr][kr][1]+jj)*gmnodes[ir][jr][kr][2]+kk;
          if ((irank[0] == 0 && i == 0)
              || (irank[1] == 0 && j == 0)
              || (irank[2] == 0 && k == 0)
              || (irank[0]+1 == p[0] && i+1 == lnodes[0])
              || (irank[1]+1 == p[1] && j+1 == lnodes[1])
              || (irank[2]+1 == p[2] && k+1 == lnodes[2]))
            continue;
          ltogind0[l0count] = ltogind[here];
          locind[l0count++] = here;
        }
    ierr = ISCreateBlock(comm, ncompu, lsize, ltogind, PETSC_OWN_POINTER,
                         &ltogis); CHKERRQ(ierr);
    ierr = VecScatterCreate(Xloc, NULL, X, ltogis, &ltog); CHKERRQ(ierr);
    CHKERRQ(ierr);
    ierr = ISCreateBlock(comm, ncompu, l0count, ltogind0, PETSC_OWN_POINTER,
                         &ltogis0); CHKERRQ(ierr);
    ierr = ISCreateBlock(comm, ncompu, l0count, locind, PETSC_OWN_POINTER,
                         &locis); CHKERRQ(ierr);
    ierr = VecScatterCreate(Xloc, locis, X, ltogis0, &ltog0); CHKERRQ(ierr);
    {
      // Create global-to-global scatter for Dirichlet values (everything not in
      // ltogis0, which is the range of ltog0)
      PetscInt xstart, xend, *indD, countD = 0;
      IS isD;
      const PetscScalar *x;
      ierr = VecZeroEntries(Xloc); CHKERRQ(ierr);
      ierr = VecSet(X, 1.0); CHKERRQ(ierr);
      ierr = VecScatterBegin(ltog0, Xloc, X, INSERT_VALUES, SCATTER_FORWARD);
      CHKERRQ(ierr);
      ierr = VecScatterEnd(ltog0, Xloc, X, INSERT_VALUES, SCATTER_FORWARD);
      CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(X, &xstart, &xend); CHKERRQ(ierr);
      ierr = PetscMalloc1(xend-xstart, &indD); CHKERRQ(ierr);
      ierr = VecGetArrayRead(X, &x); CHKERRQ(ierr);
      for (PetscInt i=0; i<xend-xstart; i++) {
        if (x[i] == 1.) indD[countD++] = xstart + i;
      }
      ierr = VecRestoreArrayRead(X, &x); CHKERRQ(ierr);
      ierr = ISCreateGeneral(comm, countD, indD, PETSC_COPY_VALUES, &isD);
      CHKERRQ(ierr);
      ierr = PetscFree(indD); CHKERRQ(ierr);
      ierr = VecScatterCreate(X, isD, X, isD, &gtogD); CHKERRQ(ierr);
      ierr = ISDestroy(&isD); CHKERRQ(ierr);
    }
    ierr = ISDestroy(&ltogis); CHKERRQ(ierr);
    ierr = ISDestroy(&ltogis0); CHKERRQ(ierr);
    ierr = ISDestroy(&locis); CHKERRQ(ierr);
  }

  // CEED bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompu, P, Q,
                                  bpOptions[bpChoice].qmode, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, Q,
                                  bpOptions[bpChoice].qmode, &basisx);

  // CEED restrictions
  CreateRestriction(ceed, melem, P, ncompu, &Erestrictu);
  CreateRestriction(ceed, melem, 2, dim, &Erestrictx);
  CeedInt nelem = melem[0]*melem[1]*melem[2];
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q*Q, nelem*Q*Q*Q, ncompu,
                                    &Erestrictui);
  CeedElemRestrictionCreateIdentity(ceed, nelem,
                                    Q*Q*Q,
                                    nelem*Q*Q*Q,
                                    bpOptions[bpChoice].qdatasize, &Erestrictqdi);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q*Q, nelem*Q*Q*Q, 1,
                                    &Erestrictxi);
  {
    CeedScalar *xloc;
    CeedInt shape[3] = {melem[0]+1, melem[1]+1, melem[2]+1}, len =
                         shape[0]*shape[1]*shape[2];
    xloc = malloc(len*ncompx*sizeof xloc[0]);
    for (CeedInt i=0; i<shape[0]; i++) {
      for (CeedInt j=0; j<shape[1]; j++) {
        for (CeedInt k=0; k<shape[2]; k++) {
          xloc[((i*shape[1]+j)*shape[2]+k) + 0*len] = 1.*(irank[0]*melem[0]+i) /
              (p[0]*melem[0]);
          xloc[((i*shape[1]+j)*shape[2]+k) + 1*len] = 1.*(irank[1]*melem[1]+j) /
              (p[1]*melem[1]);
          xloc[((i*shape[1]+j)*shape[2]+k) + 2*len] = 1.*(irank[2]*melem[2]+k) /
              (p[2]*melem[2]);
        }
      }
    }
    CeedVectorCreate(ceed, len*ncompx, &xcoord);
    CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_OWN_POINTER, xloc);
  }

  // Create the Qfunction that builds the operator quadrature data
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].setupgeo,
                              bpOptions[bpChoice].setupgeofname, &qf_setupgeo);
  CeedQFunctionAddInput(qf_setupgeo, "dx", ncompx*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setupgeo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setupgeo, "qdata", bpOptions[bpChoice].qdatasize,
                         CEED_EVAL_NONE);

  // Create the Qfunction that sets up the RHS and true solution
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].setuprhs,
                              bpOptions[bpChoice].setuprhsfname, &qf_setuprhs);
  CeedQFunctionAddInput(qf_setuprhs, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setuprhs, "dx", ncompx*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setuprhs, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setuprhs, "true_soln", ncompu, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setuprhs, "rhs", ncompu, CEED_EVAL_INTERP);

  // Set up PDE operator
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].apply,
                              bpOptions[bpChoice].applyfname, &qf_apply);
  // Add inputs and outputs
  CeedInt inscale = bpOptions[bpChoice].inmode==CEED_EVAL_GRAD ? 3 : 1;
  CeedInt outscale = bpOptions[bpChoice].outmode==CEED_EVAL_GRAD ? 3 : 1;
  CeedQFunctionAddInput(qf_apply, "u", ncompu*inscale,
                        bpOptions[bpChoice].inmode);
  CeedQFunctionAddInput(qf_apply, "qdata", bpOptions[bpChoice].qdatasize,
                        CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", ncompu*outscale,
                         bpOptions[bpChoice].outmode);

  // Create the error qfunction
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].error,
                              bpOptions[bpChoice].errorfname, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", ncompu, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", ncompu, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", ncompu, CEED_EVAL_NONE);

  // Create the persistent vectors that will be needed in setup
  CeedInt nqpts;
  CeedBasisGetNumQuadraturePoints(basisu, &nqpts);
  CeedVectorCreate(ceed, bpOptions[bpChoice].qdatasize*nelem*nqpts, &qdata);
  CeedVectorCreate(ceed, nelem*nqpts*ncompu, &target);
  CeedVectorCreate(ceed, lsize*ncompu, &rhsceed);

  // Create the operator that builds the quadrature data for the ceed operator
  CeedOperatorCreate(ceed, qf_setupgeo, NULL, NULL, &op_setupgeo);
  CeedOperatorSetField(op_setupgeo, "dx", Erestrictx, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setupgeo, "weight", Erestrictxi, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setupgeo, "qdata", Erestrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the operator that builds the RHS and true solution
  CeedOperatorCreate(ceed, qf_setuprhs, NULL, NULL, &op_setuprhs);
  CeedOperatorSetField(op_setuprhs, "x", Erestrictx, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setuprhs, "dx", Erestrictx, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setuprhs, "weight", Erestrictxi, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setuprhs, "true_soln", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_setuprhs, "rhs", Erestrictu, CEED_TRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);

  // Create the mass or diff operator
  CeedOperatorCreate(ceed, qf_apply, NULL, NULL, &op_apply);
  CeedOperatorSetField(op_apply, "u", Erestrictu, CEED_TRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata", Erestrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_apply, "v", Erestrictu, CEED_TRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, NULL, NULL, &op_error);
  CeedOperatorSetField(op_error, "u", Erestrictu, CEED_TRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "error", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Set up Mat
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);
  user->comm = comm;
  user->ltog = ltog;
  if (bpChoice != CEED_BP1 && bpChoice != CEED_BP2) {
    user->ltog0 = ltog0;
    user->gtogD = gtogD;
  }
  user->Xloc = Xloc;
  ierr = VecDuplicate(Xloc, &user->Yloc); CHKERRQ(ierr);
  CeedVectorCreate(ceed, lsize*ncompu, &user->xceed);
  CeedVectorCreate(ceed, lsize*ncompu, &user->yceed);
  user->op = op_apply;
  user->qdata = qdata;
  user->ceed = ceed;

  ierr = MatCreateShell(comm, mnodes[0]*mnodes[1]*mnodes[2]*ncompu,
                        mnodes[0]*mnodes[1]*mnodes[2]*ncompu,
                        PETSC_DECIDE, PETSC_DECIDE, user, &mat); CHKERRQ(ierr);
  if (bpChoice == CEED_BP1 || bpChoice == CEED_BP2) {
    ierr = MatShellSetOperation(mat, MATOP_MULT, (void(*)(void))MatMult_Mass);
    CHKERRQ(ierr);
  } else {
    ierr = MatShellSetOperation(mat, MATOP_MULT, (void(*)(void))MatMult_Diff);
    CHKERRQ(ierr);
  }
  ierr = MatCreateVecs(mat, &rhs, NULL); CHKERRQ(ierr);

  // Get RHS vector
  ierr = VecDuplicate(Xloc, &rhsloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhsloc); CHKERRQ(ierr);
  ierr = VecGetArray(rhsloc, &r); CHKERRQ(ierr);
  CeedVectorSetArray(rhsceed, CEED_MEM_HOST, CEED_USE_POINTER, r);

  // Setup qdata, rhs, and target
  CeedOperatorApply(op_setupgeo, xcoord, qdata, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setuprhs, xcoord, rhsceed, CEED_REQUEST_IMMEDIATE);
  ierr = CeedVectorSyncArray(rhsceed, CEED_MEM_HOST); CHKERRQ(ierr);
  CeedVectorDestroy(&xcoord);

  // Gather RHS
  ierr = VecRestoreArray(rhsloc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = VecScatterBegin(ltog, rhsloc, rhs, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(ltog, rhsloc, rhs, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  CeedVectorDestroy(&rhsceed);

  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    PC pc;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    if (bpChoice == CEED_BP1 || bpChoice == CEED_BP2) {
      ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
      ierr = PCJacobiSetType(pc, PC_JACOBI_ROWSUM); CHKERRQ(ierr);
    } else {
      ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
    }
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mat, mat); CHKERRQ(ierr);
  // First run, if benchmarking
  if (benchmark_mode) {
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1);
    CHKERRQ(ierr);
    my_rt_start = MPI_Wtime();
    ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
    my_rt = MPI_Wtime() - my_rt_start;
    ierr = MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, comm);
    CHKERRQ(ierr);
    // Set maxits based on first iteration timing
    if (my_rt > 0.02) {
      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 5);
      CHKERRQ(ierr);
    } else {
      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 20);
      CHKERRQ(ierr);
    }
  }
  // Timed solve
  ierr = PetscBarrier((PetscObject)ksp); CHKERRQ(ierr);
  my_rt_start = MPI_Wtime();
  ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
  my_rt = MPI_Wtime() - my_rt_start;
  {
    KSPType ksptype;
    KSPConvergedReason reason;
    PetscReal rnorm;
    PetscInt its;
    ierr = KSPGetType(ksp, &ksptype); CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
    ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      ierr = PetscPrintf(comm,
                         "  KSP:\n"
                         "    KSP Type                           : %s\n"
                         "    KSP Convergence                    : %s\n"
                         "    Total KSP Iterations               : %D\n"
                         "    Final rnorm                        : %e\n",
                         ksptype, KSPConvergedReasons[reason], its,
                         (double)rnorm); CHKERRQ(ierr);
    }
    if (benchmark_mode && (!test_mode)) {
      CeedInt gsize;
      ierr = VecGetSize(X, &gsize); CHKERRQ(ierr);
      ierr = MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
      CHKERRQ(ierr);
      ierr = MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, comm);
      CHKERRQ(ierr);
      ierr = PetscPrintf(comm,
                         "  Performance:\n"
                         "    CG Solve Time                      : %g (%g) sec\n"
                         "    DoFs/Sec in CG                     : %g (%g) million\n",
                         rt_max, rt_min, 1e-6*gsize*its/rt_max,
                         1e-6*gsize*its/rt_min); CHKERRQ(ierr);
    }
  }

  {
    PetscReal maxerror;
    ierr = ComputeErrorMax(user, op_error, X, target, &maxerror); CHKERRQ(ierr);
    PetscReal tol = (bpChoice == CEED_BP1 || bpChoice == CEED_BP2) ? 5e-3 : 5e-2;
    if (!test_mode || maxerror > tol) {
      ierr = PetscPrintf(comm,
                         "    Pointwise Error (max)              : %e\n",
                         (double)maxerror); CHKERRQ(ierr);
    }
  }

  if (write_solution) {
    PetscViewer vtkviewersoln;

    ierr = PetscViewerCreate(comm, &vtkviewersoln); CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewersoln, PETSCVIEWERVTK); CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewersoln, "solution.vtk"); CHKERRQ(ierr);
    ierr = VecView(X, vtkviewersoln); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewersoln); CHKERRQ(ierr);
  }

  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhsloc); CHKERRQ(ierr);
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Xloc); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Yloc); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ltog); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ltog0); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&gtogD); CHKERRQ(ierr);
  ierr = MatDestroy(&mat); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

  CeedVectorDestroy(&user->xceed);
  CeedVectorDestroy(&user->yceed);
  CeedVectorDestroy(&user->qdata);
  CeedVectorDestroy(&target);
  CeedOperatorDestroy(&op_setupgeo);
  CeedOperatorDestroy(&op_setuprhs);
  CeedOperatorDestroy(&op_apply);
  CeedOperatorDestroy(&op_error);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedElemRestrictionDestroy(&Erestrictqdi);
  CeedQFunctionDestroy(&qf_setupgeo);
  CeedQFunctionDestroy(&qf_setuprhs);
  CeedQFunctionDestroy(&qf_apply);
  CeedQFunctionDestroy(&qf_error);
  CeedBasisDestroy(&basisu);
  CeedBasisDestroy(&basisx);
  CeedDestroy(&ceed);
  ierr = PetscFree(user); CHKERRQ(ierr);
  return PetscFinalize();
}
