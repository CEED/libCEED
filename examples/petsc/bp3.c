//                        libCEED + PETSc Example: BP3
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP3 benchmark problem, see http://ceed.exascaleproject.org/bps.
//
// The code is intentionally "raw", using only low-level communication
// primitives.
//
// Build with:
//
//     make bp3 [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bp3
//     bp3 -ceed /cpu/self
//     bp3 -ceed /gpu/occa
//     bp3 -ceed /cpu/occa
//     bp3 -ceed /omp/occa
//     bp3 -ceed /ocl/occa
//
//TESTARGS -ceed {ceed_resource} -test -degree 3

/// @file
/// Diffusion operator example using PETSc
const char help[] = "Solve CEED BP3 using PETSc\n";

#include <petscksp.h>
#include <ceed.h>
#include <stdbool.h>
#include "bp3.h"

#if PETSC_VERSION_LT(3,11,0)
#  define VecScatterCreateWithData VecScatterCreate
#endif

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
static void GlobalDof(const PetscInt p[3], const PetscInt irank[3],
                      PetscInt degree, const PetscInt melem[3],
                      PetscInt mdof[3]) {
  for (int d=0; d<3; d++)
    mdof[d] = degree*melem[d] + (irank[d] == p[d]-1);
}
static PetscInt GlobalStart(const PetscInt p[3], const PetscInt irank[3],
                            PetscInt degree, const PetscInt melem[3]) {
  PetscInt start = 0;
  // Dumb brute-force is easier to read
  for (PetscInt i=0; i<p[0]; i++) {
    for (PetscInt j=0; j<p[1]; j++) {
      for (PetscInt k=0; k<p[2]; k++) {
        PetscInt mdof[3], ijkrank[] = {i,j,k};
        if (i == irank[0] && j == irank[1] && k == irank[2]) return start;
        GlobalDof(p, ijkrank, degree, melem, mdof);
        start += mdof[0] * mdof[1] * mdof[2];
      }
    }
  }
  return -1;
}
static int CreateRestriction(Ceed ceed, const CeedInt melem[3],
                             CeedInt P, CeedInt ncomp,
                             CeedElemRestriction *Erestrict) {
  const PetscInt Nelem = melem[0]*melem[1]*melem[2];
  PetscInt mdof[3], *idx, *idxp;

  for (int d=0; d<3; d++) mdof[d] = melem[d]*(P-1) + 1;
  idxp = idx = malloc(Nelem*P*P*P*sizeof idx[0]);
  for (CeedInt i=0; i<melem[0]; i++) {
    for (CeedInt j=0; j<melem[1]; j++) {
      for (CeedInt k=0; k<melem[2]; k++,idxp += P*P*P) {
        for (CeedInt ii=0; ii<P; ii++) {
          for (CeedInt jj=0; jj<P; jj++) {
            for (CeedInt kk=0; kk<P; kk++) {
              if (0) { // This is the C-style (i,j,k) ordering that I prefer
                idxp[(ii*P+jj)*P+kk] = (((i*(P-1)+ii)*mdof[1]
                                         + (j*(P-1)+jj))*mdof[2]
                                        + (k*(P-1)+kk));
              } else { // (k,j,i) ordering for consistency with MFEM example
                idxp[ii+P*(jj+P*kk)] = (((i*(P-1)+ii)*mdof[1]
                                         + (j*(P-1)+jj))*mdof[2]
                                        + (k*(P-1)+kk));
              }
            }
          }
        }
      }
    }
  }
  CeedElemRestrictionCreate(ceed, Nelem, P*P*P, mdof[0]*mdof[1]*mdof[2], ncomp,
                            CEED_MEM_HOST, CEED_OWN_POINTER, idx, Erestrict);
  PetscFunctionReturn(0);
}

typedef struct User_ *User;
struct User_ {
  MPI_Comm comm;
  VecScatter ltog;              // Scatter for all entries
  VecScatter ltog0;             // Skip Dirichlet values
  VecScatter gtogD;             // global-to-global; only Dirichlet values
  Vec Xloc, Yloc;
  CeedVector xceed, yceed;
  CeedOperator op;
  CeedVector rho;
  Ceed ceed;
};

// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
static PetscErrorCode MatMult_Diff(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  User user;
  PetscScalar *x, *y;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);
  ierr = VecScatterBegin(user->ltog0, X, user->Xloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog0, X, user->Xloc, INSERT_VALUES,
                       SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Yloc); CHKERRQ(ierr);

  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar**)&x); CHKERRQ(ierr);
  ierr = VecGetArray(user->Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorSetArray(user->yceed, CEED_MEM_HOST, CEED_USE_POINTER, y);

  CeedOperatorApply(user->op, user->xceed, user->yceed,
                    CEED_REQUEST_IMMEDIATE);
  //TODO replace this by SyncArray when available
  const CeedScalar* array;
  ierr = CeedVectorGetArrayRead(user->yceed, CEED_MEM_HOST, &array); CHKERRQ(ierr);
  ierr = CeedVectorRestoreArrayRead(user->yceed, &array); CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar**)&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Yloc, &y); CHKERRQ(ierr);

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

static PetscErrorCode ComputeErrorMax(User user, CeedOperator op_error, Vec X,
                                      CeedVector target, PetscReal *maxerror) {
  PetscErrorCode ierr;
  PetscScalar *x;
  CeedVector collocated_error;
  CeedInt length;

  PetscFunctionBeginUser;
  CeedVectorGetLength(target, &length);
  CeedVectorCreate(user->ceed, length, &collocated_error);
  ierr = VecScatterBegin(user->ltog, X, user->Xloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog, X, user->Xloc, INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar**)&x); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedOperatorApply(op_error, user->xceed, collocated_error,
                    CEED_REQUEST_IMMEDIATE);
  VecRestoreArrayRead(user->Xloc, (const PetscScalar**)&x); CHKERRQ(ierr);

  *maxerror = 0;
  const CeedScalar *e;
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &e);
  for (CeedInt i=0; i<length; i++) {
    *maxerror = PetscMax(*maxerror, PetscAbsScalar(e[i]));
  }
  CeedVectorRestoreArrayRead(collocated_error, &e);
  ierr = MPI_Allreduce(MPI_IN_PLACE, &maxerror,
                       1, MPIU_SCALAR, MPIU_MAX, user->comm); CHKERRQ(ierr);
  CeedVectorDestroy(&collocated_error);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char ceedresource[4096] = "/cpu/self";
  PetscInt degree, qextra, localdof, localelem, melem[3], mdof[3], p[3],
           irank[3], ldof[3], lsize;
  PetscScalar *r;
  PetscBool test_mode;
  PetscMPIInt size, rank;
  VecScatter ltog, ltog0, gtogD;
  Ceed ceed;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi, Erestrictui,
                      Erestrictqdi;
  CeedQFunction qf_setup, qf_diff, qf_error;
  CeedOperator op_setup, op_diff, op_error;
  CeedVector xcoord, rho, rhsceed, target;
  CeedInt P, Q;
  Vec X, Xloc, rhs, rhsloc;
  Mat mat;
  KSP ksp;
  User user;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "CEED BP3 in PETSc", NULL); CHKERRQ(ierr);
  test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, test_mode, &test_mode, NULL); CHKERRQ(ierr);
  degree = test_mode ? 3 : 1;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  qextra = 2;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  localdof = 1000;
  ierr = PetscOptionsInt("-local",
                         "Target number of locally owned degrees of freedom per process",
                         NULL, localdof, &localdof, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Determine size of process grid
  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  Split3(size, p, false);

  // Find a nicely composite number of elements no less than localdof
  for (localelem = PetscMax(1, localdof / (degree*degree*degree)); ;
       localelem++) {
    Split3(localelem, melem, true);
    if (Max3(melem) / Min3(melem) <= 2) break;
  }

  // Find my location in the process grid
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  for (int d=0,rankleft=rank; d<3; d++) {
    const int pstride[3] = {p[1]*p[2], p[2], 1};
    irank[d] = rankleft / pstride[d];
    rankleft -= irank[d] * pstride[d];
  }

  GlobalDof(p, irank, degree, melem, mdof);

  ierr = VecCreate(comm, &X); CHKERRQ(ierr);
  ierr = VecSetSizes(X, mdof[0] * mdof[1] * mdof[2], PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(X); CHKERRQ(ierr);

  if (!test_mode) {
    CeedInt gsize;
    ierr = VecGetSize(X, &gsize); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Global dofs: %D\n", gsize); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Process decomposition: %D %D %D\n",
                       p[0], p[1], p[2]); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Local elements: %D = %D %D %D\n", localelem,
                       melem[0], melem[1], melem[2]); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Owned dofs: %D = %D %D %D\n",
                       mdof[0]*mdof[1]*mdof[2], mdof[0], mdof[1], mdof[2]); CHKERRQ(ierr);
  }

  {
    lsize = 1;
    for (int d=0; d<3; d++) {
      ldof[d] = melem[d]*degree + 1;
      lsize *= ldof[d];
    }
    ierr = VecCreate(PETSC_COMM_SELF, &Xloc); CHKERRQ(ierr);
    ierr = VecSetSizes(Xloc, lsize, PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetUp(Xloc); CHKERRQ(ierr);

    // Create local-to-global scatter
    PetscInt *ltogind, *ltogind0, *locind, l0count;
    IS ltogis, ltogis0, locis;
    PetscInt gstart[2][2][2], gmdof[2][2][2][3];

    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
        for (int k=0; k<2; k++) {
          PetscInt ijkrank[3] = {irank[0]+i, irank[1]+j, irank[2]+k};
          gstart[i][j][k] = GlobalStart(p, ijkrank, degree, melem);
          GlobalDof(p, ijkrank, degree, melem, gmdof[i][j][k]);
        }
      }
    }

    ierr = PetscMalloc1(lsize, &ltogind); CHKERRQ(ierr);
    ierr = PetscMalloc1(lsize, &ltogind0); CHKERRQ(ierr);
    ierr = PetscMalloc1(lsize, &locind); CHKERRQ(ierr);
    l0count = 0;
    for (PetscInt i=0,ir,ii; ir=i>=mdof[0], ii=i-ir*mdof[0], i<ldof[0]; i++) {
      for (PetscInt j=0,jr,jj; jr=j>=mdof[1], jj=j-jr*mdof[1], j<ldof[1]; j++) {
        for (PetscInt k=0,kr,kk; kr=k>=mdof[2], kk=k-kr*mdof[2], k<ldof[2]; k++) {
          PetscInt here = (i*ldof[1]+j)*ldof[2]+k;
          ltogind[here] =
            gstart[ir][jr][kr] + (ii*gmdof[ir][jr][kr][1]+jj)*gmdof[ir][jr][kr][2]+kk;
          if ((irank[0] == 0 && i == 0)
              || (irank[1] == 0 && j == 0)
              || (irank[2] == 0 && k == 0)
              || (irank[0]+1 == p[0] && i+1 == ldof[0])
              || (irank[1]+1 == p[1] && j+1 == ldof[1])
              || (irank[2]+1 == p[2] && k+1 == ldof[2]))
            continue;
          ltogind0[l0count] = ltogind[here];
          locind[l0count++] = here;
        }
      }
    }
    ierr = ISCreateGeneral(comm, lsize, ltogind, PETSC_OWN_POINTER, &ltogis);
    CHKERRQ(ierr);
    ierr = VecScatterCreateWithData(Xloc, NULL, X, ltogis, &ltog); CHKERRQ(ierr);
    CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, l0count, ltogind0, PETSC_OWN_POINTER, &ltogis0);
    CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, l0count, locind, PETSC_OWN_POINTER,
                           &locis);
    CHKERRQ(ierr);
    ierr = VecScatterCreateWithData(Xloc, locis, X, ltogis0, &ltog0); CHKERRQ(ierr);
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
      ierr = VecScatterCreateWithData(X, isD, X, isD, &gtogD); CHKERRQ(ierr);
      ierr = ISDestroy(&isD); CHKERRQ(ierr);
    }
    ierr = ISDestroy(&ltogis); CHKERRQ(ierr);
    ierr = ISDestroy(&ltogis0); CHKERRQ(ierr);
    ierr = ISDestroy(&locis); CHKERRQ(ierr);
  }

  CeedInit(ceedresource, &ceed);
  P = degree + 1;
  Q = P + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, P, Q, CEED_GAUSS, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 3, 2, Q, CEED_GAUSS, &basisx);

  CreateRestriction(ceed, melem, P, 1, &Erestrictu);
  CreateRestriction(ceed, melem, 2, 3, &Erestrictx);
  CeedInt nelem = melem[0]*melem[1]*melem[2];
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q*Q, nelem*Q*Q*Q, 1,
                                    &Erestrictui);
  CeedElemRestrictionCreateIdentity(ceed, nelem, 6*Q*Q*Q, 6*nelem*Q*Q*Q, 1,
                                    &Erestrictqdi);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q*Q, nelem*Q*Q*Q, 1,
                                    &Erestrictxi);
  {
    CeedScalar *xloc;
    CeedInt shape[3] = {melem[0]+1, melem[1]+1, melem[2]+1}, len =
                         shape[0]*shape[1]*shape[2];
    xloc = malloc(len*3*sizeof xloc[0]);
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
    CeedVectorCreate(ceed, len*3, &xcoord);
    CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_OWN_POINTER, xloc);
  }

  // Create the Q-function that builds the diff operator (i.e. computes its
  // quadrature data) and set its context data.
  CeedQFunctionCreateInterior(ceed, 1,
                              Setup, __FILE__ ":Setup", &qf_setup);
  CeedQFunctionAddInput(qf_setup, "x", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup, "dx", 3, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "rho", 6, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup, "true_soln", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup, "rhs", 1, CEED_EVAL_INTERP);

  // Create the Q-function that defines the action of the diff operator.
  CeedQFunctionCreateInterior(ceed, 1,
                              Diff, __FILE__ ":Diff", &qf_diff);
  CeedQFunctionAddInput(qf_diff, "u", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_diff, "rho", 6, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_diff, "v", 1, CEED_EVAL_GRAD);

  // Create the error qfunction
  CeedQFunctionCreateInterior(ceed, 1,
                              Error, __FILE__ ":Error", &qf_error);
  CeedQFunctionAddInput(qf_error, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", 1, CEED_EVAL_NONE);

  // Create the persistent vectors that will be needed in setup
  CeedInt Nqpts, Nelem = melem[0]*melem[1]*melem[2];
  CeedBasisGetNumQuadraturePoints(basisu, &Nqpts);
  CeedVectorCreate(ceed, 6*Nelem*Nqpts, &rho);
  CeedVectorCreate(ceed, Nelem*Nqpts, &target);
  CeedVectorCreate(ceed, lsize, &rhsceed);

  // Create the operator that builds the quadrature data for the diff operator.
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "x", Erestrictx, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", Erestrictxi, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "rho", Erestrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "true_soln", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_setup, "rhs", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, rhsceed);

  // Create the diff operator.
  CeedOperatorCreate(ceed, qf_diff, NULL, NULL, &op_diff);
  CeedOperatorSetField(op_diff, "u", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff, "rho", Erestrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, rho);
  CeedOperatorSetField(op_diff, "v", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, NULL, NULL, &op_error);
  CeedOperatorSetField(op_error, "u", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "error", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);


  // Set up Mat
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);
  user->comm = comm;
  user->ltog = ltog;
  user->ltog0 = ltog0;
  user->gtogD = gtogD;
  user->Xloc = Xloc;
  ierr = VecDuplicate(Xloc, &user->Yloc); CHKERRQ(ierr);
  CeedVectorCreate(ceed, lsize, &user->xceed);
  CeedVectorCreate(ceed, lsize, &user->yceed);
  user->op = op_diff;
  user->rho = rho;
  user->ceed = ceed;

  ierr = MatCreateShell(comm, mdof[0]*mdof[1]*mdof[2], mdof[0]*mdof[1]*mdof[2],
                        PETSC_DECIDE, PETSC_DECIDE, user, &mat); CHKERRQ(ierr);
  ierr = MatShellSetOperation(mat, MATOP_MULT, (void(*)(void))MatMult_Diff);
  CHKERRQ(ierr);
  ierr = MatCreateVecs(mat, &rhs, NULL); CHKERRQ(ierr);

  // Get RHS vector
  ierr = VecDuplicate(Xloc, &rhsloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhsloc); CHKERRQ(ierr);
  ierr = VecGetArray(rhsloc, &r); CHKERRQ(ierr);
  CeedVectorSetArray(rhsceed, CEED_MEM_HOST, CEED_USE_POINTER, r);

  // Setup rho, rhs, and target
  CeedOperatorApply(op_setup, xcoord, rho, CEED_REQUEST_IMMEDIATE);
  //TODO replace this by SyncArray when available
  const CeedScalar* array;
  ierr = CeedVectorGetArrayRead(rho, CEED_MEM_HOST, &array); CHKERRQ(ierr);
  ierr = CeedVectorRestoreArrayRead(rho, &array); CHKERRQ(ierr);
  ierr = CeedVectorGetArrayRead(rhsceed, CEED_MEM_HOST, &array); CHKERRQ(ierr);
  ierr = CeedVectorRestoreArrayRead(rhsceed, &array); CHKERRQ(ierr);
  CeedVectorDestroy(&xcoord);

  // Gather RHS
  ierr = VecRestoreArray(rhsloc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = VecScatterBegin(ltog0, rhsloc, rhs, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(ltog0, rhsloc, rhs, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  CeedVectorDestroy(&rhsceed);

  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    PC pc;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mat, mat); CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
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
      ierr = PetscPrintf(comm, "KSP %s %s iterations %D rnorm %e\n", ksptype,
                         KSPConvergedReasons[reason], its, (double)rnorm); CHKERRQ(ierr);
    }
  }

  {
    PetscReal maxerror;
    ierr = ComputeErrorMax(user, op_error, X, target, &maxerror); CHKERRQ(ierr);
    if (!test_mode || maxerror > 5e-2) {
      ierr = PetscPrintf(comm, "Pointwise error (max) %e\n", (double)maxerror);
      CHKERRQ(ierr);
    }
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
  CeedVectorDestroy(&user->rho);
  CeedVectorDestroy(&target);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_diff);
  CeedOperatorDestroy(&op_error);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictqdi);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_diff);
  CeedQFunctionDestroy(&qf_error);
  CeedBasisDestroy(&basisu);
  CeedBasisDestroy(&basisx);
  CeedDestroy(&ceed);
  ierr = PetscFree(user); CHKERRQ(ierr);
  return PetscFinalize();
}
