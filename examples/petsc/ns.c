//                        libCEED + PETSc Example: Navier-Stokes
//
// This example demonstrates a simple usage of libCEED with PETSc to solve a
// Navier-Stokes problem.
//
// The code is intentionally "raw", using only low-level communication
// primitives.
//
// Build with:
//
//     make ns [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     ns
//     ns -ceed /cpu/self
//     ns -ceed /gpu/occa
//     ns -ceed /cpu/occa
//     ns -ceed /omp/occa
//     ns -ceed /ocl/occa
//
const char help[] = "Solve Navier-Stokes using PETSc and libCEED\n";

#include <petscksp.h>
#include <ceed.h>
#include <stdbool.h>
#include "ns.h"

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

typedef struct UserRHS_ *UserRHS;
struct UserRHS_ {
  MPI_Comm comm;
  VecScatter ltog;
  Vec Qloc, Gloc;
  CeedVector qceed, gceed;
  CeedOperator op;
  CeedVector qdata;
  Ceed ceed;
};

typedef struct UserLHS_ *UserLHS;
struct UserLHS_ {
  MPI_Comm comm;
  VecScatter ltog;
  Vec Qloc, dQloc, Floc;
  PetscScalar lsize;
};

// This is the RHS of the DAE, given as F(t,u,u_t) = G(t,u)
// This function takes in a state vector Q and writes into G
static PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *user) {
  PetscErrorCode ierr;
  UserRHS *user = (UserRHS*)user;
  PetscScalar *q, *g;

  // Global to local
  PetscFunctionBeginUser;
  ierr = VecScatterBegin(user->ltog, Q, user->Qloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog, Q, user->Qloc, INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Gloc); CHKERRQ(ierr);

  // Ceed Vectors
  ierr = VecGetArrayRead(user->Qloc, (const PetscScalar**)&q); CHKERRQ(ierr);
  ierr = VecGetArray(user->Gloc, &g); CHKERRQ(ierr);
  CeedVectorSetArray(user->qceed, CEED_MEM_HOST, CEED_USE_POINTER, q);
  CeedVectorSetArray(user->gceed, CEED_MEM_HOST, CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op, user->qceed, user->gceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  ierr = VecRestoreArrayRead(user->Qloc, (const PetscScalar**)&q); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Gloc, &g); CHKERRQ(ierr);

  // Local to global
  ierr = VecZeroEntries(G); CHKERRQ(ierr);
  ierr = VecScatterBegin(user->ltog, user->Gloc, G, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog, user->Gloc, G, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

// This is the LHS of the DAE, given as F(t,u,u_t) = G(t,u)
// This function takes in state vectors Q and dQ and writes into F
static PetscErrorCode LHS_NS(TS ts, PetscReal t, Vec Q, Vec dQ, Vec F, void *user) {
  PetscErrorCode ierr;
  User *user = (User*)user;
  PetscScalar *q, *dq, *f, lsize = user->lsize;

  // Global to local
  PetscFunctionBeginUser;
  ierr = VecScatterBegin(user->ltog, Q, user->Qloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog, Q, user->Qloc, INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecScatterBegin(user->ltog, dQ, user->dQloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog, dQ, user->dQloc, INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Floc); CHKERRQ(ierr);

  // Create LHS
  ierr = VecGetArrayRead(user->Qloc, (const PetscScalar**)&q); CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->dQloc, (const PetscScalar**)&dq); CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->Floc, (const PetscScalar**)&f); CHKERRQ(ierr);
  for (PetscInt i = 0; i < lsize; i++) {
    f[0*lsize + i] = dq[0*lsize + i];
    f[1*lsize + i] =  q[1*lsize + i] * dq[0*lsize + i] +
                     dq[1*lsize + i] *  q[0*lsize + i];
    f[2*lsize + i] =  q[2*lsize + i] * dq[0*lsize + i] +
                     dq[2*lsize + i] *  q[0*lsize + i];
    f[3*lsize + i] =  q[3*lsize + i] * dq[0*lsize + i] +
                     dq[3*lsize + i] *  q[0*lsize + i];
    f[4*lsize + i] =  q[4*lsize + i] * dq[0*lsize + i] +
                     dq[4*lsize + i] *  q[0*lsize + i];
  }

  // Restore vectors
  ierr = VecRestoreArrayRead(user->Qloc, (const PetscScalar**)&q); CHKERRQ(ierr); 
  ierr = VecRestoreArrayRead(user->dQloc, (const PetscScalar**)&dq); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Floc, &f); CHKERRQ(ierr);

  // Local to global
  ierr = VecZeroEntries(F); CHKERRQ(ierr);
  ierr = VecScatterBegin(user->ltog, user->Floc, F, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog, user->Floc, F, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char ceedresource[4096] = "/cpu/self";
  PetscInt degree, qextra, localdof, localelem, melem[3], mdof[3], p[3],
           irank[3], ldof[3], lsize, lsizeLHS, gsize;
  PetscScalar *r;
  PetscMPIInt size, rank;
  VecScatter ltog, ltogLHS;
  Ceed ceed;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi, Erestrictui,
                      Erestrictqdi;
  CeedQFunction qf_setup, qf_ns;
  CeedOperator op_setup, op_ns;
  CeedVector xcoord, qdata;
  CeedInt degP, degQ;
  CeedScalar cxt[5];
  Vec Q, Qloc, QlocLHS;
  TS ts;
  UserRHS userRHS;
  UserLHS userLHS;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "Navier-Stokes in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);
  degree = 3;
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

  ierr = PetscPrintf(comm, "Process decomposition: %D %D %D\n",
                     p[0], p[1], p[2]); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Local elements: %D = %D %D %D\n", localelem,
                     melem[0], melem[1], melem[2]); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Owned dofs: %D = %D %D %D\n",
                     mdof[0]*mdof[1]*mdof[2], mdof[0], mdof[1], mdof[2]); CHKERRQ(ierr);

  {
    ierr = VecCreate(comm, &Q); CHKERRQ(ierr);
    ierr = VecSetSizes(Q, 5*(mdof[0]-1)*(mdof[1]-1)*(mdof[2]-1), PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetUp(Q); CHKERRQ(ierr);

    lsize = 1;
    gsize = 1;
    for (int d=0; d<3; d++) {
      ldof[d] = melem[d]*degree + 1;
      lsize *= ldof[d];
      gsize *= mdof[i];
    }
    ierr = VecCreate(PETSC_COMM_SELF, &Qloc); CHKERRQ(ierr);
    ierr = VecSetSizes(Qloc, 5*lsize, PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetUp(Qloc); CHKERRQ(ierr);

    // Create local-to-global scatter for RHS
    PetscInt *ltogind;
    IS ltogis;
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

    ierr = PetscMalloc1(5*lsize, &ltogind); CHKERRQ(ierr);
    for (PetscInt i=0,ir,ii; ir=i>=mdof[0], ii=i-ir*mdof[0], i<ldof[0]; i++) {
      for (PetscInt j=0,jr,jj; jr=j>=mdof[1], jj=j-jr*mdof[1], j<ldof[1]; j++) {
        for (PetscInt k=0,kr,kk; kr=k>=mdof[2], kk=k-kr*mdof[2], k<ldof[2]; k++) {
          // 5 fields in state vector
          for (PetscInt f=0; f<5; f++) {
            // Use periodic boundary conditions
            ltogind[(i*ldof[1]+j)*ldof[2]+k+f*lsize] =
              gstart[ir][jr][kr] +
                (ii*gmdof[ir<mdof[0]?ir:0][jr<mdof[1]?jr:0][kr<mdof[2]?kr:0][1]+jj)*
                 gmdof[ir<mdof[0]?ir:0][jr<mdof[1]?jr:0][kr<mdof[2]?kr:0][2]+kk+f*gsize;
          }
        }
      }
    }
    ierr = ISCreateGeneral(comm, 5*lsize, ltogind, PETSC_OWN_POINTER, &ltogis);
    CHKERRQ(ierr);
    ierr = VecScatterCreate(Qloc, NULL, Q, ltogis, &ltog); CHKERRQ(ierr);
    CHKERRQ(ierr);
    ierr = ISDestroy(&ltogis); CHKERRQ(ierr);
  }

    // Create local-to-global scatter to LHS
    PetscInt *ltogindLHS;
    IS ltogisLHS;

    lsizeLHS = 1;
    for (int d=0; d<3; d++) {
      lsizeLHS *= melem[d]*degree;
    }
    ierr = VecCreate(PETSC_COMM_SELF, &QlocLHS); CHKERRQ(ierr);
    ierr = VecSetSizes(QlocLHS, 5*lsizeLHS, PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetUp(QlocLHS); CHKERRQ(ierr);

    ierr = PetscMalloc1(5*lsizeLHS, &ltogindLHS); CHKERRQ(ierr);
    for (PetscInt i=0,ir,ii; ir=i>=mdof[0], ii=i-ir*mdof[0], i<ldof[0]-1; i++) {
      for (PetscInt j=0,jr,jj; jr=j>=mdof[1], jj=j-jr*mdof[1], j<ldof[1]-1; j++) {
        for (PetscInt k=0,kr,kk; kr=k>=mdof[2], kk=k-kr*mdof[2], k<ldof[2]-1; k++) {
          // 5 fields in state vector
          for (PetscInt f=0; f<5; f++) {
            // Use periodic boundary conditions
            ltogindLHS[(i*ldof[1]+j)*ldof[2]+k+f*lsizeLHS] =
              gstart[ir][jr][kr] +
                (ii*gmdof[ir<mdof[0]?ir:0][jr<mdof[1]?jr:0][kr<mdof[2]?kr:0][1]+jj)*
                 gmdof[ir<mdof[0]?ir:0][jr<mdof[1]?jr:0][kr<mdof[2]?kr:0][2]+kk+f*gsize;
          }
        }
      }
    }
    ierr = ISCreateGeneral(comm, 5*lsizeLHS, ltogindLHS, PETSC_OWN_POINTER, &ltogisLHS);
    CHKERRQ(ierr);
    ierr = VecScatterCreate(QlocLHS, NULL, Q, ltogisLHS, &ltogLHS); CHKERRQ(ierr);
    CHKERRQ(ierr);
    ierr = ISDestroy(&ltogisLHS); CHKERRQ(ierr);
  }

  CeedInit(ceedresource, &ceed);
  degP = degree + 1;
  degQ = P + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, degP, degQ, CEED_GAUSS, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 3, 2, degQ, CEED_GAUSS, &basisx);

  CreateRestriction(ceed, melem, degP, 1, &Erestrictu);
  CreateRestriction(ceed, melem, 2, 3, &Erestrictx);
  CeedInt nelem = melem[0]*melem[1]*melem[2];
  CeedElemRestrictionCreateIdentity(ceed, nelem, degQ*degQ*degQ,
                                    nelem*degQ*degQ*degQ, 1,
                                    &Erestrictui);
  CeedElemRestrictionCreateIdentity(ceed, nelem, 12*degQ*degQ*degQ,
                                    12*nelem*degQ*degQ*degQ, 1,
                                    &Erestrictqdi);
  CeedElemRestrictionCreateIdentity(ceed, nelem, degQ*degQ*degQ,
                                    nelem*degQ*degQ*degQ, 1,
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

  // Create the Q-function that builds the NS operator (i.e. computes its
  // quadrature data) and set its context data.
  CeedQFunctionCreateInterior(ceed, 1,
                              Setup, __FILE__ ":Setup", &qf_setup);
  CeedQFunctionAddInput(qf_setup, "dx", 3, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", 12, CEED_EVAL_NONE);

  // Create the Q-function that defines the action of the NS operator.
  CeedQFunctionCreateInterior(ceed, 1,
                              NS, __FILE__ ":NS", &qf_ns);
  CeedQFunctionAddInput(qf_ns, "q", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_ns, "dq", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_ns, "qdata", 12, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_ns, "x", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ns, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ns, "dv", 1, CEED_EVAL_GRAD);

  // Create the persistent vectors that will be needed in setup
  CeedInt Nqpts, Nelem = melem[0]*melem[1]*melem[2];
  CeedBasisGetNumQuadraturePoints(basisu, &Nqpts);
  CeedVectorCreate(ceed, 12*Nelem*Nqpts, &qdata);

  // Create the operator that builds the quadrature data for the NS operator.
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", Erestrictxi, basisx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "qdata", Erestrictqdi,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the Navier-Stokes operator.
  CeedOperatorCreate(ceed, qf_ns, NULL, NULL, &op_ns);
  CeedOperatorSetField(op_ns, "q", Erestrictu, basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ns, "dq", Erestrictu, basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ns, "qdata", Erestrictqdi,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_ns, "x", Erestrictx, basisx, xcoord);
  CeedOperatorSetField(op_ns, "v", Erestrictu, basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ns, "dv", Erestrictu, basisu, CEED_VECTOR_ACTIVE);

  // Create the Navier-Stokes context
  CeedScalar ctx[5] = {-2./3., 1.81e-5, 7.25e-5, 1.004, 9.81};
  CeedQFunctionSetContext(qf_ns, &ctx, sizeof ctx);

  // Set up context
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);
  userRHS->comm = comm;
  userRHS->ltog = ltog;
  userRHS->Qloc = Qloc;
  ierr = VecDuplicate(Qloc, &userRHS->Gloc); CHKERRQ(ierr);
  CeedVectorCreate(ceed, lsize, &userRHS->qceed);
  CeedVectorCreate(ceed, lsize, &userRHS->gceed);
  userRHS->op = op_ns;
  userRHS->qdata = qdata;
  userRHS->ceed = ceed;
  userLHS->comm = comm;
  userLHS->ltog = ltogLHS;
  ierr = VecDuplicate(QlocLHS, &userRHS->Qloc); CHKERRQ(ierr);
  ierr = VecDuplicate(QlocLHS, &userRHS->dQloc); CHKERRQ(ierr);
  ierr = VecDuplicate(QlocLHS, &userRHS->Floc); CHKERRQ(ierr);
  userLHS->lsize = lsizeLHS;

  // Setup qdata
  CeedOperatorApply(op_setup, xcoord, qdata, CEED_REQUEST_IMMEDIATE);

  // Create and setup TS
  ierr = TSCreate(comm, &ts); CHKERRQ(ierr);
  ierr = TSSetType(ts, TSEULER); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  ierr = TSSetIFunction(ts, NULL, LHS_NS, &userLHS);
  ierr = TSSetRHSFunction(ts, NULL, RHS_NS, &userRHS);

  // Solve
  ierr = TSSolve(ts, X); CHKERRQ(ierr);

  // Output Statistics
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetSNESIterations(ts,&nits);CHKERRQ(ierr);
  ierr = TSGetKSPIterations(ts,&lits);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
           "Time integrator took (%D,%D,%D) iterations to reach final time %g\n",
           steps,nits,lits,(double)ftime);CHKERRQ(ierr);

  // Clean up PETSc
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Qloc); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Gloc); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ltog); CHKERRQ(ierr);
  ierr = TSDestroy(&ts); CHKERRQ(ierr);

  // Clean up libCEED
  CeedVectorDestroy(&user->xceed);
  CeedVectorDestroy(&user->yceed);
  CeedVectorDestroy(&user->qdata);
  CeedVectorDestroy(&xcoord);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_ns);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictqdi);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_ns);
  CeedBasisDestroy(&basisu);
  CeedBasisDestroy(&basisx);
  CeedDestroy(&ceed);
  ierr = PetscFree(user); CHKERRQ(ierr);
  return PetscFinalize();
}
