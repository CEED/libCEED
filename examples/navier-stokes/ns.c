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

#include <petscts.h>
#include <petscdmda.h>
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

typedef struct User_ *User;
struct User_ {
  MPI_Comm comm;
  VecScatter ltog;              // Scatter for all entries
  VecScatter ltog0;             // Skip Dirichlet values for U
  VecScatter gtogD;             // global-to-global; only Dirichlet values for U
  Vec Qloc, Gloc;
  CeedVector qceed, gceed;
  CeedOperator op;
  CeedVector qdata;
  PetscInt degree;
  PetscInt melem[3];
  DM dm;
  Ceed ceed;
};

// This is the RHS of the DAE, given as u_t = G(t,u)
// This function takes in a state vector Q and writes into G
static PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *userData) {
  PetscErrorCode ierr;
  User user = *(User*)userData;
  PetscScalar *q, *g;

  // Global to local
  PetscFunctionBeginUser;
  ierr = VecScatterBegin(user->ltog0, Q, user->Qloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog0, Q, user->Qloc, INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Gloc); CHKERRQ(ierr);

// !!!!!
// Temporary Print Statement
// !!!!!
  //printf("t = %f\n", t);
  //ierr = VecView(Q, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

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
  ierr = VecScatterBegin(user->gtogD, Q, G, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(user->gtogD, Q, G, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterBegin(user->ltog0, user->Gloc, G, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog0, user->Gloc, G, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode TSMonitor_NS(TS ts, PetscInt stepno, PetscReal time, Vec X, void *ctx) {
  User user = ctx;
  const PetscScalar *x;
  PetscScalar ***u;
  Vec U;
  DMDALocalInfo info;
  char filepath[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetGlobalVector(user->dm, &U); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user->dm, &info); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->dm, U, &u); CHKERRQ(ierr);
  ierr = VecGetArrayRead(X, &x); CHKERRQ(ierr);
  for (PetscInt i=info.zs; i<info.zs+info.zm; i++) {
    for (PetscInt j=info.ys; j<info.ys+info.ym; j++) {
      for (PetscInt k=info.xs; k<info.xs+info.xm; k++) {
        for (PetscInt c=0; c<5; c++) {
          u[i][j][k*5+c] = x[c*info.xm*info.ym*info.zm + ((i*info.ym+j)*info.xm+k)];
        }
      }
    }
  }
  ierr = VecRestoreArrayRead(X, &x); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->dm, U, &u); CHKERRQ(ierr);
  ierr = PetscSNPrintf(filepath, sizeof filepath, "ns-%03D.vtr", stepno); CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)U), filepath, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  ierr = VecView(U, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(user->dm, &U); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char ceedresource[4096] = "/cpu/self";
  PetscInt degree, qextra, localdof, localelem, melem[3], mdof[3], p[3],
    irank[3], ldof[3], lsize;
  PetscMPIInt size, rank;
  PetscScalar ftime;
  PetscInt steps;
  PetscScalar *q0;
  VecScatter ltog, ltog0, gtogD;
  Ceed ceed;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi, Erestrictui,
                      Erestrictqdi;
  CeedQFunction qf_setup, qf_ns;
  CeedOperator op_setup, op_ns;
  CeedVector xcoord, qdata, q0ceed;
  CeedInt degP, degQ;
  Vec Q, Qloc;
  DM dm;
  TS ts;
  User user;

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
    PetscInt gsize;

    ierr = VecCreate(comm, &Q); CHKERRQ(ierr);
    gsize = mdof[0]*mdof[1]*mdof[2];
    ierr = VecSetSizes(Q, 5*gsize, PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetUp(Q); CHKERRQ(ierr);

    lsize = 1;
    for (int d=0; d<3; d++) {
      ldof[d] = melem[d]*degree + 1;
      lsize *= ldof[d];
    }
    ierr = VecCreate(PETSC_COMM_SELF, &Qloc); CHKERRQ(ierr);
    ierr = VecSetSizes(Qloc, 5*lsize, PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetUp(Qloc); CHKERRQ(ierr);

    // Create local-to-global scatters
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

    ierr = PetscMalloc1(5*lsize, &ltogind); CHKERRQ(ierr);
    ierr = PetscMalloc1(5*lsize, &ltogind0); CHKERRQ(ierr);
    ierr = PetscMalloc1(5*lsize, &locind); CHKERRQ(ierr);
    l0count = 0;
    for (PetscInt f=0; f<5; f++) {
      for (PetscInt i=0,ir,ii; ir=i>=mdof[0], ii=i-ir*mdof[0], i<ldof[0]; i++) {
        for (PetscInt j=0,jr,jj; jr=j>=mdof[1], jj=j-jr*mdof[1], j<ldof[1]; j++) {
          for (PetscInt k=0,kr,kk; kr=k>=mdof[2], kk=k-kr*mdof[2], k<ldof[2]; k++) {
            PetscInt here = (i*ldof[1]+j)*ldof[2]+k;
              ltogind[here+f*gsize] =
                gstart[ir][jr][kr] + (ii*gmdof[ir][jr][kr][1]+jj)*gmdof[ir][jr][kr][2]+kk+f*gsize;
            if (((irank[0] == 0 && i == 0)
                 || (irank[1] == 0 && j == 0)
                 || (irank[2] == 0 && k == 0)
                 || (irank[0]+1 == p[0] && i+1 == ldof[0])
                 || (irank[1]+1 == p[1] && j+1 == ldof[1])
                 || (irank[2]+1 == p[2] && k+1 == ldof[2]))
                 && (f != 0 || f != 4))
              continue;
            ltogind0[l0count] = ltogind[here+f*gsize];
            locind[l0count++] = here+f*gsize;
          }
        }
      }
    }
    ierr = ISCreateGeneral(comm, 5*lsize, ltogind, PETSC_OWN_POINTER, &ltogis);
    CHKERRQ(ierr);
    ierr = VecScatterCreate(Qloc, NULL, Q, ltogis, &ltog); CHKERRQ(ierr);
    CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, l0count, ltogind0, PETSC_OWN_POINTER, &ltogis0);
    CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, l0count, locind, PETSC_OWN_POINTER, &locis);
    CHKERRQ(ierr);
    ierr = VecScatterCreate(Qloc, locis, Q, ltogis0, &ltog0); CHKERRQ(ierr);
    { // Create global-to-global scatter for Dirichlet values (everything not in
      // ltogis0, which is the range of ltog0)
      PetscInt qstart, qend, *indD, countD = 0;
      IS isD;
      const PetscScalar *q;
      ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
      ierr = VecSet(Q, 1.0); CHKERRQ(ierr);
      ierr = VecScatterBegin(ltog0, Qloc, Q, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecScatterEnd(ltog0, Qloc, Q, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(Q, &qstart, &qend); CHKERRQ(ierr);
      ierr = PetscMalloc1(qend-qstart, &indD); CHKERRQ(ierr);
      ierr = VecGetArrayRead(Q, &q); CHKERRQ(ierr);
      for (PetscInt i=0; i<qend-qstart; i++) {
        if (q[i] == 1.) indD[countD++] = qstart + i;
      }
      ierr = VecRestoreArrayRead(Q, &q); CHKERRQ(ierr);
      ierr = ISCreateGeneral(comm, countD, indD, PETSC_COPY_VALUES, &isD); CHKERRQ(ierr);
      ierr = PetscFree(indD); CHKERRQ(ierr);
      ierr = VecScatterCreate(Q, isD, Q, isD, &gtogD); CHKERRQ(ierr);
      ierr = ISDestroy(&isD); CHKERRQ(ierr);
    }
    ierr = ISDestroy(&ltogis); CHKERRQ(ierr);
    ierr = ISDestroy(&ltogis0); CHKERRQ(ierr);
    ierr = ISDestroy(&locis); CHKERRQ(ierr);

    PetscInt *ldofs[3];
    ierr = PetscMalloc3(p[0], &ldofs[0], p[1], &ldofs[1], p[2], &ldofs[2]); CHKERRQ(ierr);
    for (PetscInt d=0; d<3; d++) {
      for (PetscInt r=0; r<p[d]; r++) {
        PetscInt ijkrank[3] = {irank[0], irank[1], irank[2]};
        ijkrank[d] = r;
        PetscInt ijkdof[3];
        GlobalDof(p, ijkrank, degree, melem, ijkdof);
        ldofs[d][r] = ijkdof[d];
      }
    }
    ierr = DMDACreate3d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_STAR,
                        degree*melem[2]*p[2]+1, degree*melem[1]*p[1]+1, degree*melem[0]*p[0]+1,
                        p[2], p[1], p[0],
                        5, 0,
                        ldofs[2], ldofs[1], ldofs[0], &dm); CHKERRQ(ierr);
    ierr = PetscFree3(ldofs[0], ldofs[1], ldofs[2]); CHKERRQ(ierr);
    ierr = DMSetUp(dm); CHKERRQ(ierr);
    ierr = DMDASetFieldName(dm, 0, "Density"); CHKERRQ(ierr);
    ierr = DMDASetFieldName(dm, 0, "MomentumX"); CHKERRQ(ierr);
    ierr = DMDASetFieldName(dm, 0, "MomentumY"); CHKERRQ(ierr);
    ierr = DMDASetFieldName(dm, 0, "MomentumZ"); CHKERRQ(ierr);
    ierr = DMDASetFieldName(dm, 0, "Energy"); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(dm, 0, 1, 0, 1, 0, 1); CHKERRQ(ierr);
  }

  // Set up CEED
  CeedInit(ceedresource, &ceed);
  degP = degree + 1;
  degQ = degP + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 5, degP, degQ, CEED_GAUSS, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 3, 2, degQ, CEED_GAUSS, &basisx);

  CreateRestriction(ceed, melem, degP, 5, &Erestrictu);
  CreateRestriction(ceed, melem, 2, 3, &Erestrictx);
  CeedInt nelem = melem[0]*melem[1]*melem[2];
  CeedElemRestrictionCreateIdentity(ceed, nelem, degQ*degQ*degQ,
                                    nelem*degQ*degQ*degQ, 5,
                                    &Erestrictui);
  CeedElemRestrictionCreateIdentity(ceed, nelem, 16*degQ*degQ*degQ,
                                    16*nelem*degQ*degQ*degQ, 1,
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
  CeedVectorCreate(ceed, 5*lsize, &q0ceed);
  CeedQFunctionCreateInterior(ceed, 1,
                              Setup, __FILE__ ":Setup", &qf_setup);
  CeedQFunctionAddInput(qf_setup, "x", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup, "dx", 3, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", 16, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup, "q0", 5, CEED_EVAL_INTERP);

  // Create the Q-function that defines the action of the NS operator.
  CeedQFunctionCreateInterior(ceed, 1,
                              NS, __FILE__ ":NS", &qf_ns);
  CeedQFunctionAddInput(qf_ns, "q", 5, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_ns, "dq", 5, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_ns, "qdata", 16, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_ns, "x", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ns, "v", 5, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ns, "dv", 5, CEED_EVAL_GRAD);

  // Create the persistent vectors that will be needed in setup
  CeedInt Nqpts, Nelem = melem[0]*melem[1]*melem[2];
  CeedBasisGetNumQuadraturePoints(basisu, &Nqpts);
  CeedVectorCreate(ceed, 16*Nelem*Nqpts, &qdata);

  // Create the operator that builds the quadrature data for the NS operator.
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "x", Erestrictx, basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", Erestrictxi, basisx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "qdata", Erestrictqdi,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "q0", Erestrictu, basisu, q0ceed);

  // Create the Navier-Stokes operator.
  CeedOperatorCreate(ceed, qf_ns, NULL, NULL, &op_ns);
  CeedOperatorSetField(op_ns, "q", Erestrictu, basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ns, "dq", Erestrictu, basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ns, "qdata", Erestrictqdi,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_ns, "x", Erestrictx, basisx, xcoord);
  CeedOperatorSetField(op_ns, "v", Erestrictu, basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ns, "dv", Erestrictu, basisu, CEED_VECTOR_ACTIVE);

  // Create the libCEED contexts
  CeedScalar Rd         = 0.287;
  CeedScalar Ts         = 300.;
  CeedScalar p0         = 100000.;
  CeedScalar cp         = 1.004;
  CeedScalar cv         = 0.717;
  CeedScalar g          = 9.81;
  CeedScalar lambda     = -2./3.;
  CeedScalar mu         = 73.;
  CeedScalar k          = 0.02638;
  CeedScalar Pr         = cp*mu/k;
  CeedScalar ctxSetup[6] = {Rd, Ts, p0, cv, cp, g};
  CeedQFunctionSetContext(qf_setup, &ctxSetup, sizeof ctxSetup);
  CeedScalar ctxNS[6] = {lambda, mu, Pr, cp, cv, g};
  CeedQFunctionSetContext(qf_ns, &ctxNS, sizeof ctxNS);

  // Set up PETSc context
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);
  user->comm = comm;
  user->ltog = ltog;
  user->ltog0 = ltog0;
  user->gtogD = gtogD;
  user->Qloc = Qloc;
  ierr = VecDuplicate(Qloc, &user->Gloc); CHKERRQ(ierr);
  CeedVectorCreate(ceed, 5*lsize, &user->qceed);
  CeedVectorCreate(ceed, 5*lsize, &user->gceed);
  user->op = op_ns;
  user->qdata = qdata;
  user->degree = degree;
  for (int d=0; d<3; d++) user->melem[d] = melem[d];
  user->dm = dm;
  user->ceed = ceed;

  // Setup qdata and IC
  // Set up vectors
  ierr = VecGetArray(Qloc, &q0); CHKERRQ(ierr);
  CeedVectorSetArray(q0ceed, CEED_MEM_HOST, CEED_USE_POINTER, q0);

  // Apply Setup Ceed Operator
  CeedOperatorApply(op_setup, xcoord, qdata, CEED_REQUEST_IMMEDIATE);

  // Gather initial Q values
  ierr = VecRestoreArray(Qloc, &q0); CHKERRQ(ierr);
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);
  ierr = VecScatterBegin(ltog, Qloc, Q, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(ltog, Qloc, Q, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  CeedVectorDestroy(&q0ceed);

  // Create and setup TS
  ierr = TSCreate(comm, &ts); CHKERRQ(ierr);
  ierr = TSSetType(ts, TSEULER); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts, 100); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);

  ierr = TSMonitor_NS(ts, 0, 0., Q, user); CHKERRQ(ierr);

  // Solve
  ierr = TSSolve(ts, Q); CHKERRQ(ierr);

// !!!!!
// Temporary Print Statement
// !!!!!
  //printf("t = %f\n", 0.0);
  //ierr = VecView(Q, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  // Output Statistics
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
           "Time integrator took %D time steps to reach final time %g\n",
           steps,(double)ftime);CHKERRQ(ierr);

  // Clean up PETSc
  ierr = VecDestroy(&Q); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Qloc); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Gloc); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ltog); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ltog0); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&gtogD); CHKERRQ(ierr);
  ierr = TSDestroy(&ts); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  // Clean up libCEED
  CeedVectorDestroy(&user->qceed);
  CeedVectorDestroy(&user->gceed);
  CeedVectorDestroy(&user->qdata);
  CeedVectorDestroy(&xcoord);
  CeedBasisDestroy(&basisu);
  CeedBasisDestroy(&basisx);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictqdi);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_ns);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_ns);
  CeedDestroy(&ceed);
  ierr = PetscFree(user); CHKERRQ(ierr);
  return PetscFinalize();
}
