//                        libCEED + PETSc Example: Shallow-water equations
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// shallow-water equations on a cubed-sphere (i.e., a spherical surface
// tessellated by quadrilaterals, obtained by projecting the sides
// of a circumscribed cube onto a spherical surface).
//
// The code is intentionally "raw", using only low-level communication
// primitives.
//
// Build with:
//
//     make [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     shallow-water
//     shallow-water -ceed /cpu/self
//     shallow-water -ceed /gpu/occa
//     shallow-water -ceed /cpu/occa
//     shallow-water -ceed /omp/occa
//     shallow-water -ceed /ocl/occa
//

/// @file
/// Shallow-water equations example using PETSc

const char help[] = "Solve the shallow-water equations using PETSc and libCEED\n";

#include <petscts.h>
#include <petscdmda.h>
#include <ceed.h>
#include <stdbool.h>
#include <petscsys.h>
#include "common.h"
#include "shallow-water.h"

// Utility function, compute three factors of an integer
static void Split2(PetscInt size, PetscInt m[2], bool reverse) {
  for (PetscInt d=0,sizeleft=size; d<2; d++) {
    PetscInt try = (PetscInt)PetscCeilReal(PetscPowReal(sizeleft, 1./(2 - d)));
    while (try * (sizeleft / try) != sizeleft) try++;
    m[reverse ? 1-d : d] = try;
    sizeleft /= try;
  }
}

// Utility function, compute the number of DoFs from the global grid
static void GlobalDof(const PetscInt p[2], const PetscInt irank[2],
                      PetscInt degree, const PetscInt melem[2],
                      PetscInt mdof[2]) {
  for (int d=0; d<2; d++)
    mdof[d] = degree*melem[d] + (irank[d] == p[d]-1);
}

// Utility function
static PetscInt GlobalStart(const PetscInt p[2], const PetscInt irank[2],
                            PetscInt degree, const PetscInt melem[2]) {
  PetscInt start = 0;
  // Dumb brute-force is easier to read
  for (PetscInt i=0; i<p[0]; i++) {
    for (PetscInt j=0; j<p[1]; j++) {
        PetscInt mdof[2], ijrank[] = {i,j};
        if (i == irank[0] && j == irank[1]) return start;
        GlobalDof(p, ijrank, degree, melem, mdof);
        start += mdof[0] * mdof[1];
      }
  }
  return -1;
}

// Utility function to create local CEED restriction
static int CreateRestriction(Ceed ceed, const CeedInt melem[2],
                             CeedInt P, CeedInt ncomp,
                             CeedElemRestriction *Erestrict) {
  const PetscInt Nelem = melem[0]*melem[1];
  PetscInt mdof[2], *idx, *idxp;

  for (int d=0; d<2; d++) mdof[d] = melem[d]*(P-1) + 1;
  idxp = idx = malloc(Nelem*P*P*sizeof idx[0]);
  for (CeedInt i=0; i<melem[0]; i++) {
    for (CeedInt j=0; j<melem[1]; j++,idxp += P*P) {
      for (CeedInt ii=0; ii<P; ii++) {
        for (CeedInt jj=0; jj<P; jj++) {
          if (0) { // This is the C-style (i,j) ordering that I prefer
            idxp[ii*P+jj] = (((i*(P-1)+ii)*mdof[1]
                            + (j*(P-1)+jj)));
          } else { // (j,i) ordering for consistency with MFEM example
            idxp[ii+P*jj] = (((i*(P-1)+ii)*mdof[1]
                            + (j*(P-1)+jj)));
          }
        }
      }
    }
  }
  CeedElemRestrictionCreate(ceed, Nelem, P*P, mdof[0]*mdof[1], ncomp,
                            CEED_MEM_HOST, CEED_OWN_POINTER, idx, Erestrict);
  PetscFunctionReturn(0);
}

// PETSc user data
typedef struct User_ *User;
struct User_ {
  MPI_Comm comm;
  PetscInt degree;
  PetscInt melem[3];
  PetscInt outputfreq;
  DM dm;
  Ceed ceed;
  CeedVector qceed, gceed;
  CeedOperator op_explicit, op_implicit;
  VecScatter ltog;              // Scatter for all entries
  VecScatter ltog0;             // Skip Dirichlet values for Q
  VecScatter gtogD;             // global-to-global; only Dirichlet values for Q
  Vec Qloc, Gloc, M, BC;
  char outputfolder[PETSC_MAX_PATH_LEN];
  PetscInt contsteps;
};

// This is the RHS of the ODE, given as u_t = G(t,u)
// This function takes in a state vector Q and writes into G
static PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *userData) {
  PetscErrorCode ierr;
  User user = *(User*)userData;
  PetscScalar *q, *g;

  // Global-to-local
  PetscFunctionBeginUser;
  ierr = VecScatterBegin(user->ltog, Q, user->Qloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog, Q, user->Qloc, INSERT_VALUES,
                       SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Gloc); CHKERRQ(ierr);

  // Ceed Vectors
  ierr = VecGetArrayRead(user->Qloc, (const PetscScalar**)&q); CHKERRQ(ierr);
  ierr = VecGetArray(user->Gloc, &g); CHKERRQ(ierr);
  CeedVectorSetArray(user->qceed, CEED_MEM_HOST, CEED_USE_POINTER, q);
  CeedVectorSetArray(user->gceed, CEED_MEM_HOST, CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_explicit, user->qceed, user->gceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  ierr = VecRestoreArrayRead(user->Qloc, (const PetscScalar**)&q); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Gloc, &g); CHKERRQ(ierr);

  ierr = VecZeroEntries(G); CHKERRQ(ierr);

  // Global-to-global
  // G on the boundary = BC
  ierr = VecScatterBegin(user->gtogD, user->BC, G, INSERT_VALUES,
                         SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->gtogD, user->BC, G, INSERT_VALUES,
                       SCATTER_FORWARD); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecScatterBegin(user->ltog0, user->Gloc, G, ADD_VALUES,
                         SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog0, user->Gloc, G, ADD_VALUES,
                       SCATTER_FORWARD); CHKERRQ(ierr);

  // Inverse of the lumped mass matrix
  ierr = VecPointwiseMult(G,G,user->M); // M is Minv
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// User provided TS Monitor
static PetscErrorCode TSMonitor_SW(TS ts, PetscInt stepno, PetscReal time,
                                   Vec h, void *ctx) {
  User user = ctx;
  const PetscScalar *hpt;
  PetscScalar ***u;
  Vec U;
  DMDALocalInfo info;
  char filepath[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  PetscErrorCode ierr;

  // Print every 'outputfreq' steps
  if (stepno % user->outputfreq != 0)
    PetscFunctionReturn(0);

  // Set up output
  PetscFunctionBeginUser;
  ierr = DMGetGlobalVector(user->dm, &U); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)U, "StateVec");
  ierr = DMDAGetLocalInfo(user->dm, &info); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->dm, U, &u); CHKERRQ(ierr);
  ierr = VecGetArrayRead(h, &hpt); CHKERRQ(ierr);
  for (PetscInt i=0; i<info.zm; i++) {
    for (PetscInt j=0; j<info.ym; j++) {
      for (PetscInt k=0; k<info.xm; k++) {
        for (PetscInt c=0; c<5; c++) {
          u[info.zs+i][info.ys+j][(info.xs+k)*5 + c] =
                  hpt[((i*info.ym+j)*info.xm+k)*5 + c];
        }
      }
    }
  }
  ierr = VecRestoreArrayRead(h, &hpt); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->dm, U, &u); CHKERRQ(ierr);

  // Output
  ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/sw-%03D.vts",
                       user->outputfolder, stepno + user->contsteps);
  CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)U), filepath,
                            FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  ierr = VecView(U, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(user->dm, &U); CHKERRQ(ierr);

  // Save data in a binary file for continuation of simulations
  ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/sw-solution.bin",
                       user->outputfolder); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm, filepath, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = VecView(h, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  // Save time stamp
  ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/sw-time.bin",
                       user->outputfolder); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm, filepath, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL, true);
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  DM dm;
  TS ts;
  TSAdapt adapt;
  User user;
  char ceedresource[4096] = "/cpu/self";
  PetscFunctionList icsflist = NULL, qflist = NULL;
  char problemtype[PETSC_MAX_PATH_LEN] = "sphere";
  PetscInt degree, qextra, localNelem, lsize, outputfreq,
           steps, melem[2], mdof[2], p[2], irank[2], ldof[2], contsteps;
  PetscMPIInt size, rank;
  PetscScalar ftime;
  PetscScalar *q0, *m, *mult, *x;
  Vec Q, Qloc, Mloc, X, Xloc;
  VecScatter ltog, ltog0, gtogD, ltogX;

  Ceed ceed;
  CeedInt numP, numQ;
  CeedVector xcorners, xceed, qdata, q0ceed, mceed,
             onesvec, multevec, multlvec;
  CeedBasis basisx, basisxc, basisq;
  CeedElemRestriction restrictx, restrictxc, restrictxi,
                      restrictq, restrictqdi, restrictmult;
  CeedQFunction qf_setup, qf_mass, qf_ics, qf_explicit, qf_implicit;
  CeedOperator op_setup, op_mass, op_ics, op_explicit, op_implicit;

  // Create the libCEED contexts
  PetscScalar lx;                   // m
  PetscScalar ly;                   // m
  PetscScalar lz;                   // m
  PetscScalar resx;                 // m (resolution in x)
  PetscScalar resy;                 // m (resolution in y)

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  // Allocate PETSc context
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);

  // Set up problem type command line option
  //PetscFunctionListAdd(&icsflist, "sphere", &ICsSW);
  //PetscFunctionListAdd(&qflist, "shallow-water", &SW);

  // Parse command line options
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "Shallow-water equations in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  PetscOptionsFList("-problem", "Problem to solve", NULL, icsflist,
                    problemtype, problemtype, sizeof problemtype, NULL);
  lx = 8000.;
  ierr = PetscOptionsScalar("-lx", "Length scale in x direction",
                            NULL, lx, &lx, NULL); CHKERRQ(ierr);
  lx = fabs(lx);
  ly = 8000.;
  ierr = PetscOptionsScalar("-ly", "Length scale in y direction",
                            NULL, ly, &ly, NULL); CHKERRQ(ierr);
  ly = fabs(ly);
  outputfreq = 10;
  ierr = PetscOptionsInt("-output_freq", "Frequency of output, in number of steps",
                         NULL, outputfreq, &outputfreq, NULL); CHKERRQ(ierr);
  contsteps = 0;
  ierr = PetscOptionsInt("-continue", "Continue from previous solution",
                         NULL, contsteps, &contsteps, NULL); CHKERRQ(ierr);
  degree = 3;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  qextra = 2;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  PetscStrncpy(user->outputfolder, ".", 2);
  ierr = PetscOptionsString("-of", "Output folder",
                            NULL, user->outputfolder, user->outputfolder,
                            sizeof(user->outputfolder), NULL); CHKERRQ(ierr);
  resx = 1000.;
  ierr = PetscOptionsScalar("-resx","Resolution in x",
                            NULL, resx, &resx, NULL); CHKERRQ(ierr);
  resx = fabs(resx);
  resy = 1000.;
  ierr = PetscOptionsScalar("-resy","Resolution in y",
                            NULL, resy, &resy, NULL); CHKERRQ(ierr);
  resy = fabs(resy);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Determine size of process grid
  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  Split3(size, p, false);

  // Find a nicely composite number of elements given the resolution
  melem[0] = (PetscInt)(PetscRoundReal(lx / resx));
  melem[1] = (PetscInt)(PetscRoundReal(ly / resy));
  for (int d=0; d<2; d++) {
    if (melem[d] == 0)
      melem[d]++;
  }
  localNelem = melem[0] * melem[1];

  // Find my location in the process grid
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  for (int d=0,rankleft=rank; d<2; d++) {
    const int pstride[2] = {p[1], 1};
    irank[d] = rankleft / pstride[d];
    rankleft -= irank[d] * pstride[d];
  }

  GlobalDof(p, irank, degree, melem, mdof);

  // Set up global state vector
  ierr = VecCreate(comm, &Q); CHKERRQ(ierr);
  ierr = VecSetSizes(Q, 3*mdof[0]*mdof[1], PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(Q); CHKERRQ(ierr);

  // Set up local state vector
  lsize = 1;
  for (int d=0; d<2; d++) {
    ldof[d] = melem[d]*degree + 1;
    lsize *= ldof[d];
  }
  ierr = VecCreate(PETSC_COMM_SELF, &Qloc); CHKERRQ(ierr);
  ierr = VecSetSizes(Qloc, 3*lsize, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(Qloc); CHKERRQ(ierr);

  // Print grid information
  CeedInt gsize;
  ierr = VecGetSize(Q, &gsize); CHKERRQ(ierr);
//  gsize /= 5;
  ierr = PetscPrintf(comm, "Global dofs: %D\n", gsize); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Process decomposition: %D %D\n",
                     p[0], p[1]); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Local elements: %D = %D %D\n", localNelem,
                     melem[0], melem[1]); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Owned dofs: %D = %D %D\n",
                     mdof[0]*mdof[1], mdof[0], mdof[1]);
  CHKERRQ(ierr);

  // Set up global mass vector
  ierr = VecDuplicate(Q,&user->M); CHKERRQ(ierr);

  // Set up local mass vector
  ierr = VecDuplicate(Qloc,&Mloc); CHKERRQ(ierr);

  // Set up global coordinates vector
  ierr = VecCreate(comm, &X); CHKERRQ(ierr);
  ierr = VecSetSizes(X, 2*mdof[0]*mdof[1], PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(X); CHKERRQ(ierr);

  // Set up local coordinates vector
  ierr = VecCreate(PETSC_COMM_SELF, &Xloc); CHKERRQ(ierr);
  ierr = VecSetSizes(Xloc, 2*lsize, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(Xloc); CHKERRQ(ierr);

  // Set up global boundary values vector
  ierr = VecDuplicate(Q,&user->BC); CHKERRQ(ierr);

  {
    // Create local-to-global scatters
    PetscInt *ltogind, *ltogind0, *locind, l0count;
    IS ltogis, ltogxis, ltogis0, locis;
    PetscInt gstart[2][2], gmdof[2][2][2];

    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
        PetscInt ijrank[3] = {irank[0]+i, irank[1]+j};
        gstart[i][j] = GlobalStart(p, ijrank, degree, melem);
        GlobalDof(p, ijrank, degree, melem, gmdof[i][j]);
      }
    }

    // Get indices of dofs except Dirichlet BC dofs
    ierr = PetscMalloc1(lsize, &ltogind); CHKERRQ(ierr);
    ierr = PetscMalloc1(lsize, &ltogind0); CHKERRQ(ierr);
    ierr = PetscMalloc1(lsize, &locind); CHKERRQ(ierr);
    l0count = 0;
    for (PetscInt i=0,ir,ii; ir=i>=mdof[0], ii=i-ir*mdof[0], i<ldof[0]; i++) {
      for (PetscInt j=0,jr,jj; jr=j>=mdof[1], jj=j-jr*mdof[1], j<ldof[1]; j++) {
        PetscInt dofind = i*ldof[1]+j;
        ltogind[dofind] =
          gstart[ir][jr] + ii*gmdof[ir][jr][1]+jj;
        if ((irank[0] == 0 && i == 0) ||
            (irank[1] == 0 && j == 0) ||
            (irank[0]+1 == p[0] && i+1 == ldof[0]) ||
            (irank[1]+1 == p[1] && j+1 == ldof[1]))
          continue;
        ltogind0[l0count] = ltogind[dofind];
        locind[l0count++] = dofind;
      }
    }

    // Create local-to-global scatters
    ierr = ISCreateBlock(comm, 2, lsize, ltogind, PETSC_COPY_VALUES, &ltogxis);
    CHKERRQ(ierr);
    ierr = VecScatterCreate(Xloc, NULL, X, ltogxis, &ltogX);
    CHKERRQ(ierr);
    ierr = ISCreateBlock(comm, 3, lsize, ltogind, PETSC_OWN_POINTER, &ltogis);
    CHKERRQ(ierr);
    ierr = VecScatterCreate(Qloc, NULL, Q, ltogis, &ltog);
    CHKERRQ(ierr);
    ierr = ISCreateBlock(comm, 3, l0count, ltogind0, PETSC_OWN_POINTER, &ltogis0);
    CHKERRQ(ierr);
    ierr = ISCreateBlock(comm, 3, l0count, locind, PETSC_OWN_POINTER, &locis);
    CHKERRQ(ierr);
    ierr = VecScatterCreate(Qloc, locis, Q, ltogis0, &ltog0);
    CHKERRQ(ierr);

    {
      // Create global-to-global scatter for Dirichlet values (everything not in
      // ltogis0, which is the range of ltog0)
      PetscInt qstart, qend, *indD, countD = 0;
      IS isD;
      const PetscScalar *q;
      ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
      ierr = VecSet(Q, 1.0); CHKERRQ(ierr);
      ierr = VecScatterBegin(ltog0, Qloc, Q, INSERT_VALUES, SCATTER_FORWARD);
      CHKERRQ(ierr);
      ierr = VecScatterEnd(ltog0, Qloc, Q, INSERT_VALUES, SCATTER_FORWARD);
      CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(Q, &qstart, &qend); CHKERRQ(ierr);
      ierr = PetscMalloc1(qend-qstart, &indD); CHKERRQ(ierr);
      ierr = VecGetArrayRead(Q, &q); CHKERRQ(ierr);
      for (PetscInt i=0; i<qend-qstart; i++) {
        if (q[i] == 1.)
          indD[countD++] = qstart + i;
      }
      ierr = VecRestoreArrayRead(Q, &q); CHKERRQ(ierr);
      ierr = ISCreateGeneral(comm, countD, indD, PETSC_COPY_VALUES, &isD);
      CHKERRQ(ierr);
      ierr = PetscFree(indD); CHKERRQ(ierr);
      ierr = VecScatterCreate(Q, isD, Q, isD, &gtogD); CHKERRQ(ierr);
      ierr = ISDestroy(&isD); CHKERRQ(ierr);
    }
    ierr = ISDestroy(&ltogis); CHKERRQ(ierr);
    ierr = ISDestroy(&ltogxis); CHKERRQ(ierr);
    ierr = ISDestroy(&ltogis0); CHKERRQ(ierr);
    ierr = ISDestroy(&locis); CHKERRQ(ierr);

    {
      // Set up DMDA
      PetscInt *ldofs[2];
      ierr = PetscMalloc2(p[0], &ldofs[0], p[1], &ldofs[1]);
      CHKERRQ(ierr);
      for (PetscInt d=0; d<2; d++) {
        for (PetscInt r=0; r<p[d]; r++) {
          PetscInt ijrank[2] = {irank[0], irank[1]};
          ijrank[d] = r;
          PetscInt ijdof[3];
          GlobalDof(p, ijrank, degree, melem, ijdof);
          ldofs[d][r] = ijdof[d];
        }
      }
      ierr = DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                          DMDA_STENCIL_STAR,
                          degree*melem[1]*p[1]+1,
                          degree*melem[0]*p[0]+1,
                          p[1], p[0], 1, 0,
                          ldofs[1], ldofs[0], &dm); CHKERRQ(ierr);
      ierr = PetscFree2(ldofs[0], ldofs[1]); CHKERRQ(ierr);
      ierr = DMSetUp(dm); CHKERRQ(ierr);
      ierr = DMDASetFieldName(dm, 0, "VelocityLambda"); CHKERRQ(ierr);
      ierr = DMDASetFieldName(dm, 1, "VelocityTheta"); CHKERRQ(ierr);
      ierr = DMDASetFieldName(dm, 2, "Height"); CHKERRQ(ierr);
    }
  }

  // Set up CEED
  // CEED Bases
  CeedInit(ceedresource, &ceed);
  numP = degree + 1;
  numQ = numP + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, 2, 3, numP, numQ, CEED_GAUSS, &basisq);
  CeedBasisCreateTensorH1Lagrange(ceed, 2, 2, 2, numQ, CEED_GAUSS, &basisx);
  CeedBasisCreateTensorH1Lagrange(ceed, 2, 2, 2, numP, CEED_GAUSS_LOBATTO,
                                  &basisxc);

  // CEED Restrictions
  CreateRestriction(ceed, melem, numP, 3, &restrictq);
  CreateRestriction(ceed, melem, 2, 2, &restrictx);
  CreateRestriction(ceed, melem, numP, 2, &restrictxc);
  CreateRestriction(ceed, melem, numP, 1, &restrictmult);
  CeedElemRestrictionCreateIdentity(ceed, localNelem, 8*numQ*numQ,
                                    8*localNelem*numQ*numQ, 1,
                                    &restrictqdi);
  CeedElemRestrictionCreateIdentity(ceed, localNelem, numQ*numQ,
                                    localNelem*numQ*numQ, 1,
                                    &restrictxi);

  // Find physical cordinates of the corners of local elements
  {
    CeedScalar *xloc;
    CeedInt shape[2] = {melem[0]+1, melem[1]+1}, len =
                         shape[0]*shape[1];
    xloc = malloc(len*2*sizeof xloc[0]);
    for (CeedInt i=0; i<shape[0]; i++) {
      for (CeedInt j=0; j<shape[1]; j++) {
        xloc[(i*shape[1]+j) + 0*len] =
               lx * (irank[0]*melem[0]+i) / (p[0]*melem[0]);
        xloc[(i*shape[1]+j) + 1*len] =
               ly * (irank[1]*melem[1]+j) / (p[1]*melem[1]);
      }
    }
    CeedVectorCreate(ceed, len*2, &xcorners);
    CeedVectorSetArray(xcorners, CEED_MEM_HOST, CEED_OWN_POINTER, xloc);
  }

  // Create the CEED vectors that will be needed in setup
  CeedInt Nqpts;
  CeedBasisGetNumQuadraturePoints(basisq, &Nqpts);
  CeedInt Ndofs = 1;
  for (int d=0; d<2; d++) Ndofs *= numP;
  CeedVectorCreate(ceed, 8*localNelem*Nqpts, &qdata);
  CeedVectorCreate(ceed, 3*lsize, &q0ceed);
  CeedVectorCreate(ceed, 3*lsize, &mceed);
  CeedVectorCreate(ceed, 3*lsize, &onesvec);
  CeedVectorCreate(ceed, 2*lsize, &xceed);
  CeedVectorCreate(ceed, lsize, &multlvec);
  CeedVectorCreate(ceed, localNelem*Ndofs, &multevec);

  // Find multiplicity of each local point
  CeedVectorSetValue(multevec, 1.0);
  CeedVectorSetValue(multlvec, 0.);
  CeedElemRestrictionApply(restrictmult, CEED_TRANSPOSE, CEED_TRANSPOSE,
                           multevec, multlvec, CEED_REQUEST_IMMEDIATE);

  // Create the Q-function that builds the quadrature data for the NS operator
  CeedQFunctionCreateInterior(ceed, 1,
                              Setup, __FILE__ ":Setup", &qf_setup);
  CeedQFunctionAddInput(qf_setup, "dx", 2, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", 8, CEED_EVAL_NONE);

  // Create the Q-function that defines the action of the mass operator
  CeedQFunctionCreateInterior(ceed, 1,
                              Mass, __FILE__ ":Mass", &qf_mass);
  CeedQFunctionAddInput(qf_mass, "q", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "qdata", 8, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_mass, "v", 3, CEED_EVAL_INTERP);

  // Create the Q-function that sets the ICs of the operator
  CeedQFunctionCreateInterior(ceed, 1, Setup, __FILE__ ":ICsSW", &qf_ics);
  CeedQFunctionAddInput(qf_ics, "x", 2, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ics, "q0", 3, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_ics, "h_s", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_ics, "coords", 2, CEED_EVAL_NONE);

  // Create the Q-function that defines the action of the explicit operator
  CeedQFunctionCreateInterior(ceed, 1,
                              SWExplicit,  __FILE__ ":SWExplicit", &qf_explicit);
  CeedQFunctionAddInput(qf_explicit, "q", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_explicit, "dq", 3, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_explicit, "qdata", 8, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_explicit, "x", 2, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_explicit, "v", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_explicit, "dv", 3, CEED_EVAL_GRAD);

  // Create the Q-function that defines the action of the implicit operator
  CeedQFunctionCreateInterior(ceed, 1,
                              SWImplicit,  __FILE__ ":SWImplicit", &qf_implicit);
  CeedQFunctionAddInput(qf_implicit, "q", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_implicit, "dq", 3, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_implicit, "qdata", 8, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_implicit, "x", 2, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_implicit, "v", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_implicit, "dv", 3, CEED_EVAL_GRAD);

  // Create the operator that builds the quadrature data for the NS operator
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "dx", restrictx, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", restrictxi, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "qdata", restrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the mass operator
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "q", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", restrictqdi, CEED_NOTRANSPOSE,
                       basisx, qdata);
  CeedOperatorSetField(op_mass, "v", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);

  // Create the operator that sets the ICs
  CeedOperatorCreate(ceed, qf_ics, NULL, NULL, &op_ics);
  CeedOperatorSetField(op_ics, "x", restrictx, CEED_NOTRANSPOSE,
                       basisxc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ics, "q0", restrictq, CEED_TRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ics, "h_s", restrictq, CEED_TRANSPOSE,
                       CEED_BASIS_COLLOCATED, xceed); // TO DO: check if I need a different restriction
  CeedOperatorSetField(op_ics, "coords", restrictxc, CEED_TRANSPOSE,
                       CEED_BASIS_COLLOCATED, xceed);

  // Create the explicit part of physics operator
  CeedOperatorCreate(ceed, qf_explicit, NULL, NULL, &op_explicit);
  CeedOperatorSetField(op_explicit, "q", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_explicit, "dq", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_explicit, "qdata", restrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_explicit, "x", restrictx, CEED_NOTRANSPOSE,
                       basisx, xcorners);
  CeedOperatorSetField(op_explicit, "v", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_explicit, "dv", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);

  // Create the implicit part of physics operator
  CeedOperatorCreate(ceed, qf_implicit, NULL, NULL, &op_implicit);
  CeedOperatorSetField(op_implicit, "q", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_implicit, "dq", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_implicit, "qdata", restrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_implicit, "x", restrictx, CEED_NOTRANSPOSE,
                       basisx, xcorners);
  CeedOperatorSetField(op_implicit, "v", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_implicit, "dv", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);

  // Set up the libCEED context
  CeedScalar ctxSetup[3] = {lx, ly, lz};
  CeedQFunctionSetContext(qf_ics, &ctxSetup, sizeof ctxSetup);

  // Set up PETSc context
  user->comm = comm;
  user->degree = degree;
  for (int d=0; d<2; d++) user->melem[d] = melem[d];
  user->outputfreq = outputfreq;
  user->contsteps = contsteps;
  user->dm = dm;
  user->ceed = ceed;
  CeedVectorCreate(ceed, 3*lsize, &user->qceed);
  CeedVectorCreate(ceed, 3*lsize, &user->gceed);
  user->op_explicit = op_explicit;
  user->op_implicit = op_implicit;
  user->ltog = ltog;
  user->ltog0 = ltog0;
  user->gtogD = gtogD;
  user->Qloc = Qloc;
  ierr = VecDuplicate(Qloc, &user->Gloc); CHKERRQ(ierr);

  // Calculate qdata and ICs
  // Set up state global and local vectors
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);
  ierr = VecGetArray(Qloc, &q0); CHKERRQ(ierr);
  CeedVectorSetArray(q0ceed, CEED_MEM_HOST, CEED_USE_POINTER, q0);

  // Set up mass global and local vectors
  ierr = VecZeroEntries(user->M); CHKERRQ(ierr);
  ierr = VecGetArray(Mloc, &m); CHKERRQ(ierr);
  CeedVectorSetArray(mceed, CEED_MEM_HOST, CEED_USE_POINTER, m);

  // Set up dof coordinate global and local vectors
  ierr = VecZeroEntries(X); CHKERRQ(ierr);
  ierr = VecGetArray(Xloc, &x); CHKERRQ(ierr);
  CeedVectorSetArray(xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Apply Setup Ceed Operators
  CeedOperatorApply(op_setup, xcorners, qdata, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_ics, xcorners, q0ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorSetValue(onesvec, 1.0);
  CeedOperatorApply(op_mass, onesvec, mceed, CEED_REQUEST_IMMEDIATE);

  // Fix multiplicity for output of ICs
  CeedVectorGetArray(q0ceed, CEED_MEM_HOST, &q0);
  CeedVectorGetArray(xceed, CEED_MEM_HOST, &x);
  CeedVectorGetArray(multlvec, CEED_MEM_HOST, &mult);
  for (PetscInt i=0; i<lsize; i++) {
    for (PetscInt f=0; f<3; f++)
      q0[i*3+f] /= mult[i];
    for (PetscInt d=0; d<2; d++)
      x[i*2+d] /= mult[i];
  }

  CeedVectorRestoreArray(q0ceed, &q0);
  CeedVectorRestoreArray(xceed, &x);
  CeedVectorRestoreArray(multlvec, &mult);

  // Destroy mult vecs
  CeedVectorDestroy(&multevec);
  CeedVectorDestroy(&multlvec);

  // Gather initial h values
  ierr = VecRestoreArray(Qloc, &q0); CHKERRQ(ierr);
  // In case of continuation of simulation, set up initial values from binary file
  if (contsteps){ // continue from existent solution
    PetscViewer viewer;
    char filepath[PETSC_MAX_PATH_LEN];
    // Read input
    ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/sw-solution.bin",
                         user->outputfolder);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, filepath, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = VecLoad(Q, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(ltog, Qloc, Q, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(ltog, Qloc, Q, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
  }
  CeedVectorDestroy(&q0ceed);

  // Copy boundary values
  ierr = VecZeroEntries(user->BC); CHKERRQ(ierr);
  ierr = VecScatterBegin(gtogD, Q, user->BC, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(gtogD, Q, user->BC, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);

  // Gather dof coordinates
  ierr = VecRestoreArray(Xloc, &x); CHKERRQ(ierr);
  ierr = VecScatterBegin(ltogX, Xloc, X, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(ltogX, Xloc, X, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);

  // Clean up
  CeedVectorDestroy(&xceed);
  ierr = VecDestroy(&Xloc); CHKERRQ(ierr);

  // Set dof coordinates in DMDA
  ierr = DMSetCoordinates(dm, X); CHKERRQ(ierr);
  ierr = VecDestroy(&X); CHKERRQ(ierr);

  // Gather the inverse of the mass operator
  ierr = VecRestoreArray(Mloc, &m); CHKERRQ(ierr);
  ierr = VecScatterBegin(ltog, Mloc, user->M, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(ltog, Mloc, user->M, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecDestroy(&Mloc); CHKERRQ(ierr);
  CeedVectorDestroy(&mceed);

  // Invert diagonally lumped mass vector for RHS function
  ierr = VecReciprocal(user->M); // M is now Minv
  CHKERRQ(ierr);

  // Create and setup TS
  ierr = TSCreate(comm, &ts); CHKERRQ(ierr);
  ierr = TSSetType(ts, TSRK); CHKERRQ(ierr);
  ierr = TSRKSetType(ts, TSRK5F); CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, 500.); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 1.e-5); CHKERRQ(ierr);
  ierr = TSGetAdapt(ts, &adapt); CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt, 1.e-12, 1.e-2); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  if (!contsteps){ // print initial condition
    ierr = TSMonitor_SW(ts, 0, 0., Q, user); CHKERRQ(ierr);
  } else { // continue from time of last output
    PetscReal time;
    PetscInt count;
    PetscViewer viewer;
    char filepath[PETSC_MAX_PATH_LEN];
    ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/sw-time.bin",
                         user->outputfolder); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, filepath, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer, &time, 1, &count, PETSC_REAL);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = TSSetTime(ts, time); CHKERRQ(ierr);
  }
  ierr = TSMonitorSet(ts, TSMonitor_SW, user, NULL); CHKERRQ(ierr);

  // Solve
  ierr = TSSolve(ts, Q); CHKERRQ(ierr);

  // Output Statistics
  ierr = TSGetSolveTime(ts,&ftime); CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "Time integrator took %D time steps to reach final time %g\n",
                     steps,(double)ftime); CHKERRQ(ierr);

  // Clean up libCEED
  CeedVectorDestroy(&qdata);
  CeedVectorDestroy(&user->qceed);
  CeedVectorDestroy(&user->gceed);
  CeedVectorDestroy(&xceed);
  CeedVectorDestroy(&xcorners);
  CeedVectorDestroy(&onesvec);
  CeedBasisDestroy(&basisq);
  CeedBasisDestroy(&basisx);
  CeedBasisDestroy(&basisxc);
  CeedElemRestrictionDestroy(&restrictq);
  CeedElemRestrictionDestroy(&restrictx);
  CeedElemRestrictionDestroy(&restrictqdi);
  CeedElemRestrictionDestroy(&restrictxi);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_ics);
  CeedQFunctionDestroy(&qf_explicit);
  CeedQFunctionDestroy(&qf_implicit);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_ics);
  CeedOperatorDestroy(&op_explicit);
  CeedOperatorDestroy(&op_implicit);
  CeedDestroy(&ceed);

  // Clean up PETSc
  ierr = VecDestroy(&Q); CHKERRQ(ierr);
  ierr = VecDestroy(&user->M); CHKERRQ(ierr);
  ierr = VecDestroy(&user->BC); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Qloc); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Gloc); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ltog); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ltog0); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&gtogD); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ltogX); CHKERRQ(ierr);
  ierr = TSDestroy(&ts); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscFree(user); CHKERRQ(ierr);
  return PetscFinalize();
}
