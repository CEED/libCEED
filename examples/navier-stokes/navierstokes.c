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
#include <petscsys.h>
#include "common.h"
#include "advection.h"
#include "densitycurrent.h"

#if PETSC_VERSION_LT(3,11,0)
#  define VecScatterCreateWithData VecScatterCreate
#endif

// Utility function, compute three factors of an integer
static void Split3(PetscInt size, PetscInt m[3], bool reverse) {
  for (PetscInt d=0,sizeleft=size; d<3; d++) {
    PetscInt try = (PetscInt)PetscCeilReal(PetscPowReal(sizeleft, 1./(3 - d)));
    while (try * (sizeleft / try) != sizeleft) try++;
    m[reverse ? 2-d : d] = try;
    sizeleft /= try;
  }
}

// Utility function, compute the number of DoFs from the global grid
static void GlobalDof(const PetscInt p[3], const PetscInt irank[3],
                      PetscInt degree, const PetscInt melem[3],
                      PetscInt mdof[3]) {
  for (int d=0; d<3; d++)
    mdof[d] = degree*melem[d] + (irank[d] == p[d]-1);
}

// Utility function
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

// Utility function to create local CEED restriction
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

// PETSc user data
typedef struct User_ *User;
struct User_ {
  MPI_Comm comm;
  PetscInt degree;
  PetscInt melem[3];
  PetscInt outputfreq;
  DM dm;
  Ceed ceed;
  CeedVector qceed, gceed, qdata;
  CeedOperator op;
  VecScatter ltog;              // Scatter for all entries
  VecScatter ltog0;             // Skip Dirichlet values for Q
  VecScatter gtogD;             // global-to-global; only Dirichlet values for Q
  Vec Qloc, Gloc, M, BC;
  char outputfolder[PETSC_MAX_PATH_LEN];
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
  CeedOperatorApply(user->op, user->qceed, user->gceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  ierr = VecRestoreArrayRead(user->Qloc, (const PetscScalar**)&q); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Gloc, &g); CHKERRQ(ierr);

  ierr = VecZeroEntries(G); CHKERRQ(ierr);

  // Global-to-global
  // G on the boundary = BC
  ierr = VecScatterBegin(user->gtogD, user->BC, G, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(user->gtogD, user->BC, G, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);

  // Local-to-global
  ierr = VecScatterBegin(user->ltog0, user->Gloc, G, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog0, user->Gloc, G, ADD_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);

  // Inverse of the lumped mass matrix
  ierr = VecPointwiseMult(G,G,user->M); // M is Minv
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// User provided TS Monitor
static PetscErrorCode TSMonitor_NS(TS ts, PetscInt stepno, PetscReal time,
                                   Vec Q, void *ctx) {
  User user = ctx;
  const PetscScalar *q;
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
  ierr = DMDAGetLocalInfo(user->dm, &info); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->dm, U, &u); CHKERRQ(ierr);
  ierr = VecGetArrayRead(Q, &q); CHKERRQ(ierr);
  for (PetscInt i=0; i<info.zm; i++) {
    for (PetscInt j=0; j<info.ym; j++) {
      for (PetscInt k=0; k<info.xm; k++) {
        for (PetscInt c=0; c<5; c++) {
          u[info.zs+i][info.ys+j][(info.xs+k)*5+c] = q[((i*info.ym+j)*info.xm+k)*5 + c];
        }
      }
    }
  }
  ierr = VecRestoreArrayRead(Q, &q); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->dm, U, &u); CHKERRQ(ierr);

  // Output
  ierr = PetscSNPrintf(filepath, sizeof filepath, user->outputfolder, stepno);
  CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)U), filepath,
                            FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = VecView(U, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(user->dm, &U); CHKERRQ(ierr);
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
  char problemtype[256] = "advection";
  PetscInt degree, qextra, localdof, localelem, lsize, outputfreq,
           steps, melem[3], mdof[3], p[3], irank[3], ldof[3];
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
  CeedQFunction qf_setup, qf_mass, qf_ics, qf;
  CeedOperator op_setup, op_mass, op_ics, op;

  // Create the libCEED contexts
  CeedScalar theta0     = 300.;     // K
  CeedScalar thetaC     = -15.;     // K
  CeedScalar P0         = 1.e5;     // kg/m s^2
  CeedScalar N          = 0.01;     // 1/s
  CeedScalar cv         = 717.;     // J/kg K
  CeedScalar cp         = 1004.;    // J/kg K
  CeedScalar Rd         = cp - cv;  // J/kg K
  CeedScalar g          = 9.81;     // m/s^2
  CeedScalar lambda     = -2./3.;   // -
  CeedScalar mu         = 1.e-5;    // Pa s (dynamic viscosity)
  CeedScalar k          = 0.02638;  // W/m K
  CeedScalar rc;                    // m (Radius of bubble)
  PetscScalar lx;                   // m
  PetscScalar ly;                   // m
  PetscScalar lz;                   // m
  PetscScalar resx;                 // m (resolution in x)
  PetscScalar resy;                 // m (resolution in y)
  PetscScalar resz;                 // m (resolution in z)

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  // Allocate PETSc context
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);

  // Set up problem type command line option
  PetscFunctionListAdd(&icsflist, "advection", &ICsAdvection);
  PetscFunctionListAdd(&icsflist, "density_current", &ICsDC);
  PetscFunctionListAdd(&qflist, "advection", &Advection);
  PetscFunctionListAdd(&qflist, "density_current", &DC);

  // Parse command line options
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "Navier-Stokes in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  PetscOptionsFList("-problem", "Problem to solve", NULL, icsflist,
                    problemtype, problemtype, sizeof problemtype, NULL);
  ierr = PetscOptionsScalar("-theta0", "Reference potential temperature",
                         NULL, theta0, &theta0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-thetaC", "Perturbation of potential temperature",
                         NULL, thetaC, &thetaC, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-P0", "Atmospheric pressure",
                         NULL, P0, &P0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-N", "Brunt-Vaisala frequency",
                         NULL, N, &N, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-cv", "Heat capacity at constant volume",
                         NULL, cv, &cv, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-cp", "Heat capacity at constant pressure",
                         NULL, cp, &cp, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-g", "Gravitational acceleration",
                         NULL, g, &g, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lambda", "Stokes hypothesis second viscosity coefficient",
                         NULL, lambda, &lambda, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-mu", "Shear (dynamic) viscosity coefficient",
                         NULL, mu, &mu, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-k", "Thermal conductivity",
                         NULL, k, &k, NULL); CHKERRQ(ierr);
  lx = 1000.;
  ierr = PetscOptionsScalar("-lx", "Length scale in x direction",
                         NULL, lx, &lx, NULL); CHKERRQ(ierr);
  lx = fabs(lx);
  ly = 1000.;
  ierr = PetscOptionsScalar("-ly", "Length scale in y direction",
                         NULL, ly, &ly, NULL); CHKERRQ(ierr);
  ly = fabs(ly);
  lz = 400.;
  ierr = PetscOptionsScalar("-lz", "Length scale in z direction",
                         NULL, lz, &lz, NULL); CHKERRQ(ierr);
  lz = fabs(lz);
  rc = PetscMin(PetscMin(lx,ly),lz)/8.;
  ierr = PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble",
                         NULL, rc, &rc, NULL); CHKERRQ(ierr);
  rc = fabs(rc);
  outputfreq = 10;
  ierr = PetscOptionsInt("-output_freq", "Frequency of output, in number of steps",
                         NULL, outputfreq, &outputfreq, NULL); CHKERRQ(ierr);
  degree = 3;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  qextra = 2;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  localdof = 8*8*8*degree*degree*degree;
  ierr = PetscOptionsInt("-local",
                         "Target number of locally owned degrees of freedom per process",
                         NULL, localdof, &localdof, NULL); CHKERRQ(ierr);
  PetscStrcpy(user->outputfolder, "./");
  ierr = PetscOptionsString("-of", "Output folder",
                            NULL, user->outputfolder, user->outputfolder,
                            sizeof(user->outputfolder), NULL); CHKERRQ(ierr);
  PetscStrcat(user->outputfolder, "/ns-%03D.vts");
  resx = 100.;
  ierr = PetscOptionsScalar("-resx","Resolution in x",
                         NULL, resx, &resx, NULL); CHKERRQ(ierr);
  resx = fabs(resx);
  resy = 100.;
  ierr = PetscOptionsScalar("-resy","Resolution in y",
                         NULL, resy, &resy, NULL); CHKERRQ(ierr);
  resy = fabs(resy);
  resz = 100.;
  ierr = PetscOptionsScalar("-resz","Resolution in z",
                         NULL, resz, &resz, NULL); CHKERRQ(ierr);
  resz = fabs(resz);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Determine size of process grid
  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  Split3(size, p, false);

  // Find a nicely composite number of elements given the resolution
  melem[0] = (PetscInt)(PetscRoundReal(lx / resx));
  melem[1] = (PetscInt)(PetscRoundReal(ly / resy));
  melem[2] = (PetscInt)(PetscRoundReal(lz / resz));
  for (int d=0; d<3; d++) {
    if (melem[d] == 0)
      melem[d]++;
  }
  localelem = melem[0] * melem[1] * melem[2];

  // Find my location in the process grid
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  for (int d=0,rankleft=rank; d<3; d++) {
    const int pstride[3] = {p[1]*p[2], p[2], 1};
    irank[d] = rankleft / pstride[d];
    rankleft -= irank[d] * pstride[d];
  }

  GlobalDof(p, irank, degree, melem, mdof);

  // Set up global state vector
  ierr = VecCreate(comm, &Q); CHKERRQ(ierr);
  ierr = VecSetSizes(Q, 5*mdof[0]*mdof[1]*mdof[2], PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(Q); CHKERRQ(ierr);

  // Set up local state vector
  lsize = 1;
  for (int d=0; d<3; d++) {
    ldof[d] = melem[d]*degree + 1;
    lsize *= ldof[d];
  }
  ierr = VecCreate(PETSC_COMM_SELF, &Qloc); CHKERRQ(ierr);
  ierr = VecSetSizes(Qloc, 5*lsize, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(Qloc); CHKERRQ(ierr);

  // Print grid information
  CeedInt gsize;
  ierr = VecGetSize(Q, &gsize); CHKERRQ(ierr);
  gsize /= 5;
  ierr = PetscPrintf(comm, "Global dofs: %D\n", gsize); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Process decomposition: %D %D %D\n",
                     p[0], p[1], p[2]); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Local elements: %D = %D %D %D\n", localelem,
                     melem[0], melem[1], melem[2]); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Owned dofs: %D = %D %D %D\n",
                     mdof[0]*mdof[1]*mdof[2], mdof[0], mdof[1], mdof[2]);
  CHKERRQ(ierr);

  // Set up global mass vector
  ierr = VecDuplicate(Q,&user->M); CHKERRQ(ierr);

  // Set up local mass vector
  ierr = VecDuplicate(Qloc,&Mloc); CHKERRQ(ierr);

  // Set up global coordinates vector
  ierr = VecCreate(comm, &X); CHKERRQ(ierr);
  ierr = VecSetSizes(X, 3*mdof[0]*mdof[1]*mdof[2], PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(X); CHKERRQ(ierr);

  // Set up local coordinates vector
  ierr = VecCreate(PETSC_COMM_SELF, &Xloc); CHKERRQ(ierr);
  ierr = VecSetSizes(Xloc, 3*lsize, PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(Xloc); CHKERRQ(ierr);

  // Set up global boundary values vector
  ierr = VecDuplicate(Q,&user->BC); CHKERRQ(ierr);

  {
    // Create local-to-global scatters
    PetscInt *ltogind, *ltogind0, *locind, l0count;
    IS ltogis, ltogxis, ltogis0, locis;
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

    // Get indices of dofs except Dirichlet BC dofs
    ierr = PetscMalloc1(lsize, &ltogind); CHKERRQ(ierr);
    ierr = PetscMalloc1(lsize, &ltogind0); CHKERRQ(ierr);
    ierr = PetscMalloc1(lsize, &locind); CHKERRQ(ierr);
    l0count = 0;
    for (PetscInt i=0,ir,ii; ir=i>=mdof[0], ii=i-ir*mdof[0], i<ldof[0]; i++) {
      for (PetscInt j=0,jr,jj; jr=j>=mdof[1], jj=j-jr*mdof[1], j<ldof[1]; j++) {
        for (PetscInt k=0,kr,kk; kr=k>=mdof[2], kk=k-kr*mdof[2], k<ldof[2]; k++) {
          PetscInt dofind = (i*ldof[1]+j)*ldof[2]+k;
          ltogind[dofind] =
            gstart[ir][jr][kr] + (ii*gmdof[ir][jr][kr][1]+jj)*gmdof[ir][jr][kr][2]+kk;
          if ((irank[0] == 0 && i == 0)
              || (irank[1] == 0 && j == 0)
              || (irank[2] == 0 && k == 0)
              || (irank[0]+1 == p[0] && i+1 == ldof[0])
              || (irank[1]+1 == p[1] && j+1 == ldof[1])
              || (irank[2]+1 == p[2] && k+1 == ldof[2]))
            continue;
          ltogind0[l0count] = ltogind[dofind];
          locind[l0count++] = dofind;
        }
      }
    }

    // Create local-to-global scatters
    ierr = ISCreateBlock(comm, 3, lsize, ltogind, PETSC_COPY_VALUES, &ltogxis);
    CHKERRQ(ierr);
    ierr = VecScatterCreateWithData(Xloc, NULL, X, ltogxis, &ltogX);
    CHKERRQ(ierr);
    ierr = ISCreateBlock(comm, 5, lsize, ltogind, PETSC_OWN_POINTER, &ltogis);
    CHKERRQ(ierr);
    ierr = VecScatterCreateWithData(Qloc, NULL, Q, ltogis, &ltog);
    CHKERRQ(ierr);
    ierr = ISCreateBlock(comm, 5, l0count, ltogind0, PETSC_OWN_POINTER, &ltogis0);
    CHKERRQ(ierr);
    ierr = ISCreateBlock(comm, 5, l0count, locind, PETSC_OWN_POINTER, &locis);
    CHKERRQ(ierr);
    ierr = VecScatterCreateWithData(Qloc, locis, Q, ltogis0, &ltog0);
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
        if (q[i] == 1. && (i % 5 == 1 || i % 5 == 2 || i % 5 == 3))
          indD[countD++] = qstart + i;
      }
      ierr = VecRestoreArrayRead(Q, &q); CHKERRQ(ierr);
      ierr = ISCreateGeneral(comm, countD, indD, PETSC_COPY_VALUES, &isD);
      CHKERRQ(ierr);
      ierr = PetscFree(indD); CHKERRQ(ierr);
      ierr = VecScatterCreateWithData(Q, isD, Q, isD, &gtogD); CHKERRQ(ierr);
      ierr = ISDestroy(&isD); CHKERRQ(ierr);
    }
    ierr = ISDestroy(&ltogis); CHKERRQ(ierr);
    ierr = ISDestroy(&ltogxis); CHKERRQ(ierr);
    ierr = ISDestroy(&ltogis0); CHKERRQ(ierr);
    ierr = ISDestroy(&locis); CHKERRQ(ierr);

    {
      // Set up DMDA
      PetscInt *ldofs[3];
      ierr = PetscMalloc3(p[0], &ldofs[0], p[1], &ldofs[1], p[2], &ldofs[2]);
      CHKERRQ(ierr);
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
                          degree*melem[2]*p[2]+1, degree*melem[1]*p[1]+1,
                          degree*melem[0]*p[0]+1,
                          p[2], p[1], p[0], 5, 0,
                          ldofs[2], ldofs[1], ldofs[0], &dm); CHKERRQ(ierr);
      ierr = PetscFree3(ldofs[0], ldofs[1], ldofs[2]); CHKERRQ(ierr);
      ierr = DMSetUp(dm); CHKERRQ(ierr);
      ierr = DMDASetFieldName(dm, 0, "Density"); CHKERRQ(ierr);
      ierr = DMDASetFieldName(dm, 1, "MomentumX"); CHKERRQ(ierr);
      ierr = DMDASetFieldName(dm, 2, "MomentumY"); CHKERRQ(ierr);
      ierr = DMDASetFieldName(dm, 3, "MomentumZ"); CHKERRQ(ierr);
      ierr = DMDASetFieldName(dm, 4, "Total Energy"); CHKERRQ(ierr);
    }
  }

  // Set up CEED
  // CEED Bases
  CeedInit(ceedresource, &ceed);
  numP = degree + 1;
  numQ = numP + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 5, numP, numQ, CEED_GAUSS, &basisq);
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 3, 2, numQ, CEED_GAUSS, &basisx);
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 3, 2, numP, CEED_GAUSS_LOBATTO,
                                  &basisxc);

  // CEED Restrictions
  CreateRestriction(ceed, melem, numP, 5, &restrictq);
  CreateRestriction(ceed, melem, 2, 3, &restrictx);
  CreateRestriction(ceed, melem, numP, 3, &restrictxc);
  CreateRestriction(ceed, melem, numP, 1, &restrictmult);
  CeedInt nelem = melem[0]*melem[1]*melem[2];
  CeedElemRestrictionCreateIdentity(ceed, nelem, 16*numQ*numQ*numQ,
                                    16*nelem*numQ*numQ*numQ, 1,
                                    &restrictqdi);
  CeedElemRestrictionCreateIdentity(ceed, nelem, numQ*numQ*numQ,
                                    nelem*numQ*numQ*numQ, 1,
                                    &restrictxi);

  // Find physical cordinates of the corners of local elements
  {
    CeedScalar *xloc;
    CeedInt shape[3] = {melem[0]+1, melem[1]+1, melem[2]+1}, len =
                         shape[0]*shape[1]*shape[2];
    xloc = malloc(len*3*sizeof xloc[0]);
    for (CeedInt i=0; i<shape[0]; i++) {
      for (CeedInt j=0; j<shape[1]; j++) {
        for (CeedInt k=0; k<shape[2]; k++) {
          xloc[((i*shape[1]+j)*shape[2]+k) + 0*len] = lx*(irank[0]*melem[0]+i) /
              (p[0]*melem[0]);
          xloc[((i*shape[1]+j)*shape[2]+k) + 1*len] = ly*(irank[1]*melem[1]+j) /
              (p[1]*melem[1]);
          xloc[((i*shape[1]+j)*shape[2]+k) + 2*len] = lz*(irank[2]*melem[2]+k) /
              (p[2]*melem[2]);
        }
      }
    }
    CeedVectorCreate(ceed, len*3, &xcorners);
    CeedVectorSetArray(xcorners, CEED_MEM_HOST, CEED_OWN_POINTER, xloc);
  }

  // Create the CEED vectors that will be needed in setup
  CeedInt Nqpts, Nelem = melem[0]*melem[1]*melem[2];
  CeedBasisGetNumQuadraturePoints(basisq, &Nqpts);
  CeedInt Ndofs = 1;
  for (int d=0; d<3; d++) Ndofs *= numP;
  CeedVectorCreate(ceed, 16*Nelem*Nqpts, &qdata);
  CeedVectorCreate(ceed, 5*lsize, &q0ceed);
  CeedVectorCreate(ceed, 5*lsize, &mceed);
  CeedVectorCreate(ceed, 5*lsize, &onesvec);
  CeedVectorCreate(ceed, 3*lsize, &xceed);
  CeedVectorCreate(ceed, lsize, &multlvec);
  CeedVectorCreate(ceed, Nelem*Ndofs, &multevec);

  // Find multiplicity of each local point
  CeedVectorSetValue(multevec, 1.0);
  CeedVectorSetValue(multlvec, 0.);
  CeedElemRestrictionApply(restrictmult, CEED_TRANSPOSE, CEED_TRANSPOSE,
                           multevec, multlvec, CEED_REQUEST_IMMEDIATE);

  // Create the Q-function that builds the quadrature data for the NS operator
  CeedQFunctionCreateInterior(ceed, 1,
                              Setup, __FILE__ ":Setup", &qf_setup);
  CeedQFunctionAddInput(qf_setup, "dx", 3, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", 16, CEED_EVAL_NONE);

  // Create the Q-function that defines the action of the mass operator
  CeedQFunctionCreateInterior(ceed, 1,
                              Mass, __FILE__ ":Mass", &qf_mass);
  CeedQFunctionAddInput(qf_mass, "q", 5, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "qdata", 16, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_mass, "v", 5, CEED_EVAL_INTERP);

  // Create the Q-function that sets the ICs of the operator
  void (*icsfp)(void);
  PetscFunctionListFind(icsflist, problemtype, &icsfp);
  if (!icsfp)
      return CeedError(ceed, 1, "Function not found in the list");
  char str[256] = __FILE__":ICs";
  strcat(str, problemtype);
  CeedQFunctionCreateInterior(ceed, 1,
                              (int(*)(void *, CeedInt, const CeedScalar *const *, CeedScalar *const *))icsfp, str, &qf_ics);
  CeedQFunctionAddInput(qf_ics, "x", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ics, "q0", 5, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_ics, "coords", 3, CEED_EVAL_NONE);

  // Create the Q-function that defines the action of the operator
  void (*fp)(void);
  PetscFunctionListFind(qflist, problemtype, &fp);
  if (!fp)
      return CeedError(ceed, 1, "Function not found in the list");
  strcpy(str, __FILE__":");
  strcat(str, problemtype);
  CeedQFunctionCreateInterior(ceed, 1,
                              (int(*)(void *, CeedInt, const CeedScalar *const *, CeedScalar *const *))fp, str, &qf);
  CeedQFunctionAddInput(qf, "q", 5, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf, "dq", 5, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf, "qdata", 16, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf, "x", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf, "v", 5, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf, "dv", 5, CEED_EVAL_GRAD);

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
  CeedOperatorSetField(op_ics, "coords", restrictxc, CEED_TRANSPOSE,
                       CEED_BASIS_COLLOCATED, xceed);

  // Create the physics operator
  CeedOperatorCreate(ceed, qf, NULL, NULL, &op);
  CeedOperatorSetField(op, "q", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op, "dq", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op, "qdata", restrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op, "x", restrictx, CEED_NOTRANSPOSE,
                       basisx, xcorners);
  CeedOperatorSetField(op, "v", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op, "dv", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);

  // Set up the libCEED context
  CeedScalar ctxSetup[12] = {theta0, thetaC, P0, N, cv, cp, Rd, g, rc, lx, ly, lz};
  CeedQFunctionSetContext(qf_ics, &ctxSetup, sizeof ctxSetup);
  CeedScalar ctxNS[6] = {lambda, mu, k, cv, cp, g};
  CeedQFunctionSetContext(qf, &ctxNS, sizeof ctxNS);

  // Set up PETSc context
  user->comm = comm;
  user->degree = degree;
  for (int d=0; d<3; d++) user->melem[d] = melem[d];
  user->outputfreq = outputfreq;
  user->dm = dm;
  user->ceed = ceed;
  CeedVectorCreate(ceed, 5*lsize, &user->qceed);
  CeedVectorCreate(ceed, 5*lsize, &user->gceed);
  user->op = op;
  user->qdata = qdata;
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
    for (PetscInt f=0; f<5; f++)
      q0[i*5+f] /= mult[i];
    for (PetscInt d=0; d<3; d++)
      x[i*3+d] /= mult[i];
  }

  CeedVectorRestoreArray(q0ceed, &q0);
  CeedVectorRestoreArray(xceed, &x);
  CeedVectorRestoreArray(multlvec, &mult);

  // Destroy mult vecs
  CeedVectorDestroy(&multevec);
  CeedVectorDestroy(&multlvec);

  // Gather initial Q values
  ierr = VecRestoreArray(Qloc, &q0); CHKERRQ(ierr);
  ierr = VecScatterBegin(ltog, Qloc, Q, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(ltog, Qloc, Q, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
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
  ierr = TSMonitor_NS(ts, 0, 0., Q, user); CHKERRQ(ierr);
  ierr = TSMonitorSet(ts, TSMonitor_NS, user, NULL); CHKERRQ(ierr);

  // Solve
  ierr = TSSolve(ts, Q); CHKERRQ(ierr);

  // Output Statistics
  ierr = TSGetSolveTime(ts,&ftime); CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "Time integrator took %D time steps to reach final time %g\n",
                     steps,(double)ftime); CHKERRQ(ierr);

  // Clean up libCEED
  CeedVectorDestroy(&user->qceed);
  CeedVectorDestroy(&user->gceed);
  CeedVectorDestroy(&user->qdata);
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
  CeedQFunctionDestroy(&qf);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_ics);
  CeedOperatorDestroy(&op);
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
