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
//     make [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>] nsplex
//
// Sample runs:
//
//     nsplex
//     nsplex -ceed -problem density_current /cpu/self
//     nsplex -ceed -problem advection /gpu/occa
//

/// @file
/// Navier-Stokes example using PETSc

const char help[] = "Solve Navier-Stokes using PETSc and libCEED\n";

#include <petscts.h>
#include <petscdmplex.h>
#include <ceed.h>
#include <stdbool.h>
#include <petscsys.h>
#include "common.h"
#include "advection.h"
#include "advection2d.h"
#include "densitycurrent.h"

// Problem Options
typedef enum {
  NS_DENSITY_CURRENT = 0,
  NS_ADVECTION = 1,
  NS_ADVECTION2D = 2,
} problemType;
static const char *const problemTypes[] = {
  "density_current",
  "advection",
  "advection2d",
  "problemType","NS_",0
};

typedef enum {
  STAB_NONE = 0,
  STAB_SU = 1,   // Streamline Upwind
  STAB_SUPG = 2, // Streamline Upwind Petrov-Galerkin
} StabilizationType;
static const char *const StabilizationTypes[] = {
  "NONE",
  "SU",
  "SUPG",
  "StabilizationType", "STAB_", NULL
};

// Problem specific data
typedef struct {
  CeedInt dim, qdatasize;
  CeedQFunctionUser setup, ics, apply_rhs, apply_ifunction;
  PetscErrorCode (*bc)(PetscInt, PetscReal, const PetscReal[], PetscInt,
                       PetscScalar[], void*);
  const char *setup_loc, *ics_loc, *apply_rhs_loc, *apply_ifunction_loc;
} problemData;

problemData problemOptions[] = {
  [NS_DENSITY_CURRENT] = {
    .dim = 3,
    .qdatasize = 16,
    .setup = Setup,
    .setup_loc = Setup_loc,
    .ics = ICsDC,
    .apply_rhs = DC,
    .bc = NULL,
    .ics_loc = ICsDC_loc,
    .apply_rhs_loc = DC_loc
  },
  [NS_ADVECTION] = {
    .dim = 3,
    .qdatasize = 16,
    .setup = Setup,
    .setup_loc = Setup_loc,
    .ics = ICsAdvection,
    .apply_rhs = Advection,
    .bc = NULL,
    .ics_loc = ICsAdvection_loc,
    .apply_rhs_loc = Advection_loc,
  },
  [NS_ADVECTION2D] = {
    .dim = 2,
    .qdatasize = 5,
    .setup = Setup2d,
    .setup_loc = Setup2d_loc,
    .ics = ICsAdvection2d,
    .ics_loc = ICsAdvection2d_loc,
    .apply_rhs = Advection2d,
    .apply_rhs_loc = Advection2d_loc,
    .apply_ifunction = IFunction_Advection2d,
    .apply_ifunction_loc = IFunction_Advection2d_loc,
    .bc = NULL,
  },
};

// Essential BC dofs are encoded in closure indices as -(i+1).
static PetscInt Involute(PetscInt i) {
  return i >= 0 ? i : -(i+1);
}

// Utility function to create local CEED restriction
static PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
                                                CeedElemRestriction *Erestrict) {

  PetscSection   section;
  PetscInt       c, cStart, cEnd, Nelem, Ndof, *erestrict, eoffset, nfields, dim;
  PetscErrorCode ierr;
  Vec Uloc;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetSection(dm,&section);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &nfields);CHKERRQ(ierr);
  PetscInt ncomp[nfields], fieldoff[nfields+1];
  fieldoff[0] = 0;
  for (PetscInt f=0; f<nfields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &ncomp[f]);CHKERRQ(ierr);
    fieldoff[f+1] = fieldoff[f] + ncomp[f];
  }

  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  Nelem = cEnd - cStart;
  ierr = PetscMalloc1(Nelem*PetscPowInt(P, dim), &erestrict);CHKERRQ(ierr);
  for (c=cStart,eoffset=0; c<cEnd; c++) {
    PetscInt numindices, *indices, nnodes;
    ierr = DMPlexGetClosureIndices(dm,section,section,c,&numindices,&indices,NULL);CHKERRQ(ierr);
    if (numindices % fieldoff[nfields]) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of closure indices not compatible with Cell %D",c);
    nnodes = numindices / fieldoff[nfields];
    for (PetscInt i=0; i<nnodes; i++) {
      // Check that indices are blocked by node and thus can be coalesced as a single field with
      // fieldoff[nfields] = sum(ncomp) components.
      for (PetscInt f=0; f<nfields; f++) {
        for (PetscInt j=0; j<ncomp[f]; j++) {
          if (Involute(indices[fieldoff[f]*nnodes + i*ncomp[f] + j])
              != Involute(indices[i*ncomp[0]]) + fieldoff[f] + j)
            SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell %D closure indices not interlaced for node %D field %D component %D",c,i,f,j);
        }
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = Involute(indices[i*ncomp[0]]);
      erestrict[eoffset++] = loc / fieldoff[nfields];
    }
    ierr = DMPlexRestoreClosureIndices(dm,section,section,c,&numindices,&indices,NULL);CHKERRQ(ierr);
  }
  if (eoffset != Nelem*PetscPowInt(P, dim)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_LIB,"ElemRestriction of size (%D,%D) initialized %D nodes",Nelem,PetscPowInt(P, dim),eoffset);
  ierr = DMGetLocalVector(dm, &Uloc);CHKERRQ(ierr);
  ierr = VecGetLocalSize(Uloc, &Ndof);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Uloc);CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, Nelem, PetscPowInt(P, dim), Ndof/fieldoff[nfields], fieldoff[nfields],
                            CEED_MEM_HOST, CEED_COPY_VALUES, erestrict, Erestrict);
  ierr = PetscFree(erestrict);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static int CreateVectorFromPetscVec(Ceed ceed, Vec p, CeedVector *v) {
  PetscErrorCode ierr;
  PetscInt m;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(p, &m);CHKERRQ(ierr);
  ierr = CeedVectorCreate(ceed, m, v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static int VectorPlacePetscVec(CeedVector c, Vec p) {
  PetscErrorCode ierr;
  PetscInt mceed,mpetsc;
  PetscScalar *a;

  PetscFunctionBeginUser;
  ierr = CeedVectorGetLength(c, &mceed);CHKERRQ(ierr);
  ierr = VecGetLocalSize(p, &mpetsc);CHKERRQ(ierr);
  if (mceed != mpetsc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cannot place PETSc Vec of length %D in CeedVector of length %D",mpetsc,mceed);
  ierr = VecGetArray(p, &a);CHKERRQ(ierr);
  CeedVectorSetArray(c, CEED_MEM_HOST, CEED_USE_POINTER, a);
  PetscFunctionReturn(0);
}

// PETSc user data
typedef struct User_ *User;
typedef struct Units_ *Units;

struct User_ {
  MPI_Comm comm;
  PetscInt outputfreq;
  DM dm;
  Ceed ceed;
  Units units;
  CeedVector qceed, qdotceed, gceed;
  CeedOperator op_rhs, op_ifunction;
  Vec M;
  char outputfolder[PETSC_MAX_PATH_LEN];
  PetscInt contsteps;
};

struct Units_ {
  // fundamental units
  PetscScalar meter;
  PetscScalar kilogram;
  PetscScalar second;
  PetscScalar Kelvin;
  // derived units
  PetscScalar Pascal;
  PetscScalar JperkgK;
  PetscScalar mpersquareds;
  PetscScalar WpermK;
  PetscScalar kgpercubicm;
  PetscScalar kgpersquaredms;
  PetscScalar Joulepercubicm;
};

static PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm, PetscBool insertEssential, Vec Qloc, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM) {
  PetscErrorCode ierr;
  Vec Qbc;

  PetscFunctionBegin;
  ierr = DMGetNamedLocalVector(dm, "Qbc", &Qbc);CHKERRQ(ierr);
  ierr = VecAXPY(Qloc, 1., Qbc);CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(dm, "Qbc", &Qbc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// This is the RHS of the ODE, given as u_t = G(t,u)
// This function takes in a state vector Q and writes into G
static PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *userData) {
  PetscErrorCode ierr;
  User user = *(User *)userData;
  PetscScalar *q, *g;
  Vec Qloc, Gloc;

  // Global-to-local
  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(user->dm, &Qloc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Gloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Qloc, 0.0,
                                    NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = VecZeroEntries(Gloc); CHKERRQ(ierr);

  // Ceed Vectors
  ierr = VecGetArrayRead(Qloc, (const PetscScalar**)&q); CHKERRQ(ierr);
  ierr = VecGetArray(Gloc, &g); CHKERRQ(ierr);
  CeedVectorSetArray(user->qceed, CEED_MEM_HOST, CEED_USE_POINTER, q);
  CeedVectorSetArray(user->gceed, CEED_MEM_HOST, CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_rhs, user->qceed, user->gceed, CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  ierr = VecRestoreArrayRead(Qloc, (const PetscScalar**)&q); CHKERRQ(ierr);
  ierr = VecRestoreArray(Gloc, &g); CHKERRQ(ierr);

  ierr = VecZeroEntries(G); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Gloc, ADD_VALUES, G);CHKERRQ(ierr);

  // Inverse of the lumped mass matrix
  ierr = VecPointwiseMult(G,G,user->M); // M is Minv
  CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(user->dm, &Qloc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Gloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Qdot, Vec G, void *userData) {
  PetscErrorCode ierr;
  User user = *(User *)userData;
  const PetscScalar *q, *qdot;
  PetscScalar *g;
  Vec Qloc, Qdotloc, Gloc;

  // Global-to-local
  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(user->dm, &Qloc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Qdotloc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Gloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Qloc, 0.0,
                                    NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = VecZeroEntries(Qdotloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Qdot, INSERT_VALUES, Qdotloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(Gloc); CHKERRQ(ierr);

  // Ceed Vectors
  ierr = VecGetArrayRead(Qloc, &q); CHKERRQ(ierr);
  ierr = VecGetArrayRead(Qdotloc, &qdot); CHKERRQ(ierr);
  ierr = VecGetArray(Gloc, &g); CHKERRQ(ierr);
  CeedVectorSetArray(user->qceed, CEED_MEM_HOST, CEED_USE_POINTER, (PetscScalar*)q);
  CeedVectorSetArray(user->qdotceed, CEED_MEM_HOST, CEED_USE_POINTER, (PetscScalar*)qdot);
  CeedVectorSetArray(user->gceed, CEED_MEM_HOST, CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_ifunction, user->qceed, user->gceed, CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  ierr = VecRestoreArrayRead(Qloc, &q); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Qdotloc, &qdot); CHKERRQ(ierr);
  ierr = VecRestoreArray(Gloc, &g); CHKERRQ(ierr);

  ierr = VecZeroEntries(G); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Gloc, ADD_VALUES, G);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(user->dm, &Qloc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Qdotloc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Gloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// User provided TS Monitor
static PetscErrorCode TSMonitor_NS(TS ts, PetscInt stepno, PetscReal time,
                                   Vec Q, void *ctx) {
  User user = ctx;
  const PetscReal scale[] = {1/user->units->kgpercubicm,
                             1/user->units->kgpersquaredms,
                             1/user->units->kgpersquaredms,
                             1/user->units->kgpersquaredms,
                             1/user->units->Joulepercubicm
                            };
  PetscScalar *qloc;
  Vec Qloc;
  PetscInt m;
  char filepath[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  PetscErrorCode ierr;

  // Set up output
  PetscFunctionBeginUser;
  // Print every 'outputfreq' steps
  if (stepno % user->outputfreq != 0)
    PetscFunctionReturn(0);
  ierr = DMGetLocalVector(user->dm, &Qloc);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Qloc, "StateVec"); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);
  if (0) {
    // Default scaling is by 1.0.  This should really be done by PETSc so it can be applied after applying boundary
    // conditions, which is done in VecView to the VTK viewer below.
    ierr = VecStrideScaleAll(Qloc, scale); CHKERRQ(ierr);
  }

  // Output
  ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/ns-%03D.vtu",
                       user->outputfolder, stepno + user->contsteps);
  CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q), filepath,
                            FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  ierr = VecView(Qloc, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Qloc); CHKERRQ(ierr);

  // Save data in a binary file for continuation of simulations
  ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/ns-solution.bin",
                       user->outputfolder); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm, filepath, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = VecView(Q, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  // Save time stamp
  // Dimensionalize time back
  time /= user->units->second;
  ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/ns-time.bin",
                       user->outputfolder); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm, filepath, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL, true);
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeLumpedMassMatrix(Ceed ceed, DM dm,
                                              CeedElemRestriction restrictq,
                                              CeedBasis basisq,
                                              CeedElemRestriction restrictqdi,
                                              CeedVector qdata,
                                              Vec M) {
  PetscErrorCode ierr;
  CeedQFunction qf_mass;
  CeedOperator op_mass;
  CeedVector mceed, onesvec;
  Vec Mloc;
  CeedInt ncompq, qdatasize;

  PetscFunctionBeginUser;
  CeedElemRestrictionGetNumComponents(restrictq, &ncompq);
  CeedElemRestrictionGetNumComponents(restrictqdi, &qdatasize);
  // Create the Q-function that defines the action of the mass operator
  CeedQFunctionCreateInterior(ceed, 1,
                              Mass, __FILE__ ":Mass", &qf_mass);
  CeedQFunctionAddInput(qf_mass, "q", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_mass, "v", ncompq, CEED_EVAL_INTERP);

  // Create the mass operator
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "q", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", restrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_mass, "v", restrictq, CEED_TRANSPOSE,
                       basisq, CEED_VECTOR_ACTIVE);

  ierr = DMGetLocalVector(dm, &Mloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Mloc);CHKERRQ(ierr);
  CeedElemRestrictionCreateVector(restrictq, &mceed, NULL);
  ierr = VectorPlacePetscVec(mceed, Mloc);CHKERRQ(ierr);

  { // Compute a lumped mass matrix
    CeedVector onesvec;
    CeedElemRestrictionCreateVector(restrictq, &onesvec, NULL);
    CeedVectorSetValue(onesvec, 1.0);
    CeedOperatorApply(op_mass, onesvec, mceed, CEED_REQUEST_IMMEDIATE);
    CeedVectorDestroy(&onesvec);
    CeedOperatorDestroy(&op_mass);
    CeedVectorDestroy(&mceed);
  }
  CeedQFunctionDestroy(&qf_mass);

  ierr = VecZeroEntries(M); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, Mloc, ADD_VALUES, M); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Mloc); CHKERRQ(ierr);

  // Invert diagonally lumped mass vector for RHS function
  ierr = VecReciprocal(M); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  DM dm, dmcoord;
  TS ts;
  TSAdapt adapt;
  User user;
  Units units;
  char ceedresource[4096] = "/cpu/self";
  PetscInt cStart, cEnd, localNelem, lnodes, steps,
           mdof[3], p[3], irank[3], ldof[3];
  const PetscInt ncompq = 5;
  PetscMPIInt size, rank;
  PetscScalar ftime;
  PetscScalar *q0, *m, *x;
  const PetscScalar *mult;
  Vec Q, Qloc, Xloc;
  Ceed ceed;
  CeedInt numP, numQ;
  CeedVector xcorners, qdata, q0ceed, xceed;
  CeedBasis basisx, basisxc, basisq;
  CeedElemRestriction restrictx, restrictxi, restrictxcoord,
                      restrictq, restrictqdi;
  CeedQFunction qf_setup, qf_mass, qf_ics, qf_rhs, qf_ifunction;
  CeedOperator op_setup, op_ics;
  CeedScalar Rd;
  PetscScalar WpermK, Pascal, JperkgK, mpersquareds, kgpercubicm,
              kgpersquaredms, Joulepercubicm;
  problemType problemChoice;
  problemData *problem;
  StabilizationType stab;
  PetscBool   implicit;

  // Create the libCEED contexts
  PetscScalar meter     = 1e-2;     // 1 meter in scaled length units
  PetscScalar second    = 1e-2;     // 1 second in scaled time units
  PetscScalar kilogram  = 1e-6;     // 1 kilogram in scaled mass units
  PetscScalar Kelvin    = 1;        // 1 Kelvin in scaled temperature units
  CeedScalar theta0     = 300.;     // K
  CeedScalar thetaC     = -15.;     // K
  CeedScalar P0         = 1.e5;     // Pa
  CeedScalar N          = 0.01;     // 1/s
  CeedScalar cv         = 717.;     // J/(kg K)
  CeedScalar cp         = 1004.;    // J/(kg K)
  CeedScalar g          = 9.81;     // m/s^2
  CeedScalar lambda     = -2./3.;   // -
  CeedScalar mu         = 75.;      // Pa s, dynamic viscosity
  // mu = 75 is not physical for air, but is good for numerical stability
  CeedScalar k          = 0.02638;  // W/(m K)
  CeedScalar CtauS      = 0.;       // dimensionless
  CeedScalar strong_form = 0.;      // [0,1]
  PetscScalar lx        = 8000.;    // m
  PetscScalar ly        = 8000.;    // m
  PetscScalar lz        = 4000.;    // m
  CeedScalar rc         = 1000.;    // m (Radius of bubble)
  PetscScalar resx      = 1000.;    // m (resolution in x)
  PetscScalar resy      = 1000.;    // m (resolution in y)
  PetscScalar resz      = 1000.;    // m (resolution in z)
  PetscInt outputfreq   = 10;       // -
  PetscInt contsteps    = 0;        // -
  PetscInt degree;
  PetscInt qextra       = 2;        // -
  DMBoundaryType periodicity[] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  // Allocate PETSc context
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &units); CHKERRQ(ierr);

  // Parse command line options
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "Navier-Stokes in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  problemChoice = NS_DENSITY_CURRENT;
  ierr = PetscOptionsEnum("-problem", "Problem to solve", NULL,
                          problemTypes, (PetscEnum)problemChoice,
                          (PetscEnum *)&problemChoice, NULL); CHKERRQ(ierr);
  problem = &problemOptions[problemChoice];
  ierr = PetscOptionsEnum("-stab", "Stabilization method", NULL,
                          StabilizationTypes, (PetscEnum)(stab = STAB_NONE),
                          (PetscEnum *)&stab, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation",
                          NULL, implicit=PETSC_FALSE, &implicit, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, meter, &meter, NULL); CHKERRQ(ierr);
  meter = fabs(meter);
  ierr = PetscOptionsScalar("-units_second","1 second in scaled time units",
                            NULL, second, &second, NULL); CHKERRQ(ierr);
  second = fabs(second);
  ierr = PetscOptionsScalar("-units_kilogram","1 kilogram in scaled mass units",
                            NULL, kilogram, &kilogram, NULL); CHKERRQ(ierr);
  kilogram = fabs(kilogram);
  ierr = PetscOptionsScalar("-units_Kelvin",
                            "1 Kelvin in scaled temperature units",
                            NULL, Kelvin, &Kelvin, NULL); CHKERRQ(ierr);
  Kelvin = fabs(Kelvin);
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
  ierr = PetscOptionsScalar("-lambda",
                            "Stokes hypothesis second viscosity coefficient",
                            NULL, lambda, &lambda, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-mu", "Shear dynamic viscosity coefficient",
                            NULL, mu, &mu, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-k", "Thermal conductivity",
                            NULL, k, &k, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-CtauS", "Scale coefficient for tau (nondimensional)",
                            NULL, CtauS, &CtauS, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-strong_form", "Strong (1) or weak/integrated by parts (0) advection residual",
                            NULL, strong_form, &strong_form, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lx", "Length scale in x direction",
                            NULL, lx, &lx, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-ly", "Length scale in y direction",
                            NULL, ly, &ly, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-lz", "Length scale in z direction",
                            NULL, lz, &lz, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble",
                            NULL, rc, &rc, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-resx","Target resolution in x",
                            NULL, resx, &resx, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-resy","Target resolution in y",
                            NULL, resy, &resy, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-resz","Target resolution in z",
                            NULL, resz, &resz, NULL); CHKERRQ(ierr);
  PetscInt n = problem->dim;
  ierr = PetscOptionsEnumArray("-periodicity", "Periodicity per direction",
                               NULL, DMBoundaryTypes, (PetscEnum *)periodicity,
                               &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-output_freq", "Frequency of output, in number of steps",
                         NULL, outputfreq, &outputfreq, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-continue", "Continue from previous solution",
                         NULL, contsteps, &contsteps, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  PetscStrncpy(user->outputfolder, ".", 2);
  ierr = PetscOptionsString("-of", "Output folder",
                            NULL, user->outputfolder, user->outputfolder,
                            sizeof(user->outputfolder), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Define derived units
  Pascal = kilogram / (meter * PetscSqr(second));
  JperkgK =  PetscSqr(meter) / (PetscSqr(second) * Kelvin);
  mpersquareds = meter / PetscSqr(second);
  WpermK = kilogram * meter / (pow(second,3) * Kelvin);
  kgpercubicm = kilogram / pow(meter,3);
  kgpersquaredms = kilogram / (PetscSqr(meter) * second);
  Joulepercubicm = kilogram / (meter * PetscSqr(second));

  // Scale variables to desired units
  theta0 *= Kelvin;
  thetaC *= Kelvin;
  P0 *= Pascal;
  N *= (1./second);
  cv *= JperkgK;
  cp *= JperkgK;
  Rd = cp - cv;
  g *= mpersquareds;
  mu *= Pascal * second;
  k *= WpermK;
  lx = fabs(lx) * meter;
  ly = fabs(ly) * meter;
  lz = fabs(lz) * meter;
  rc = fabs(rc) * meter;
  resx = fabs(resx) * meter;
  resy = fabs(resy) * meter;
  resz = fabs(resz) * meter;

  const CeedInt dim = problem->dim, ncompx = problem->dim, qdatasize = problem->qdatasize;
  // Set up the libCEED context
  CeedScalar ctxSetup[] = {theta0, thetaC, P0, N, cv, cp, Rd, g, rc,
                           lx, ly, lz,
                           periodicity[0], periodicity[1], periodicity[2],
                          };

  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, NULL, NULL, (PetscReal[]){lx, ly, lz}, periodicity, PETSC_TRUE, &dm);CHKERRQ(ierr);
  if (1) {
    DM               dmDist = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm  = dmDist;
    }
  }
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = DMLocalizeCoordinates(dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  { // Configure the finite element space and boundary conditions
    PetscFE fe;
    PetscSpace fespace;
    ierr = PetscFECreateDefault(PETSC_COMM_SELF,dim,ncompq,PETSC_FALSE,NULL,PETSC_DETERMINE,&fe);CHKERRQ(ierr);
    ierr = DMAddField(dm,NULL,(PetscObject)fe);CHKERRQ(ierr);

    ierr = DMCreateDS(dm);CHKERRQ(ierr);
    ierr = DMAddBoundary(dm,DM_BC_ESSENTIAL,"wall","marker",0,0,NULL,(void(*)(void))problem->bc,1,(PetscInt[]){1},ctxSetup);CHKERRQ(ierr);
    ierr = DMPlexSetClosurePermutationTensor(dm,PETSC_DETERMINE,NULL);CHKERRQ(ierr);
    ierr = PetscFEGetBasisSpace(fe, &fespace);CHKERRQ(ierr);
    ierr = PetscSpaceGetDegree(fespace, &degree, NULL);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  }
  { // Empty name for conserved field
    PetscSection section;
    ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, 0, ""); CHKERRQ(ierr);
  }
  ierr = DMCreateGlobalVector(dm, &Q);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &Qloc);CHKERRQ(ierr);
  ierr = VecGetSize(Qloc, &lnodes);CHKERRQ(ierr);
  lnodes /= ncompq;

  {  // Print grid information
    CeedInt gnodes, onodes;
    int comm_size;
    ierr = VecGetSize(Q, &gnodes); CHKERRQ(ierr);
    gnodes /= ncompq;
    ierr = VecGetLocalSize(Q, &onodes); CHKERRQ(ierr);
    onodes /= ncompq;
    ierr = MPI_Comm_size(comm, &comm_size); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Global FEM nodes: %d on %d ranks\n", gnodes, comm_size); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Local FEM nodes: %d (%d owned)\n", lnodes, onodes); CHKERRQ(ierr);
  }

  // Set up global mass vector
  ierr = VecDuplicate(Q,&user->M); CHKERRQ(ierr);

  // Set up CEED
  // CEED Bases
  CeedInit(ceedresource, &ceed);
  numP = degree + 1;
  numQ = numP + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompq, numP, numQ, CEED_GAUSS,
                                  &basisq);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, numQ, CEED_GAUSS,
                                  &basisx);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, numP,
                                  CEED_GAUSS_LOBATTO, &basisxc);

  ierr = DMGetCoordinateDM(dm, &dmcoord);CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord,PETSC_DETERMINE,NULL);CHKERRQ(ierr);

  // CEED Restrictions
  ierr = CreateRestrictionFromPlex(ceed, dm, degree+1, &restrictq);CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dmcoord, 2, &restrictx);CHKERRQ(ierr);
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  localNelem = cEnd - cStart;
  CeedInt numQdim = CeedIntPow(numQ, dim);
  CeedElemRestrictionCreateIdentity(ceed, localNelem, numQdim,
                                    localNelem*numQdim, qdatasize,
                                    &restrictqdi);
  CeedElemRestrictionCreateIdentity(ceed, localNelem, numQdim,
                                    localNelem*numQdim, 1,
                                    &restrictxi);
  CeedElemRestrictionCreateIdentity(ceed, localNelem, PetscPowInt(numP, dim),
                                    localNelem*PetscPowInt(numP, dim), ncompx,
                                    &restrictxcoord);

  ierr = DMGetCoordinatesLocal(dm, &Xloc);CHKERRQ(ierr);
  ierr = CreateVectorFromPetscVec(ceed, Xloc, &xcorners);CHKERRQ(ierr);

  // Create the CEED vectors that will be needed in setup
  CeedInt Nqpts, Nnodes;
  CeedBasisGetNumQuadraturePoints(basisq, &Nqpts);
  CeedInt Ndofs = 1;
  for (int d=0; d<3; d++) Ndofs *= numP;
  CeedVectorCreate(ceed, qdatasize*localNelem*Nqpts, &qdata);
  CeedElemRestrictionCreateVector(restrictq, &q0ceed, NULL);
  CeedElemRestrictionCreateVector(restrictxcoord, &xceed, NULL);

  // Create the Q-function that builds the quadrature data for the NS operator
  CeedQFunctionCreateInterior(ceed, 1, problem->setup, problem->setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "dx", ncompx*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", qdatasize, CEED_EVAL_NONE);

  // Create the Q-function that sets the ICs of the operator
  CeedQFunctionCreateInterior(ceed, 1, problem->ics, problem->ics_loc, &qf_ics);
  CeedQFunctionAddInput(qf_ics, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ics, "q0", ncompq, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_ics, "coords", ncompx, CEED_EVAL_NONE);

  // Create the Q-function that defines the action of the RHS operator
  CeedQFunctionCreateInterior(ceed, 1, problem->apply_rhs,
                              problem->apply_rhs_loc, &qf_rhs);
  CeedQFunctionAddInput(qf_rhs, "q", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_rhs, "dq", ncompq*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_rhs, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_rhs, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_rhs, "v", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_rhs, "dv", ncompq*dim, CEED_EVAL_GRAD);

  // Create the Q-function that defines the action of the IFunction
  CeedQFunctionCreateInterior(ceed, 1, problem->apply_ifunction,
                              problem->apply_ifunction_loc, &qf_ifunction);
  CeedQFunctionAddInput(qf_ifunction, "q", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_ifunction, "dq", ncompq*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_ifunction, "qdot", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_ifunction, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_ifunction, "v", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ifunction, "dv", ncompq*dim, CEED_EVAL_GRAD);

  // Create the operator that builds the quadrature data for the NS operator
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "dx", restrictx, CEED_TRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", restrictxi, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "qdata", restrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the operator that sets the ICs
  CeedOperatorCreate(ceed, qf_ics, NULL, NULL, &op_ics);
  CeedOperatorSetField(op_ics, "x", restrictx, CEED_TRANSPOSE,
                       basisxc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ics, "q0", restrictq, CEED_TRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ics, "coords", restrictxcoord, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, xceed);

  CeedElemRestrictionCreateVector(restrictq, &user->qceed, NULL);
  CeedElemRestrictionCreateVector(restrictq, &user->qdotceed, NULL);
  CeedElemRestrictionCreateVector(restrictq, &user->gceed, NULL);

  { // Create the RHS physics operator
    CeedOperator op;
    CeedOperatorCreate(ceed, qf_rhs, NULL, NULL, &op);
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
    user->op_rhs = op;
  }

  { // Create the IFunction operator
    CeedOperator op;
    CeedOperatorCreate(ceed, qf_ifunction, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", restrictq, CEED_TRANSPOSE,
                         basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", restrictq, CEED_TRANSPOSE,
                         basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdot", restrictq, CEED_TRANSPOSE,
                         basisq, user->qdotceed);
    CeedOperatorSetField(op, "qdata", restrictqdi, CEED_NOTRANSPOSE,
                         CEED_BASIS_COLLOCATED, qdata);
    CeedOperatorSetField(op, "v", restrictq, CEED_TRANSPOSE,
                         basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", restrictq, CEED_TRANSPOSE,
                         basisq, CEED_VECTOR_ACTIVE);
    user->op_ifunction = op;
  }

  CeedQFunctionSetContext(qf_ics, &ctxSetup, sizeof ctxSetup);
  CeedScalar ctxNS[6] = {lambda, mu, k, cv, cp, g};
  struct Advection2dContext_ ctxAdvection2d = {
    .CtauS = CtauS,
    .strong_form = strong_form,
    .stabilization = stab,
  };
  switch (problemChoice) {
  case NS_DENSITY_CURRENT:
    CeedQFunctionSetContext(qf_rhs, &ctxNS, sizeof ctxNS);
    CeedQFunctionSetContext(qf_ifunction, &ctxNS, sizeof ctxNS);
    break;
  case NS_ADVECTION:
  case NS_ADVECTION2D:
    CeedQFunctionSetContext(qf_rhs, &ctxAdvection2d, sizeof ctxAdvection2d);
    CeedQFunctionSetContext(qf_ifunction, &ctxAdvection2d, sizeof ctxAdvection2d);
  }

  // Set up PETSc context
  // Set up units structure
  units->meter = meter;
  units->kilogram = kilogram;
  units->second = second;
  units->Kelvin = Kelvin;
  units->Pascal = Pascal;
  units->JperkgK = JperkgK;
  units->mpersquareds = mpersquareds;
  units->WpermK = WpermK;
  units->kgpercubicm = kgpercubicm;
  units->kgpersquaredms = kgpersquaredms;
  units->Joulepercubicm = Joulepercubicm;

  // Set up user structure
  user->comm = comm;
  user->outputfreq = outputfreq;
  user->contsteps = contsteps;
  user->units = units;
  user->dm = dm;
  user->ceed = ceed;

  // Calculate qdata and ICs
  // Set up state global and local vectors
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);

  ierr = VectorPlacePetscVec(q0ceed, Qloc);CHKERRQ(ierr);

  // Apply Setup Ceed Operators
  ierr = VectorPlacePetscVec(xcorners, Xloc);CHKERRQ(ierr);
  CeedOperatorApply(op_setup, xcorners, qdata, CEED_REQUEST_IMMEDIATE);
  ierr = ComputeLumpedMassMatrix(ceed, dm, restrictq, basisq, restrictqdi, qdata, user->M);CHKERRQ(ierr);

  CeedOperatorApply(op_ics, xcorners, q0ceed, CEED_REQUEST_IMMEDIATE);
  ierr = DMLocalToGlobal(dm, Qloc, ADD_VALUES, Q);CHKERRQ(ierr);
  CeedVectorDestroy(&q0ceed);
  CeedVectorDestroy(&xceed);
  // Fix multiplicity for output of ICs
  {
    CeedVector multlvec;
    Vec Multiplicity, MultiplicityLoc;
    ierr = DMGetLocalVector(dm, &MultiplicityLoc);CHKERRQ(ierr);
    CeedElemRestrictionCreateVector(restrictq, &multlvec, NULL);
    ierr = VectorPlacePetscVec(multlvec, MultiplicityLoc);CHKERRQ(ierr);
    CeedElemRestrictionGetMultiplicity(restrictq, CEED_TRANSPOSE, multlvec);
    CeedVectorDestroy(&multlvec);
    ierr = DMGetGlobalVector(dm, &Multiplicity);CHKERRQ(ierr);
    ierr = VecZeroEntries(Multiplicity);CHKERRQ(ierr);
    ierr = DMLocalToGlobal(dm, MultiplicityLoc, ADD_VALUES, Multiplicity);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(Q, Q, Multiplicity);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(Qloc, Qloc, MultiplicityLoc);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &MultiplicityLoc);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &Multiplicity);CHKERRQ(ierr);
  }
  { // Record boundary values from initial condition and override DMPlexInsertBoundaryValues()
    Vec Qbc;
    ierr = DMGetNamedLocalVector(dm, "Qbc", &Qbc);CHKERRQ(ierr);
    ierr = VecCopy(Qloc, Qbc);CHKERRQ(ierr);
    ierr = VecZeroEntries(Qloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(dm, Q, INSERT_VALUES, Qloc);CHKERRQ(ierr);
    ierr = VecAXPY(Qbc, -1., Qloc);CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(dm, "Qbc", &Qbc);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)dm,"DMPlexInsertBoundaryValues_C",DMPlexInsertBoundaryValues_NS);CHKERRQ(ierr);
  }

  MPI_Comm_rank(comm, &rank);
  if (!rank) {ierr = PetscMkdir(user->outputfolder);CHKERRQ(ierr);}
  // Gather initial Q values
  // In case of continuation of simulation, set up initial values from binary file
  if (contsteps) { // continue from existent solution
    PetscViewer viewer;
    char filepath[PETSC_MAX_PATH_LEN];
    // Read input
    ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/ns-solution.bin",
                         user->outputfolder);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, filepath, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = VecLoad(Q, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  } else {
    //ierr = DMLocalToGlobal(dm, Qloc, INSERT_VALUES, Q);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &Qloc);CHKERRQ(ierr);

  // Create and setup TS
  ierr = TSCreate(comm, &ts); CHKERRQ(ierr);
  if (implicit) {
    ierr = TSSetType(ts, TSBDF); CHKERRQ(ierr);
    ierr = TSSetIFunction(ts, NULL, IFunction_NS, &user);CHKERRQ(ierr);
  } else {
    ierr = TSSetType(ts, TSRK); CHKERRQ(ierr);
    ierr = TSRKSetType(ts, TSRK5F); CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(ts, 500. * units->second); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 1.e-5 * units->second); CHKERRQ(ierr);
  ierr = TSGetAdapt(ts, &adapt); CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,
                              1.e-12 * units->second,
                              1.e-2 * units->second); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  if (!contsteps) { // print initial condition
    ierr = TSMonitor_NS(ts, 0, 0., Q, user); CHKERRQ(ierr);
  } else { // continue from time of last output
    PetscReal time;
    PetscInt count;
    PetscViewer viewer;
    char filepath[PETSC_MAX_PATH_LEN];
    ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/ns-time.bin",
                         user->outputfolder); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, filepath, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer, &time, 1, &count, PETSC_REAL);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = TSSetTime(ts, time * user->units->second); CHKERRQ(ierr);
  }
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
  CeedVectorDestroy(&qdata);
  CeedVectorDestroy(&user->qceed);
  CeedVectorDestroy(&user->qdotceed);
  CeedVectorDestroy(&user->gceed);
  CeedVectorDestroy(&xcorners);
  CeedBasisDestroy(&basisq);
  CeedBasisDestroy(&basisx);
  CeedBasisDestroy(&basisxc);
  CeedElemRestrictionDestroy(&restrictq);
  CeedElemRestrictionDestroy(&restrictx);
  CeedElemRestrictionDestroy(&restrictqdi);
  CeedElemRestrictionDestroy(&restrictxi);
  CeedElemRestrictionDestroy(&restrictxcoord);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_ics);
  CeedQFunctionDestroy(&qf_rhs);
  CeedQFunctionDestroy(&qf_ifunction);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_ics);
  CeedOperatorDestroy(&user->op_rhs);
  CeedOperatorDestroy(&user->op_ifunction);
  CeedDestroy(&ceed);

  // Clean up PETSc
  ierr = VecDestroy(&Q); CHKERRQ(ierr);
  ierr = VecDestroy(&user->M); CHKERRQ(ierr);
  ierr = TSDestroy(&ts); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscFree(units); CHKERRQ(ierr);
  ierr = PetscFree(user); CHKERRQ(ierr);
  return PetscFinalize();
}
