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
//     make [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>] navierstokes
//
// Sample runs:
//
//     ./navierstokes -ceed /cpu/self -problem density_current -petscspace_degree 1
//     ./navierstokes -ceed /gpu/occa -problem advection -petscspace_degree 1
//
//TESTARGS -ceed {ceed_resource} -test -petscspace_degree 1

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
                       PetscScalar[], void *);
  const char *setup_loc, *ics_loc, *apply_rhs_loc, *apply_ifunction_loc;
  const bool non_zero_time;
} problemData;

problemData problemOptions[] = {
  [NS_DENSITY_CURRENT] = {
    .dim = 3,
    .qdatasize = 10,
    .setup = Setup,
    .setup_loc = Setup_loc,
    .ics = ICsDC,
    .apply_rhs = DC,
    .ics_loc = ICsDC_loc,
    .apply_rhs_loc = DC_loc,
    .apply_ifunction = IFunction_DC,
    .apply_ifunction_loc = IFunction_DC_loc,
    .bc = Exact_DC,
    .non_zero_time = false,
  },
  [NS_ADVECTION] = {
    .dim = 3,
    .qdatasize = 10,
    .setup = Setup,
    .setup_loc = Setup_loc,
    .ics = ICsAdvection,
    .apply_rhs = Advection,
    .ics_loc = ICsAdvection_loc,
    .apply_rhs_loc = Advection_loc,
    .apply_ifunction = IFunction_Advection,
    .apply_ifunction_loc = IFunction_Advection_loc,
    .bc = Exact_Advection,
    .non_zero_time = true,
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
    .bc = Exact_Advection2d,
    .non_zero_time = true,
  },
};

typedef struct SimpleBC_ *SimpleBC;
struct SimpleBC_ {
  PetscInt nwall, nslip[3];
  PetscInt walls[10], slips[3][10];
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
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm,&section); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &nfields); CHKERRQ(ierr);
  PetscInt ncomp[nfields], fieldoff[nfields+1];
  fieldoff[0] = 0;
  for (PetscInt f=0; f<nfields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &ncomp[f]); CHKERRQ(ierr);
    fieldoff[f+1] = fieldoff[f] + ncomp[f];
  }

  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd); CHKERRQ(ierr);
  Nelem = cEnd - cStart;
  ierr = PetscMalloc1(Nelem*PetscPowInt(P, dim), &erestrict); CHKERRQ(ierr);
  for (c=cStart,eoffset=0; c<cEnd; c++) {
    PetscInt numindices, *indices, nnodes;
    ierr = DMPlexGetClosureIndices(dm,section,section,c,&numindices,&indices,NULL);
    CHKERRQ(ierr);
    if (numindices % fieldoff[nfields]) SETERRQ1(PETSC_COMM_SELF,
          PETSC_ERR_ARG_INCOMP,"Number of closure indices not compatible with Cell %D",c);
    nnodes = numindices / fieldoff[nfields];
    for (PetscInt i=0; i<nnodes; i++) {
      // Check that indices are blocked by node and thus can be coalesced as a single field with
      // fieldoff[nfields] = sum(ncomp) components.
      for (PetscInt f=0; f<nfields; f++) {
        for (PetscInt j=0; j<ncomp[f]; j++) {
          if (Involute(indices[fieldoff[f]*nnodes + i*ncomp[f] + j])
              != Involute(indices[i*ncomp[0]]) + fieldoff[f] + j)
            SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,
                     "Cell %D closure indices not interlaced for node %D field %D component %D",c,i,
                     f,j);
        }
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = Involute(indices[i*ncomp[0]]);
      erestrict[eoffset++] = loc / fieldoff[nfields];
    }
    ierr = DMPlexRestoreClosureIndices(dm,section,section,c,&numindices,&indices,
                                       NULL); CHKERRQ(ierr);
  }
  if (eoffset != Nelem*PetscPowInt(P, dim)) SETERRQ3(PETSC_COMM_SELF,
        PETSC_ERR_LIB,"ElemRestriction of size (%D,%D) initialized %D nodes",Nelem,
        PetscPowInt(P, dim),eoffset);
  ierr = DMGetLocalVector(dm, &Uloc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Uloc, &Ndof); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Uloc); CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, CEED_INTERLACED, Nelem, PetscPowInt(P, dim),
                            Ndof/fieldoff[nfields], fieldoff[nfields],
                            CEED_MEM_HOST, CEED_COPY_VALUES, erestrict, Erestrict);
  ierr = PetscFree(erestrict); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static int CreateVectorFromPetscVec(Ceed ceed, Vec p, CeedVector *v) {
  PetscErrorCode ierr;
  PetscInt m;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(p, &m); CHKERRQ(ierr);
  ierr = CeedVectorCreate(ceed, m, v); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static int VectorPlacePetscVec(CeedVector c, Vec p) {
  PetscErrorCode ierr;
  PetscInt mceed,mpetsc;
  PetscScalar *a;

  PetscFunctionBeginUser;
  ierr = CeedVectorGetLength(c, &mceed); CHKERRQ(ierr);
  ierr = VecGetLocalSize(p, &mpetsc); CHKERRQ(ierr);
  if (mceed != mpetsc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,
                                  "Cannot place PETSc Vec of length %D in CeedVector of length %D",mpetsc,mceed);
  ierr = VecGetArray(p, &a); CHKERRQ(ierr);
  CeedVectorSetArray(c, CEED_MEM_HOST, CEED_USE_POINTER, a);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm,
    PetscBool insertEssential, Vec Qloc, PetscReal time, Vec faceGeomFVM,
    Vec cellGeomFVM, Vec gradFVM) {
  PetscErrorCode ierr;
  Vec Qbc;

  PetscFunctionBegin;
  ierr = DMGetNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
  ierr = VecAXPY(Qloc, 1., Qbc); CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
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
  ierr = DMGetLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Gloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Qloc, 0.0,
                                    NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = VecZeroEntries(Gloc); CHKERRQ(ierr);

  // Ceed Vectors
  ierr = VecGetArrayRead(Qloc, (const PetscScalar **)&q); CHKERRQ(ierr);
  ierr = VecGetArray(Gloc, &g); CHKERRQ(ierr);
  CeedVectorSetArray(user->qceed, CEED_MEM_HOST, CEED_USE_POINTER, q);
  CeedVectorSetArray(user->gceed, CEED_MEM_HOST, CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_rhs, user->qceed, user->gceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  ierr = VecRestoreArrayRead(Qloc, (const PetscScalar **)&q); CHKERRQ(ierr);
  ierr = VecRestoreArray(Gloc, &g); CHKERRQ(ierr);

  ierr = VecZeroEntries(G); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Gloc, ADD_VALUES, G); CHKERRQ(ierr);

  // Inverse of the lumped mass matrix
  ierr = VecPointwiseMult(G, G, user->M); // M is Minv
  CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Gloc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Qdot, Vec G,
                                   void *userData) {
  PetscErrorCode ierr;
  User user = *(User *)userData;
  const PetscScalar *q, *qdot;
  PetscScalar *g;
  Vec Qloc, Qdotloc, Gloc;

  // Global-to-local
  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Qdotloc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Gloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Qloc, 0.0,
                                    NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qdotloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Qdot, INSERT_VALUES, Qdotloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Gloc); CHKERRQ(ierr);

  // Ceed Vectors
  ierr = VecGetArrayRead(Qloc, &q); CHKERRQ(ierr);
  ierr = VecGetArrayRead(Qdotloc, &qdot); CHKERRQ(ierr);
  ierr = VecGetArray(Gloc, &g); CHKERRQ(ierr);
  CeedVectorSetArray(user->qceed, CEED_MEM_HOST, CEED_USE_POINTER,
                     (PetscScalar *)q);
  CeedVectorSetArray(user->qdotceed, CEED_MEM_HOST, CEED_USE_POINTER,
                     (PetscScalar *)qdot);
  CeedVectorSetArray(user->gceed, CEED_MEM_HOST, CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_ifunction, user->qceed, user->gceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  ierr = VecRestoreArrayRead(Qloc, &q); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Qdotloc, &qdot); CHKERRQ(ierr);
  ierr = VecRestoreArray(Gloc, &g); CHKERRQ(ierr);

  ierr = VecZeroEntries(G); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Gloc, ADD_VALUES, G); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Qdotloc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Gloc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// User provided TS Monitor
static PetscErrorCode TSMonitor_NS(TS ts, PetscInt stepno, PetscReal time,
                                   Vec Q, void *ctx) {
  User user = ctx;
  Vec Qloc;
  char filepath[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  PetscErrorCode ierr;

  // Set up output
  PetscFunctionBeginUser;
  // Print every 'outputfreq' steps
  if (stepno % user->outputfreq != 0)
    PetscFunctionReturn(0);
  ierr = DMGetLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Qloc, "StateVec"); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);

  // Output
  ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/ns-%03D.vtu",
                       user->outputfolder, stepno + user->contsteps);
  CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q), filepath,
                            FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  ierr = VecView(Qloc, viewer); CHKERRQ(ierr);
  if (user->dmviz) {
    Vec Qrefined, Qrefined_loc;
    char filepath_refined[PETSC_MAX_PATH_LEN];
    PetscViewer viewer_refined;

    ierr = DMGetGlobalVector(user->dmviz, &Qrefined); CHKERRQ(ierr);
    ierr = DMGetLocalVector(user->dmviz, &Qrefined_loc); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)Qrefined_loc, "Refined"); CHKERRQ(ierr);
    ierr = MatInterpolate(user->interpviz, Q, Qrefined); CHKERRQ(ierr);
    ierr = VecZeroEntries(Qrefined_loc); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(user->dmviz, Qrefined, INSERT_VALUES, Qrefined_loc);
    CHKERRQ(ierr);
    ierr = PetscSNPrintf(filepath_refined, sizeof filepath_refined,
                         "%s/nsrefined-%03D.vtu",
                         user->outputfolder, stepno + user->contsteps);
    CHKERRQ(ierr);
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)Qrefined),
                              filepath_refined,
                              FILE_MODE_WRITE, &viewer_refined); CHKERRQ(ierr);
    ierr = VecView(Qrefined_loc, viewer_refined); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(user->dmviz, &Qrefined_loc); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(user->dmviz, &Qrefined); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer_refined); CHKERRQ(ierr);
  }
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

static PetscErrorCode ICs_PetscMultiplicity(CeedOperator op_ics,
    CeedVector xcorners, CeedVector q0ceed, DM dm, Vec Qloc, Vec Q,
    CeedElemRestriction restrictq, SetupContext ctxSetup, CeedScalar time) {
  PetscErrorCode ierr;
  CeedVector multlvec;
  Vec Multiplicity, MultiplicityLoc;

  ctxSetup->time = time;
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = VectorPlacePetscVec(q0ceed, Qloc); CHKERRQ(ierr);
  CeedOperatorApply(op_ics, xcorners, q0ceed, CEED_REQUEST_IMMEDIATE);
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, Qloc, ADD_VALUES, Q); CHKERRQ(ierr);

  // Fix multiplicity for output of ICs
  ierr = DMGetLocalVector(dm, &MultiplicityLoc); CHKERRQ(ierr);
  CeedElemRestrictionCreateVector(restrictq, &multlvec, NULL);
  ierr = VectorPlacePetscVec(multlvec, MultiplicityLoc); CHKERRQ(ierr);
  CeedElemRestrictionGetMultiplicity(restrictq, multlvec);
  CeedVectorDestroy(&multlvec);
  ierr = DMGetGlobalVector(dm, &Multiplicity); CHKERRQ(ierr);
  ierr = VecZeroEntries(Multiplicity); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, MultiplicityLoc, ADD_VALUES, Multiplicity);
  CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Q, Q, Multiplicity); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Qloc, Qloc, MultiplicityLoc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &MultiplicityLoc); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &Multiplicity); CHKERRQ(ierr);

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
  CeedVector mceed;
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
  CeedOperatorSetField(op_mass, "q", restrictq,
                       basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", restrictqdi,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_mass, "v", restrictq,
                       basisq, CEED_VECTOR_ACTIVE);

  ierr = DMGetLocalVector(dm, &Mloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Mloc); CHKERRQ(ierr);
  CeedElemRestrictionCreateVector(restrictq, &mceed, NULL);
  ierr = VectorPlacePetscVec(mceed, Mloc); CHKERRQ(ierr);

  {
    // Compute a lumped mass matrix
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

PetscErrorCode SetUpDM(DM dm, problemData *problem, const char *prefix,
                       SimpleBC bc, void *ctxSetup, PetscInt *degree) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  {
    // Configure the finite element space and boundary conditions
    PetscFE fe;
    PetscSpace fespace;
    PetscInt ncompq = 5;
    ierr = PetscFECreateDefault(PETSC_COMM_SELF,problem->dim, ncompq, PETSC_FALSE,
                                prefix, PETSC_DETERMINE, &fe); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe, "Q"); CHKERRQ(ierr);
    ierr = DMAddField(dm,NULL,(PetscObject)fe); CHKERRQ(ierr);
    ierr = DMCreateDS(dm); CHKERRQ(ierr);
    /* Wall boundary conditions are zero velocity and zero flux for density and energy */
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "Face Sets", 0,
                         3,(PetscInt[]) {1,2,3},
                         (void(*)(void))problem->bc, bc->nwall, bc->walls, ctxSetup);
    CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", "Face Sets", 0,
                         1, (PetscInt[]) {1},
                         (void(*)(void))NULL, bc->nslip[0], bc->slips[0], ctxSetup);
    CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", "Face Sets", 0,
                         1, (PetscInt[]) {2},
                         (void(*)(void))NULL, bc->nslip[1], bc->slips[1], ctxSetup);
    CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", "Face Sets", 0,
                         1, (PetscInt[]) {3},
                         (void(*)(void))NULL, bc->nslip[2], bc->slips[2], ctxSetup);
    CHKERRQ(ierr);
    ierr = DMPlexSetClosurePermutationTensor(dm,PETSC_DETERMINE,NULL);
    CHKERRQ(ierr);
    ierr = PetscFEGetBasisSpace(fe, &fespace); CHKERRQ(ierr);
    if (degree) {
      ierr = PetscSpaceGetDegree(fespace, degree, NULL); CHKERRQ(ierr);
      if (*degree < 1) SETERRQ1(PetscObjectComm((PetscObject)dm),
                                  PETSC_ERR_ARG_OUTOFRANGE,
                                  "Degree %D; must specify -petscspace_degree 1 (or greater)", *degree);
    }
    ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);
  }
  {
    // Empty name for conserved field (because there is only one field)
    PetscSection section;
    ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, 0, ""); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 0, "Density"); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 1, "MomentumX"); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 2, "MomentumY"); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 3, "MomentumZ"); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 4, "EnergyDensity");
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  DM dm, dmcoord, dmviz;
  Mat interpviz;
  TS ts;
  TSAdapt adapt;
  User user;
  Units units;
  char ceedresource[4096] = "/cpu/self";
  PetscInt cStart, cEnd, localNelem, lnodes, steps;
  const PetscInt ncompq = 5;
  PetscMPIInt rank;
  PetscScalar ftime;
  Vec Q, Qloc, Xloc;
  Ceed ceed;
  CeedInt numP, numQ;
  CeedVector xcorners, qdata, q0ceed;
  CeedBasis basisx, basisxc, basisq;
  CeedElemRestriction restrictx, restrictxcoord, restrictq, restrictqdi;
  CeedQFunction qf_setup, qf_ics, qf_rhs, qf_ifunction;
  CeedOperator op_setup, op_ics;
  CeedScalar Rd;
  PetscScalar WpermK, Pascal, JperkgK, mpersquareds, kgpercubicm,
              kgpersquaredms, Joulepercubicm;
  problemType problemChoice;
  problemData *problem = NULL;
  StabilizationType stab;
  PetscBool   test, implicit, viz_refine;
  struct SimpleBC_ bc = {
    .nwall = 6,
    .walls = {1,2,3,4,5,6},
  };
  double start, cpu_time_used;

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
  PetscReal center[3], dc_axis[3] = {0, 0, 0};

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  // Allocate PETSc context
  ierr = PetscCalloc1(1, &user); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &units); CHKERRQ(ierr);

  // Parse command line options
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "Navier-Stokes in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test", "Run in test mode",
                          NULL, test=PETSC_FALSE, &test, NULL); CHKERRQ(ierr);
  problemChoice = NS_DENSITY_CURRENT;
  ierr = PetscOptionsEnum("-problem", "Problem to solve", NULL,
                          problemTypes, (PetscEnum)problemChoice,
                          (PetscEnum *)&problemChoice, NULL); CHKERRQ(ierr);
  problem = &problemOptions[problemChoice];
  ierr = PetscOptionsEnum("-stab", "Stabilization method", NULL,
                          StabilizationTypes, (PetscEnum)(stab = STAB_NONE),
                          (PetscEnum *)&stab, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation",
                          NULL, implicit=PETSC_FALSE, &implicit, NULL); CHKERRQ(ierr);
  {
    PetscInt len;
    PetscBool flg;
    ierr = PetscOptionsIntArray("-bc_wall",
                                "Use wall boundary conditions on this list of faces", NULL, bc.walls,
                                (len = sizeof(bc.walls) / sizeof(bc.walls[0]), &len), &flg); CHKERRQ(ierr);
    if (flg) bc.nwall = len;
    for (PetscInt j=0; j<3; j++) {
      const char *flags[3] = {"-bc_slip_x", "-bc_slip_y", "-bc_slip_z"};
      ierr = PetscOptionsIntArray(flags[j],
                                  "Use slip boundary conditions on this list of faces", NULL, bc.slips[j],
                                  (len = sizeof(bc.slips[j]) / sizeof(bc.slips[j][0]), &len), &flg);
      CHKERRQ(ierr);
      if (flg) bc.nslip[j] = len;
    }
  }
  ierr = PetscOptionsBool("-viz_refine",
                          "Use regular refinement for visualization",
                          NULL, viz_refine=PETSC_FALSE, &viz_refine, NULL); CHKERRQ(ierr);
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
  ierr = PetscOptionsScalar("-CtauS",
                            "Scale coefficient for tau (nondimensional)",
                            NULL, CtauS, &CtauS, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-strong_form",
                            "Strong (1) or weak/integrated by parts (0) advection residual",
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
                               &n, NULL); CHKERRQ(ierr);
  n = problem->dim;
  center[0] = 0.5 * lx;
  center[1] = 0.5 * ly;
  center[2] = 0.5 * lz;
  ierr = PetscOptionsRealArray("-center", "Location of bubble center",
                               NULL, center, &n, NULL); CHKERRQ(ierr);
  n = problem->dim;
  ierr = PetscOptionsRealArray("-dc_axis",
                               "Axis of density current cylindrical anomaly, or {0,0,0} for spherically symmetric",
                               NULL, dc_axis, &n, NULL); CHKERRQ(ierr);
  {
    PetscReal norm = PetscSqrtReal(PetscSqr(dc_axis[0]) + PetscSqr(
                                     dc_axis[1]) + PetscSqr(dc_axis[2]));
    if (norm > 0) {
      for (int i=0; i<3; i++) dc_axis[i] /= norm;
    }
  }
  ierr = PetscOptionsInt("-output_freq",
                         "Frequency of output, in number of steps",
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
  for (int i=0; i<3; i++) center[i] *= meter;

  const CeedInt dim = problem->dim, ncompx = problem->dim,
                qdatasize = problem->qdatasize;
  // Set up the libCEED context
  struct SetupContext_ ctxSetup =  {
    .theta0 = theta0,
    .thetaC = thetaC,
    .P0 = P0,
    .N = N,
    .cv = cv,
    .cp = cp,
    .Rd = Rd,
    .g = g,
    .rc = rc,
    .lx = lx,
    .ly = ly,
    .lz = lz,
    .periodicity0 = periodicity[0],
    .periodicity1 = periodicity[1],
    .periodicity2 = periodicity[2],
    .center[0] = center[0],
    .center[1] = center[1],
    .center[2] = center[2],
    .dc_axis[0] = dc_axis[0],
    .dc_axis[1] = dc_axis[1],
    .dc_axis[2] = dc_axis[2],
    .time = 0,
  };

  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, NULL, NULL, (PetscReal[]) {lx, ly, lz},
  periodicity, PETSC_TRUE, &dm); CHKERRQ(ierr);
  if (1) {
    DM               dmDist = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(dm, &part); CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, 0, NULL, &dmDist); CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(&dm); CHKERRQ(ierr);
      dm  = dmDist;
    }
  }
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

  ierr = DMLocalizeCoordinates(dm); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = SetUpDM(dm, problem, NULL, &bc, &ctxSetup, &degree); CHKERRQ(ierr);
  if (!test) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Degree of FEM Space: %D\n",
                       (PetscInt)degree); CHKERRQ(ierr);
  }
  dmviz = NULL;
  interpviz = NULL;
  if (viz_refine) {
    ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE); CHKERRQ(ierr);
    ierr = DMRefine(dm, MPI_COMM_NULL, &dmviz); CHKERRQ(ierr);
    ierr = DMSetCoarseDM(dmviz, dm); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL,"-viz_petscspace_degree","1"); CHKERRQ(ierr);
    ierr = SetUpDM(dmviz, problem, "viz_", &bc, &ctxSetup, NULL); CHKERRQ(ierr);
    ierr = DMCreateInterpolation(dm, dmviz, &interpviz, NULL); CHKERRQ(ierr);
  }
  ierr = DMCreateGlobalVector(dm, &Q); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &Qloc); CHKERRQ(ierr);
  ierr = VecGetSize(Qloc, &lnodes); CHKERRQ(ierr);
  lnodes /= ncompq;

  {
    // Print grid information
    CeedInt gdofs, odofs;
    int comm_size;
    char box_faces_str[PETSC_MAX_PATH_LEN] = "NONE";
    ierr = VecGetSize(Q, &gdofs); CHKERRQ(ierr);
    ierr = VecGetLocalSize(Q, &odofs); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &comm_size); CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL, NULL, "-dm_plex_box_faces", box_faces_str,
                                 sizeof(box_faces_str), NULL); CHKERRQ(ierr);
    if (!test) {
      ierr = PetscPrintf(comm, "Global FEM dofs: %D (%D owned) on %d ranks\n", gdofs,
                         odofs, comm_size); CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "Local FEM nodes: %D\n", lnodes); CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "dm_plex_box_faces: %s\n", box_faces_str);
      CHKERRQ(ierr);
    }

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

  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord,PETSC_DETERMINE,NULL);
  CHKERRQ(ierr);

  // CEED Restrictions
  ierr = CreateRestrictionFromPlex(ceed, dm, degree+1, &restrictq); CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dmcoord, 2, &restrictx); CHKERRQ(ierr);
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  localNelem = cEnd - cStart;
  CeedInt numQdim = CeedIntPow(numQ, dim);
  CeedElemRestrictionCreateStrided(ceed, localNelem, numQdim,
                                   localNelem*numQdim, qdatasize,
                                   CEED_STRIDES_BACKEND, &restrictqdi);
  CeedElemRestrictionCreateStrided(ceed, localNelem, PetscPowInt(numP, dim),
                                   localNelem*PetscPowInt(numP, dim), ncompx,
                                   CEED_STRIDES_BACKEND, &restrictxcoord);

  ierr = DMGetCoordinatesLocal(dm, &Xloc); CHKERRQ(ierr);
  ierr = CreateVectorFromPetscVec(ceed, Xloc, &xcorners); CHKERRQ(ierr);

  // Create the CEED vectors that will be needed in setup
  CeedInt Nqpts;
  CeedBasisGetNumQuadraturePoints(basisq, &Nqpts);
  CeedInt Ndofs = 1;
  for (int d=0; d<3; d++) Ndofs *= numP;
  CeedVectorCreate(ceed, qdatasize*localNelem*Nqpts, &qdata);
  CeedElemRestrictionCreateVector(restrictq, &q0ceed, NULL);

  // Create the Q-function that builds the quadrature data for the NS operator
  CeedQFunctionCreateInterior(ceed, 1, problem->setup, problem->setup_loc,
                              &qf_setup);
  CeedQFunctionAddInput(qf_setup, "dx", ncompx*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", qdatasize, CEED_EVAL_NONE);

  // Create the Q-function that sets the ICs of the operator
  CeedQFunctionCreateInterior(ceed, 1, problem->ics, problem->ics_loc, &qf_ics);
  CeedQFunctionAddInput(qf_ics, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ics, "q0", ncompq, CEED_EVAL_NONE);

  qf_rhs = NULL;
  if (problem->apply_rhs) { // Create the Q-function that defines the action of the RHS operator
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_rhs,
                                problem->apply_rhs_loc, &qf_rhs);
    CeedQFunctionAddInput(qf_rhs, "q", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_rhs, "dq", ncompq*dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_rhs, "qdata", qdatasize, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_rhs, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_rhs, "v", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_rhs, "dv", ncompq*dim, CEED_EVAL_GRAD);
  }

  qf_ifunction = NULL;
  if (problem->apply_ifunction) { // Create the Q-function that defines the action of the IFunction
    CeedQFunctionCreateInterior(ceed, 1, problem->apply_ifunction,
                                problem->apply_ifunction_loc, &qf_ifunction);
    CeedQFunctionAddInput(qf_ifunction, "q", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_ifunction, "dq", ncompq*dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_ifunction, "qdot", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_ifunction, "qdata", qdatasize, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_ifunction, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_ifunction, "v", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_ifunction, "dv", ncompq*dim, CEED_EVAL_GRAD);
  }

  // Create the operator that builds the quadrature data for the NS operator
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "dx", restrictx,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "qdata", restrictqdi,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the operator that sets the ICs
  CeedOperatorCreate(ceed, qf_ics, NULL, NULL, &op_ics);
  CeedOperatorSetField(op_ics, "x", restrictx,
                       basisxc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ics, "q0", restrictq,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedElemRestrictionCreateVector(restrictq, &user->qceed, NULL);
  CeedElemRestrictionCreateVector(restrictq, &user->qdotceed, NULL);
  CeedElemRestrictionCreateVector(restrictq, &user->gceed, NULL);

  if (qf_rhs) { // Create the RHS physics operator
    CeedOperator op;
    CeedOperatorCreate(ceed, qf_rhs, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", restrictq,
                         basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", restrictq,
                         basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdata", restrictqdi,
                         CEED_BASIS_COLLOCATED, qdata);
    CeedOperatorSetField(op, "x", restrictx,
                         basisx, xcorners);
    CeedOperatorSetField(op, "v", restrictq,
                         basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", restrictq,
                         basisq, CEED_VECTOR_ACTIVE);
    user->op_rhs = op;
  }

  if (qf_ifunction) { // Create the IFunction operator
    CeedOperator op;
    CeedOperatorCreate(ceed, qf_ifunction, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", restrictq,
                         basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", restrictq,
                         basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdot", restrictq,
                         basisq, user->qdotceed);
    CeedOperatorSetField(op, "qdata", restrictqdi,
                         CEED_BASIS_COLLOCATED, qdata);
    CeedOperatorSetField(op, "x", restrictx,
                         basisx, xcorners);
    CeedOperatorSetField(op, "v", restrictq,
                         basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", restrictq,
                         basisq, CEED_VECTOR_ACTIVE);
    user->op_ifunction = op;
  }

  CeedQFunctionSetContext(qf_ics, &ctxSetup, sizeof ctxSetup);
  CeedScalar ctxNS[8] = {lambda, mu, k, cv, cp, g, Rd};
  struct Advection2dContext_ ctxAdvection2d = {
    .CtauS = CtauS,
    .strong_form = strong_form,
    .stabilization = stab,
  };
  switch (problemChoice) {
  case NS_DENSITY_CURRENT:
    if (qf_rhs) CeedQFunctionSetContext(qf_rhs, &ctxNS, sizeof ctxNS);
    if (qf_ifunction) CeedQFunctionSetContext(qf_ifunction, &ctxNS, sizeof ctxNS);
    break;
  case NS_ADVECTION:
  case NS_ADVECTION2D:
    if (qf_rhs) CeedQFunctionSetContext(qf_rhs, &ctxAdvection2d,
                                          sizeof ctxAdvection2d);
    if (qf_ifunction) CeedQFunctionSetContext(qf_ifunction, &ctxAdvection2d,
          sizeof ctxAdvection2d);
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
  user->dmviz = dmviz;
  user->interpviz = interpviz;
  user->ceed = ceed;

  // Calculate qdata and ICs
  // Set up state global and local vectors
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);

  ierr = VectorPlacePetscVec(q0ceed, Qloc); CHKERRQ(ierr);

  // Apply Setup Ceed Operators
  ierr = VectorPlacePetscVec(xcorners, Xloc); CHKERRQ(ierr);
  CeedOperatorApply(op_setup, xcorners, qdata, CEED_REQUEST_IMMEDIATE);
  ierr = ComputeLumpedMassMatrix(ceed, dm, restrictq, basisq, restrictqdi, qdata,
                                 user->M); CHKERRQ(ierr);

  ierr = ICs_PetscMultiplicity(op_ics, xcorners, q0ceed, dm, Qloc, Q, restrictq,
                               &ctxSetup, 0.0);
  if (1) { // Record boundary values from initial condition and override DMPlexInsertBoundaryValues()
    // We use this for the main simulation DM because the reference DMPlexInsertBoundaryValues() is very slow.  If we
    // disable this, we should still get the same results due to the problem->bc function, but with potentially much
    // slower execution.
    Vec Qbc;
    ierr = DMGetNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
    ierr = VecCopy(Qloc, Qbc); CHKERRQ(ierr);
    ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);
    ierr = VecAXPY(Qbc, -1., Qloc); CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)dm,
                                      "DMPlexInsertBoundaryValues_C",DMPlexInsertBoundaryValues_NS); CHKERRQ(ierr);
  }

  MPI_Comm_rank(comm, &rank);
  if (!rank) {ierr = PetscMkdir(user->outputfolder); CHKERRQ(ierr);}
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
  ierr = DMRestoreLocalVector(dm, &Qloc); CHKERRQ(ierr);

  // Create and setup TS
  ierr = TSCreate(comm, &ts); CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm); CHKERRQ(ierr);
  if (implicit) {
    ierr = TSSetType(ts, TSBDF); CHKERRQ(ierr);
    if (user->op_ifunction) {
      ierr = TSSetIFunction(ts, NULL, IFunction_NS, &user); CHKERRQ(ierr);
    } else {                    // Implicit integrators can fall back to using an RHSFunction
      ierr = TSSetRHSFunction(ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
    }
  } else {
    if (!user->op_rhs) SETERRQ(comm,PETSC_ERR_ARG_NULL,
                                 "Problem does not provide RHSFunction");
    ierr = TSSetType(ts, TSRK); CHKERRQ(ierr);
    ierr = TSRKSetType(ts, TSRK5F); CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(ts, 500. * units->second); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 1.e-2 * units->second); CHKERRQ(ierr);
  if (test) {ierr = TSSetMaxSteps(ts, 1); CHKERRQ(ierr);}
  ierr = TSGetAdapt(ts, &adapt); CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,
                              1.e-12 * units->second,
                              1.e2 * units->second); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  if (!contsteps) { // print initial condition
    if (!test) {
      ierr = TSMonitor_NS(ts, 0, 0., Q, user); CHKERRQ(ierr);
    }
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
  if (!test) {
    ierr = TSMonitorSet(ts, TSMonitor_NS, user, NULL); CHKERRQ(ierr);
  }

  // Solve
  start = MPI_Wtime();
  ierr = PetscBarrier((PetscObject)ts); CHKERRQ(ierr);
  ierr = TSSolve(ts, Q); CHKERRQ(ierr);
  cpu_time_used = MPI_Wtime() - start;
  ierr = TSGetSolveTime(ts,&ftime); CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE, &cpu_time_used, 1, MPI_DOUBLE, MPI_MIN,
                       comm); CHKERRQ(ierr);
  if (!test) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time taken for solution: %g\n",
                       (double)cpu_time_used); CHKERRQ(ierr);
  }

  // Get error
  if (problem->non_zero_time && !test) {
    Vec Qexact, Qexactloc;
    PetscReal norm;
    ierr = DMCreateGlobalVector(dm, &Qexact); CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm, &Qexactloc); CHKERRQ(ierr);
    ierr = VecGetSize(Qexactloc, &lnodes); CHKERRQ(ierr);

    ierr = ICs_PetscMultiplicity(op_ics, xcorners, q0ceed, dm, Qexactloc, Qexact,
                                 restrictq, &ctxSetup, ftime); CHKERRQ(ierr);

    ierr = VecAXPY(Q, -1.0, Qexact);  CHKERRQ(ierr);
    ierr = VecNorm(Q, NORM_MAX, &norm); CHKERRQ(ierr);
    CeedVectorDestroy(&q0ceed);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Max Error: %g\n",
                       (double)norm); CHKERRQ(ierr);
  }

  // Output Statistics
  ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
  if (!test) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time integrator took %D time steps to reach final time %g\n",
                       steps,(double)ftime); CHKERRQ(ierr);
  }

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
  ierr = MatDestroy(&interpviz); CHKERRQ(ierr);
  ierr = DMDestroy(&dmviz); CHKERRQ(ierr);
  ierr = TSDestroy(&ts); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscFree(units); CHKERRQ(ierr);
  ierr = PetscFree(user); CHKERRQ(ierr);
  return PetscFinalize();
}

