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
//     ./navierstokes -ceed /cpu/self -problem density_current -degree 1
//     ./navierstokes -ceed /gpu/cuda -problem advection -degree 1
//
//TESTARGS -ceed {ceed_resource} -test dc_explicit -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3
//TESTARGS -ceed {ceed_resource} -test dc_implicit_stab_none -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha
//TESTARGS -ceed {ceed_resource} -test dc_implicit_stab_supg -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -stab supg

/// @file
/// Navier-Stokes example using PETSc

const char help[] = "Solve Navier-Stokes using PETSc and libCEED\n";

#include <ceed.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <petscts.h>
#include <stdbool.h>
#include "advection.h"
#include "advection2d.h"
#include "common.h"
#include "densitycurrent.h"
#include "setup-boundary.h"

#if PETSC_VERSION_LT(3,14,0)
#  define DMPlexGetClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexGetClosureIndices(a,b,c,d,f,g,i)
#  define DMPlexRestoreClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexRestoreClosureIndices(a,b,c,d,f,g,i)
#endif

#if PETSC_VERSION_LT(3,14,0)
#  define DMAddBoundary(a,b,c,d,e,f,g,h,i,j,k,l) DMAddBoundary(a,b,c,d,e,f,g,h,j,k,l)
#endif

// MemType Options
static const char *const memTypes[] = {
  "host",
  "device",
  "memType", "CEED_MEM_", NULL
};

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
  "problemType", "NS_", NULL
};

// Wind Options for Advection
typedef enum {
  ADVECTION_WIND_ROTATION = 0,
  ADVECTION_WIND_TRANSLATION = 1,
} WindType;
static const char *const WindTypes[] = {
  "rotation",
  "translation",
  "WindType", "ADVECTION_WIND_", NULL
};

typedef enum {
  STAB_NONE = 0,
  STAB_SU = 1,   // Streamline Upwind
  STAB_SUPG = 2, // Streamline Upwind Petrov-Galerkin
} StabilizationType;
static const char *const StabilizationTypes[] = {
  "none",
  "SU",
  "SUPG",
  "StabilizationType", "STAB_", NULL
};

// Test Options
typedef enum {
  TEST_NONE = 0,                  // Non test mode
  // DENSITY_CURRENT
  TEST_DC_EXPLICIT = 1,           // Explicit test
  TEST_DC_IMPLICIT_STAB_NONE = 2, // Implicit test no stab
  TEST_DC_IMPLICIT_STAB_SUPG = 3, // Implicit test supg stab
} testType;
static const char *const testTypes[] = {
  "none",
  "dc_explicit",
  "dc_implicit_stab_none",
  "dc_implicit_stab_supg",
  "testType", "TEST_", NULL
};

// Tests specific data
typedef struct {
  PetscScalar testtol;
  const char *filepath;
} testData;

testData testOptions[] = {
  [TEST_NONE] = {
    .testtol = 0.,
    .filepath = NULL
  },
  [TEST_DC_EXPLICIT] = {
    .testtol = 1E-5,
    .filepath = "examples/fluids/tests-output/fluids-navierstokes-dc-explicit.bin"
  },
  [TEST_DC_IMPLICIT_STAB_NONE] = {
    .testtol = 5E-4,
    .filepath = "examples/fluids/tests-output/fluids-navierstokes-dc-implicit-stab-none.bin"
  },
  [TEST_DC_IMPLICIT_STAB_SUPG] = {
    .testtol = 5E-4,
    .filepath = "examples/fluids/tests-output/fluids-navierstokes-dc-implicit-stab-supg.bin"
  }
};

// Problem specific data
typedef struct {
  CeedInt dim, qdatasizeVol, qdatasizeSur;
  CeedQFunctionUser setupVol, setupSur, ics, applyVol_rhs, applyVol_ifunction,
                    applySur;
  PetscErrorCode (*bc)(PetscInt, PetscReal, const PetscReal[], PetscInt,
                       PetscScalar[], void *);
  const char *setupVol_loc, *setupSur_loc, *ics_loc, *applyVol_rhs_loc,
        *applyVol_ifunction_loc, *applySur_loc;
  const bool non_zero_time;
} problemData;

problemData problemOptions[] = {
  [NS_DENSITY_CURRENT] = {
    .dim                       = 3,
    .qdatasizeVol              = 10,
    .qdatasizeSur              = 4,
    .setupVol                  = Setup,
    .setupVol_loc              = Setup_loc,
    .setupSur                  = SetupBoundary,
    .setupSur_loc              = SetupBoundary_loc,
    .ics                       = ICsDC,
    .ics_loc                   = ICsDC_loc,
    .applyVol_rhs              = DC,
    .applyVol_rhs_loc          = DC_loc,
    .applyVol_ifunction        = IFunction_DC,
    .applyVol_ifunction_loc    = IFunction_DC_loc,
    .bc                        = Exact_DC,
    .non_zero_time             = PETSC_FALSE,
  },
  [NS_ADVECTION] = {
    .dim                       = 3,
    .qdatasizeVol              = 10,
    .qdatasizeSur              = 4,
    .setupVol                  = Setup,
    .setupVol_loc              = Setup_loc,
    .setupSur                  = SetupBoundary,
    .setupSur_loc              = SetupBoundary_loc,
    .ics                       = ICsAdvection,
    .ics_loc                   = ICsAdvection_loc,
    .applyVol_rhs              = Advection,
    .applyVol_rhs_loc          = Advection_loc,
    .applyVol_ifunction        = IFunction_Advection,
    .applyVol_ifunction_loc    = IFunction_Advection_loc,
    .applySur                  = Advection_Sur,
    .applySur_loc              = Advection_Sur_loc,
    .bc                        = Exact_Advection,
    .non_zero_time             = PETSC_FALSE,
  },
  [NS_ADVECTION2D] = {
    .dim                       = 2,
    .qdatasizeVol              = 5,
    .qdatasizeSur              = 3,
    .setupVol                  = Setup2d,
    .setupVol_loc              = Setup2d_loc,
    .setupSur                  = SetupBoundary2d,
    .setupSur_loc              = SetupBoundary2d_loc,
    .ics                       = ICsAdvection2d,
    .ics_loc                   = ICsAdvection2d_loc,
    .applyVol_rhs              = Advection2d,
    .applyVol_rhs_loc          = Advection2d_loc,
    .applyVol_ifunction        = IFunction_Advection2d,
    .applyVol_ifunction_loc    = IFunction_Advection2d_loc,
    .applySur                  = Advection2d_Sur,
    .applySur_loc              = Advection2d_Sur_loc,
    .bc                        = Exact_Advection2d,
    .non_zero_time             = PETSC_TRUE,
  },
};

// PETSc user data
typedef struct User_ *User;
typedef struct Units_ *Units;

struct User_ {
  MPI_Comm comm;
  PetscInt outputfreq;
  DM dm;
  DM dmviz;
  Mat interpviz;
  Ceed ceed;
  Units units;
  CeedVector qceed, qdotceed, gceed;
  CeedOperator op_rhs_vol, op_rhs, op_ifunction_vol, op_ifunction;
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
  PetscScalar Joule;
};

typedef struct SimpleBC_ *SimpleBC;
struct SimpleBC_ {
  PetscInt nwall, nslip[3];
  PetscInt walls[6], slips[3][6];
  PetscBool userbc;
};

// Essential BC dofs are encoded in closure indices as -(i+1).
static PetscInt Involute(PetscInt i) {
  return i >= 0 ? i : -(i+1);
}

// Utility function to create local CEED restriction
static PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt height, DMLabel domainLabel, CeedInt value,
    CeedElemRestriction *Erestrict) {

  PetscSection section;
  PetscInt p, Nelem, Ndof, *erestrict, eoffset, nfields, dim, depth;
  DMLabel depthLabel;
  IS depthIS, iterIS;
  Vec Uloc;
  const PetscInt *iterIndices;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &nfields); CHKERRQ(ierr);
  PetscInt ncomp[nfields], fieldoff[nfields+1];
  fieldoff[0] = 0;
  for (PetscInt f=0; f<nfields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &ncomp[f]); CHKERRQ(ierr);
    fieldoff[f+1] = fieldoff[f] + ncomp[f];
  }

  ierr = DMPlexGetDepth(dm, &depth); CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel); CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel, depth - height, &depthIS); CHKERRQ(ierr);
  if (domainLabel) {
    IS domainIS;
    ierr = DMLabelGetStratumIS(domainLabel, value, &domainIS); CHKERRQ(ierr);
    if (domainIS) { // domainIS is non-empty
      ierr = ISIntersect(depthIS, domainIS, &iterIS); CHKERRQ(ierr);
      ierr = ISDestroy(&domainIS); CHKERRQ(ierr);
    } else { // domainIS is NULL (empty)
      iterIS = NULL;
    }
    ierr = ISDestroy(&depthIS); CHKERRQ(ierr);
  } else {
    iterIS = depthIS;
  }
  if (iterIS) {
    ierr = ISGetLocalSize(iterIS, &Nelem); CHKERRQ(ierr);
    ierr = ISGetIndices(iterIS, &iterIndices); CHKERRQ(ierr);
  } else {
    Nelem = 0;
    iterIndices = NULL;
  }
  ierr = PetscMalloc1(Nelem*PetscPowInt(P, dim), &erestrict); CHKERRQ(ierr);
  for (p=0,eoffset=0; p<Nelem; p++) {
    PetscInt c = iterIndices[p];
    PetscInt numindices, *indices, nnodes;
    ierr = DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE,
                                   &numindices, &indices, NULL, NULL);
    CHKERRQ(ierr);
    bool flip = false;
    if (height > 0) {
      PetscInt numCells, numFaces, start = -1;
      const PetscInt *orients, *faces, *cells;
      ierr = DMPlexGetSupport(dm, c, &cells); CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, c, &numCells); CHKERRQ(ierr);
      if (numCells != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                                    "Expected one cell in support of exterior face, but got %D cells",
                                    numCells);
      ierr = DMPlexGetCone(dm, cells[0], &faces); CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, cells[0], &numFaces); CHKERRQ(ierr);
      for (PetscInt i=0; i<numFaces; i++) {if (faces[i] == c) start = i;}
      if (start < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT,
                                "Could not find face %D in cone of its support",
                                c);
      ierr = DMPlexGetConeOrientation(dm, cells[0], &orients); CHKERRQ(ierr);
      if (orients[start] < 0) flip = true;
    }
    if (numindices % fieldoff[nfields]) SETERRQ1(PETSC_COMM_SELF,
          PETSC_ERR_ARG_INCOMP, "Number of closure indices not compatible with Cell %D",
          c);
    nnodes = numindices / fieldoff[nfields];
    for (PetscInt i=0; i<nnodes; i++) {
      PetscInt ii = i;
      if (flip) {
        if (P == nnodes) ii = nnodes - 1 - i;
        else if (P*P == nnodes) {
          PetscInt row = i / P, col = i % P;
          ii = row + col * P;
        } else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP,
                          "No support for flipping point with %D nodes != P (%D) or P^2",
                          nnodes, P);
      }
      // Check that indices are blocked by node and thus can be coalesced as a single field with
      // fieldoff[nfields] = sum(ncomp) components.
      for (PetscInt f=0; f<nfields; f++) {
        for (PetscInt j=0; j<ncomp[f]; j++) {
          if (Involute(indices[fieldoff[f]*nnodes + ii*ncomp[f] + j])
              != Involute(indices[ii*ncomp[0]]) + fieldoff[f] + j)
            SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                     "Cell %D closure indices not interlaced for node %D field %D component %D",
                     c, ii, f, j);
        }
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = Involute(indices[ii*ncomp[0]]);
      erestrict[eoffset++] = loc;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE,
                                       &numindices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }
  if (eoffset != Nelem*PetscPowInt(P, dim))
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB,
             "ElemRestriction of size (%D,%D) initialized %D nodes", Nelem,
             PetscPowInt(P, dim),eoffset);
  if (iterIS) {
    ierr = ISRestoreIndices(iterIS, &iterIndices); CHKERRQ(ierr);
  }
  ierr = ISDestroy(&iterIS); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &Uloc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Uloc, &Ndof); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Uloc); CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, Nelem, PetscPowInt(P, dim), fieldoff[nfields],
                            1, Ndof, CEED_MEM_HOST, CEED_COPY_VALUES, erestrict,
                            Erestrict);
  ierr = PetscFree(erestrict); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// Utility function to get Ceed Restriction for each domain
static PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height,
    DMLabel domainLabel, PetscInt value, CeedInt P, CeedInt Q, CeedInt qdatasize,
    CeedElemRestriction *restrictq, CeedElemRestriction *restrictx,
    CeedElemRestriction *restrictqdi) {

  DM dmcoord;
  CeedInt dim, localNelem;
  CeedInt Qdim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  Qdim = CeedIntPow(Q, dim);
  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dm, P, height, domainLabel, value,
                                   restrictq);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dmcoord, 2, height, domainLabel, value,
                                   restrictx);
  CHKERRQ(ierr);
  CeedElemRestrictionGetNumElements(*restrictq, &localNelem);
  CeedElemRestrictionCreateStrided(ceed, localNelem, Qdim,
                                   qdatasize, qdatasize*localNelem*Qdim,
                                   CEED_STRIDES_BACKEND, restrictqdi);
  PetscFunctionReturn(0);
}

// Utility function to create CEED Composite Operator for the entire domain
static PetscErrorCode CreateOperatorForDomain(Ceed ceed, DM dm, SimpleBC bc,
    WindType wind_type, CeedOperator op_applyVol, CeedQFunction qf_applySur,
    CeedQFunction qf_setupSur, CeedInt height, CeedInt numP_Sur, CeedInt numQ_Sur,
    CeedInt qdatasizeSur, CeedInt NqptsSur, CeedBasis basisxSur,
    CeedBasis basisqSur, CeedOperator *op_apply) {

  CeedInt dim, nFace;
  PetscInt lsize, localNelemSur[6];
  Vec Xloc;
  CeedVector xcorners, qdataSur[6];
  CeedOperator op_setupSur[6], op_applySur[6];
  CeedElemRestriction restrictxSur[6], restrictqSur[6], restrictqdiSur[6];
  DMLabel domainLabel;
  PetscScalar *x;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  // Composite Operaters
  CeedCompositeOperatorCreate(ceed, op_apply);
  CeedCompositeOperatorAddSub(*op_apply,
                              op_applyVol); // Apply a Sub-Operator for the volume

  if (wind_type == ADVECTION_WIND_TRANSLATION) {
    bc->nwall = 0;
    bc->nslip[0] = bc->nslip[1] = bc->nslip[2] = 0;
    ierr = DMGetCoordinatesLocal(dm, &Xloc); CHKERRQ(ierr);
    ierr = VecGetLocalSize(Xloc, &lsize); CHKERRQ(ierr);
    ierr = CeedVectorCreate(ceed, lsize, &xcorners); CHKERRQ(ierr);
    ierr = VecGetArray(Xloc, &x); CHKERRQ(ierr);
    CeedVectorSetArray(xcorners, CEED_MEM_HOST, CEED_USE_POINTER, x);
    ierr = DMGetLabel(dm, "Face Sets", &domainLabel); CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
    if (dim == 2) nFace = 4;
    if (dim == 3) nFace = 6;

    // Create CEED Operator for each boundary face
    for (CeedInt i=0; i<nFace; i++) {
      ierr = GetRestrictionForDomain(ceed, dm, height, domainLabel, i+1, numP_Sur,
                                     numQ_Sur, qdatasizeSur, &restrictqSur[i],
                                     &restrictxSur[i], &restrictqdiSur[i]);
      CHKERRQ(ierr);
      // Create the CEED vectors that will be needed in Boundary setup
      CeedElemRestrictionGetNumElements(restrictqSur[i], &localNelemSur[i]);
      CeedVectorCreate(ceed, qdatasizeSur*localNelemSur[i]*NqptsSur,
                       &qdataSur[i]);
      // Create the operator that builds the quadrature data for the Boundary operator
      CeedOperatorCreate(ceed, qf_setupSur, NULL, NULL, &op_setupSur[i]);
      CeedOperatorSetField(op_setupSur[i], "dx", restrictxSur[i], basisxSur,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_setupSur[i], "weight", CEED_ELEMRESTRICTION_NONE,
                           basisxSur, CEED_VECTOR_NONE);
      CeedOperatorSetField(op_setupSur[i], "qdataSur", restrictqdiSur[i],
                           CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
      // Create Boundary operator
      CeedOperatorCreate(ceed, qf_applySur, NULL, NULL, &op_applySur[i]);
      CeedOperatorSetField(op_applySur[i], "q", restrictqSur[i], basisqSur,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_applySur[i], "qdataSur", restrictqdiSur[i],
                           CEED_BASIS_COLLOCATED, qdataSur[i]);
      CeedOperatorSetField(op_applySur[i], "x", restrictxSur[i], basisxSur,
                           xcorners);
      CeedOperatorSetField(op_applySur[i], "v", restrictqSur[i], basisqSur,
                           CEED_VECTOR_ACTIVE);
      // Apply CEED operator for Boundary setup
      CeedOperatorApply(op_setupSur[i], xcorners, qdataSur[i],
                        CEED_REQUEST_IMMEDIATE);
      // Apply Sub-Operator for the Boundary
      CeedCompositeOperatorAddSub(*op_apply, op_applySur[i]);
    }
    CeedVectorDestroy(&xcorners);
  }
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
  PetscInt mceed, mpetsc;
  PetscScalar *a;

  PetscFunctionBeginUser;
  ierr = CeedVectorGetLength(c, &mceed); CHKERRQ(ierr);
  ierr = VecGetLocalSize(p, &mpetsc); CHKERRQ(ierr);
  if (mceed != mpetsc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                                  "Cannot place PETSc Vec of length %D in CeedVector of length %D",
                                  mpetsc, mceed);
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
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  if (user->dmviz) {
    Vec Qrefined, Qrefined_loc;
    char filepath_refined[PETSC_MAX_PATH_LEN];
    PetscViewer viewer_refined;

    ierr = DMGetGlobalVector(user->dmviz, &Qrefined); CHKERRQ(ierr);
    ierr = DMGetLocalVector(user->dmviz, &Qrefined_loc); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)Qrefined_loc, "Refined");
    CHKERRQ(ierr);
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
  #if PETSC_VERSION_GE(3,13,0)
  ierr = PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL);
  #else
  ierr = PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL, true);
  #endif
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ICs_FixMultiplicity(CeedOperator op_ics,
    CeedVector xcorners, CeedVector q0ceed, DM dm, Vec Qloc, Vec Q,
    CeedElemRestriction restrictq, CeedQFunctionContext ctxSetup, CeedScalar time) {
  PetscErrorCode ierr;
  CeedVector multlvec;
  Vec Multiplicity, MultiplicityLoc;

  SetupContext ctxSetupData;
  CeedQFunctionContextGetData(ctxSetup, CEED_MEM_HOST, (void **)&ctxSetupData);
  ctxSetupData->time = time;
  CeedQFunctionContextRestoreData(ctxSetup, (void **)&ctxSetupData);

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
    CeedElemRestriction restrictq, CeedBasis basisq,
    CeedElemRestriction restrictqdi, CeedVector qdata, Vec M) {
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
  CeedQFunctionCreateInterior(ceed, 1, Mass, Mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "q", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_mass, "v", ncompq, CEED_EVAL_INTERP);

  // Create the mass operator
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "q", restrictq, basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", restrictqdi,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_mass, "v", restrictq, basisq, CEED_VECTOR_ACTIVE);

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

static PetscErrorCode SetUpDM(DM dm, problemData *problem, PetscInt degree,
                              SimpleBC bc, void *ctxSetupData) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  {
    // Configure the finite element space and boundary conditions
    PetscFE fe;
    PetscInt ncompq = 5;
    ierr = PetscFECreateLagrange(PETSC_COMM_SELF, problem->dim, ncompq,
                                 PETSC_FALSE, degree, PETSC_DECIDE,
                                 &fe); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe, "Q"); CHKERRQ(ierr);
    ierr = DMAddField(dm, NULL,(PetscObject)fe); CHKERRQ(ierr);
    ierr = DMCreateDS(dm); CHKERRQ(ierr);
    {
      PetscInt comps[1] = {1};
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", "Face Sets", 0,
                           1, comps, (void(*)(void))NULL, NULL, bc->nslip[0],
                           bc->slips[0], ctxSetupData); CHKERRQ(ierr);
      comps[0] = 2;
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", "Face Sets", 0,
                           1, comps, (void(*)(void))NULL, NULL, bc->nslip[1],
                           bc->slips[1], ctxSetupData); CHKERRQ(ierr);
      comps[0] = 3;
      ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", "Face Sets", 0,
                           1, comps, (void(*)(void))NULL, NULL, bc->nslip[2],
                           bc->slips[2], ctxSetupData); CHKERRQ(ierr);
    }
    if (bc->userbc == PETSC_TRUE) {
      for (PetscInt c = 0; c < 3; c++) {
        for (PetscInt s = 0; s < bc->nslip[c]; s++) {
          for (PetscInt w = 0; w < bc->nwall; w++) {
            if (bc->slips[c][s] == bc->walls[w])
              SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                       "Boundary condition already set on face %D!\n",
                       bc->walls[w]);

          }
        }
      }
    }
    // Wall boundary conditions are zero energy density and zero flux for
    //   velocity in advection/advection2d, and zero velocity and zero flux
    //   for mass density and energy density in density_current
    {
      if (problem->bc == Exact_Advection || problem->bc == Exact_Advection2d) {
        PetscInt comps[1] = {4};
        ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "Face Sets", 0,
                             1, comps, (void(*)(void))problem->bc, NULL,
                             bc->nwall, bc->walls, ctxSetupData); CHKERRQ(ierr);
      } else if (problem->bc == Exact_DC) {
        PetscInt comps[3] = {1, 2, 3};
        ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "Face Sets", 0,
                             3, comps, (void(*)(void))problem->bc, NULL,
                             bc->nwall, bc->walls, ctxSetupData); CHKERRQ(ierr);
      } else
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
                "Undefined boundary conditions for this problem");
    }
    ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);
    CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);
  }
  {
    // Empty name for conserved field (because there is only one field)
    PetscSection section;
    ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, 0, ""); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 0, "Density");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 1, "MomentumX");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 2, "MomentumY");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 3, "MomentumZ");
    CHKERRQ(ierr);
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
  PetscInt localNelemVol, lnodes, gnodes, steps;
  const PetscInt ncompq = 5;
  PetscMPIInt rank;
  PetscScalar ftime;
  Vec Q, Qloc, Xloc;
  Ceed ceed;
  CeedInt numP, numQ;
  CeedVector xcorners, qdata, q0ceed;
  CeedBasis basisx, basisxc, basisq;
  CeedElemRestriction restrictx, restrictq, restrictqdi;
  CeedQFunction qf_setupVol, qf_ics, qf_rhsVol, qf_ifunctionVol;
  CeedQFunctionContext ctxSetup, ctxNS, ctxAdvection2d, ctxSurface;
  CeedOperator op_setupVol, op_ics;
  CeedScalar Rd;
  CeedMemType memtyperequested;
  PetscScalar WpermK, Pascal, JperkgK, mpersquareds, kgpercubicm,
              kgpersquaredms, Joulepercubicm, Joule;
  problemType problemChoice;
  problemData *problem = NULL;
  WindType wind_type;
  StabilizationType stab;
  testType testChoice;
  testData *test = NULL;
  PetscBool implicit;
  PetscInt    viz_refine = 0;
  struct SimpleBC_ bc = {
    .nslip = {2, 2, 2},
    .slips = {{5, 6}, {3, 4}, {1, 2}}
  };
  double start, cpu_time_used;
  // Check PETSc CUDA support
  PetscBool petschavecuda, setmemtyperequest = PETSC_FALSE;
  // *INDENT-OFF*
  #ifdef PETSC_HAVE_CUDA
  petschavecuda = PETSC_TRUE;
  #else
  petschavecuda = PETSC_FALSE;
  #endif
  // *INDENT-ON*

  // Create the libCEED contexts
  PetscScalar meter      = 1e-2;     // 1 meter in scaled length units
  PetscScalar second     = 1e-2;     // 1 second in scaled time units
  PetscScalar kilogram   = 1e-6;     // 1 kilogram in scaled mass units
  PetscScalar Kelvin     = 1;        // 1 Kelvin in scaled temperature units
  CeedScalar theta0      = 300.;     // K
  CeedScalar thetaC      = -15.;     // K
  CeedScalar P0          = 1.e5;     // Pa
  CeedScalar E_wind      = 1.e6;     // J
  CeedScalar N           = 0.01;     // 1/s
  CeedScalar cv          = 717.;     // J/(kg K)
  CeedScalar cp          = 1004.;    // J/(kg K)
  CeedScalar g           = 9.81;     // m/s^2
  CeedScalar lambda      = -2./3.;   // -
  CeedScalar mu          = 75.;      // Pa s, dynamic viscosity
  // mu = 75 is not physical for air, but is good for numerical stability
  CeedScalar k           = 0.02638;  // W/(m K)
  CeedScalar CtauS       = 0.;       // dimensionless
  CeedScalar strong_form = 0.;       // [0,1]
  PetscScalar lx         = 8000.;    // m
  PetscScalar ly         = 8000.;    // m
  PetscScalar lz         = 4000.;    // m
  CeedScalar rc          = 1000.;    // m (Radius of bubble)
  PetscScalar resx       = 1000.;    // m (resolution in x)
  PetscScalar resy       = 1000.;    // m (resolution in y)
  PetscScalar resz       = 1000.;    // m (resolution in z)
  PetscInt outputfreq    = 10;       // -
  PetscInt contsteps     = 0;        // -
  PetscInt degree        = 1;        // -
  PetscInt qextra        = 2;        // -
  PetscInt qextraSur     = 2;        // -
  PetscReal center[3], dc_axis[3] = {0, 0, 0}, wind[3] = {1., 0, 0};

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
  testChoice = TEST_NONE;
  ierr = PetscOptionsEnum("-test", "Run tests", NULL,
                          testTypes, (PetscEnum)testChoice,
                          (PetscEnum *)&testChoice,
                          NULL); CHKERRQ(ierr);
  test = &testOptions[testChoice];
  problemChoice = NS_DENSITY_CURRENT;
  ierr = PetscOptionsEnum("-problem", "Problem to solve", NULL,
                          problemTypes, (PetscEnum)problemChoice,
                          (PetscEnum *)&problemChoice, NULL); CHKERRQ(ierr);
  problem = &problemOptions[problemChoice];
  ierr = PetscOptionsEnum("-problem_advection_wind", "Wind type in Advection",
                          NULL, WindTypes,
                          (PetscEnum)(wind_type = ADVECTION_WIND_ROTATION),
                          (PetscEnum *)&wind_type, NULL); CHKERRQ(ierr);
  PetscInt n = problem->dim;
  PetscBool userWind;
  ierr = PetscOptionsRealArray("-problem_advection_wind_translation",
                               "Constant wind vector",
                               NULL, wind, &n, &userWind); CHKERRQ(ierr);
  if (wind_type == ADVECTION_WIND_ROTATION && userWind) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -problem_advection_wind_translation only with -problem_advection_wind translation\n");
    CHKERRQ(ierr);
  }
  if (wind_type == ADVECTION_WIND_TRANSLATION
      && problemChoice == NS_DENSITY_CURRENT) {
    SETERRQ(comm, PETSC_ERR_ARG_INCOMP,
            "-problem_advection_wind translation is not defined for -problem density_current");
  }
  ierr = PetscOptionsEnum("-stab", "Stabilization method", NULL,
                          StabilizationTypes, (PetscEnum)(stab = STAB_NONE),
                          (PetscEnum *)&stab, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation",
                          NULL, implicit=PETSC_FALSE, &implicit, NULL);
  CHKERRQ(ierr);
  if (!implicit && stab != STAB_NONE) {
    ierr = PetscPrintf(comm, "Warning! Use -stab only with -implicit\n");
    CHKERRQ(ierr);
  }
  {
    PetscInt len;
    PetscBool flg;
    ierr = PetscOptionsIntArray("-bc_wall",
                                "Use wall boundary conditions on this list of faces",
                                NULL, bc.walls,
                                (len = sizeof(bc.walls) / sizeof(bc.walls[0]),
                                 &len), &flg); CHKERRQ(ierr);
    if (flg) {
      bc.nwall = len;
      // Using a no-slip wall disables automatic slip walls (they must be set explicitly)
      bc.nslip[0] = bc.nslip[1] = bc.nslip[2] = 0;
    }
    for (PetscInt j=0; j<3; j++) {
      const char *flags[3] = {"-bc_slip_x", "-bc_slip_y", "-bc_slip_z"};
      ierr = PetscOptionsIntArray(flags[j],
                                  "Use slip boundary conditions on this list of faces",
                                  NULL, bc.slips[j],
                                  (len = sizeof(bc.slips[j]) / sizeof(bc.slips[j][0]),
                                   &len), &flg);
      CHKERRQ(ierr);
      if (flg) {
        bc.nslip[j] = len;
        bc.userbc = PETSC_TRUE;
      }
    }
  }
  ierr = PetscOptionsInt("-viz_refine",
                         "Regular refinement levels for visualization",
                         NULL, viz_refine, &viz_refine, NULL);
  CHKERRQ(ierr);
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
  ierr = PetscOptionsScalar("-E_wind", "Total energy of inflow wind",
                            NULL, E_wind, &E_wind, NULL); CHKERRQ(ierr);
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
  if (stab == STAB_NONE && CtauS != 0) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -CtauS only with -stab su or -stab supg\n");
    CHKERRQ(ierr);
  }
  ierr = PetscOptionsScalar("-strong_form",
                            "Strong (1) or weak/integrated by parts (0) advection residual",
                            NULL, strong_form, &strong_form, NULL);
  CHKERRQ(ierr);
  if (problemChoice == NS_DENSITY_CURRENT && (CtauS != 0 || strong_form != 0)) {
    ierr = PetscPrintf(comm,
                       "Warning! Problem density_current does not support -CtauS or -strong_form\n");
    CHKERRQ(ierr);
  }
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
    PetscReal norm = PetscSqrtReal(PetscSqr(dc_axis[0]) +
                                   PetscSqr(dc_axis[1]) + PetscSqr(dc_axis[2]));
    if (norm > 0) {
      for (int i=0; i<3; i++) dc_axis[i] /= norm;
    }
  }
  ierr = PetscOptionsInt("-output_freq",
                         "Frequency of output, in number of steps",
                         NULL, outputfreq, &outputfreq, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-continue", "Continue from previous solution",
                         NULL, contsteps, &contsteps, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-degree", "Polynomial degree of finite elements",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  PetscBool userQextraSur;
  ierr = PetscOptionsInt("-qextra_boundary",
                         "Number of extra quadrature points on in/outflow faces",
                         NULL, qextraSur, &qextraSur, &userQextraSur);
  CHKERRQ(ierr);
  if ((wind_type == ADVECTION_WIND_ROTATION
       || problemChoice == NS_DENSITY_CURRENT) && userQextraSur) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -qextra_boundary only with -problem_advection_wind translation\n");
    CHKERRQ(ierr);
  }
  ierr = PetscStrncpy(user->outputfolder, ".", 2); CHKERRQ(ierr);
  ierr = PetscOptionsString("-of", "Output folder",
                            NULL, user->outputfolder, user->outputfolder,
                            sizeof(user->outputfolder), NULL); CHKERRQ(ierr);
  memtyperequested = petschavecuda ? CEED_MEM_DEVICE : CEED_MEM_HOST;
  ierr = PetscOptionsEnum("-memtype",
                          "CEED MemType requested", NULL,
                          memTypes, (PetscEnum)memtyperequested,
                          (PetscEnum *)&memtyperequested, &setmemtyperequest);
  CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Define derived units
  Pascal = kilogram / (meter * PetscSqr(second));
  JperkgK =  PetscSqr(meter) / (PetscSqr(second) * Kelvin);
  mpersquareds = meter / PetscSqr(second);
  WpermK = kilogram * meter / (pow(second,3) * Kelvin);
  kgpercubicm = kilogram / pow(meter,3);
  kgpersquaredms = kilogram / (PetscSqr(meter) * second);
  Joulepercubicm = kilogram / (meter * PetscSqr(second));
  Joule = kilogram * PetscSqr(meter) / PetscSqr(second);

  // Scale variables to desired units
  theta0 *= Kelvin;
  thetaC *= Kelvin;
  P0 *= Pascal;
  E_wind *= Joule;
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
                qdatasizeVol = problem->qdatasizeVol;
  // Set up the libCEED context
  struct SetupContext_ ctxSetupData = {
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
    .center[0] = center[0],
    .center[1] = center[1],
    .center[2] = center[2],
    .dc_axis[0] = dc_axis[0],
    .dc_axis[1] = dc_axis[1],
    .dc_axis[2] = dc_axis[2],
    .wind[0] = wind[0],
    .wind[1] = wind[1],
    .wind[2] = wind[2],
    .time = 0,
    .wind_type = wind_type,
  };

  // Create the mesh
  {
    const PetscReal scale[3] = {lx, ly, lz};
    ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, NULL, NULL, scale,
                               NULL, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
  }

  // Distribute the mesh over processes
  {
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

  // Setup DM
  ierr = DMLocalizeCoordinates(dm); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = SetUpDM(dm, problem, degree, &bc, &ctxSetupData); CHKERRQ(ierr);

  // Refine DM for high-order viz
  dmviz = NULL;
  interpviz = NULL;
  if (viz_refine) {
    DM dmhierarchy[viz_refine+1];

    ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE); CHKERRQ(ierr);
    dmhierarchy[0] = dm;
    for (PetscInt i = 0, d = degree; i < viz_refine; i++) {
      Mat interp_next;

      ierr = DMRefine(dmhierarchy[i], MPI_COMM_NULL, &dmhierarchy[i+1]);
      CHKERRQ(ierr);
      ierr = DMClearDS(dmhierarchy[i+1]); CHKERRQ(ierr);
      ierr = DMClearFields(dmhierarchy[i+1]); CHKERRQ(ierr);
      ierr = DMSetCoarseDM(dmhierarchy[i+1], dmhierarchy[i]); CHKERRQ(ierr);
      d = (d + 1) / 2;
      if (i + 1 == viz_refine) d = 1;
      ierr = SetUpDM(dmhierarchy[i+1], problem, d, &bc, &ctxSetupData);
      CHKERRQ(ierr);
      ierr = DMCreateInterpolation(dmhierarchy[i], dmhierarchy[i+1],
                                   &interp_next, NULL); CHKERRQ(ierr);
      if (!i) interpviz = interp_next;
      else {
        Mat C;
        ierr = MatMatMult(interp_next, interpviz, MAT_INITIAL_MATRIX,
                          PETSC_DECIDE, &C); CHKERRQ(ierr);
        ierr = MatDestroy(&interp_next); CHKERRQ(ierr);
        ierr = MatDestroy(&interpviz); CHKERRQ(ierr);
        interpviz = C;
      }
    }
    for (PetscInt i=1; i<viz_refine; i++) {
      ierr = DMDestroy(&dmhierarchy[i]); CHKERRQ(ierr);
    }
    dmviz = dmhierarchy[viz_refine];
  }
  ierr = DMCreateGlobalVector(dm, &Q); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &Qloc); CHKERRQ(ierr);
  ierr = VecGetSize(Qloc, &lnodes); CHKERRQ(ierr);
  lnodes /= ncompq;

  // Initialize CEED
  CeedInit(ceedresource, &ceed);
  // Set memtype
  CeedMemType memtypebackend;
  CeedGetPreferredMemType(ceed, &memtypebackend);
  // Check memtype compatibility
  if (!setmemtyperequest)
    memtyperequested = memtypebackend;
  else if (!petschavecuda && memtyperequested == CEED_MEM_DEVICE)
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP_SYS,
             "PETSc was not built with CUDA. "
             "Requested MemType CEED_MEM_DEVICE is not supported.", NULL);

  // Set number of 1D nodes and quadrature points
  numP = degree + 1;
  numQ = numP + qextra;

  // Print summary
  if (testChoice == TEST_NONE) {
    CeedInt gdofs, odofs;
    int comm_size;
    char box_faces_str[PETSC_MAX_PATH_LEN] = "NONE";
    ierr = VecGetSize(Q, &gdofs); CHKERRQ(ierr);
    ierr = VecGetLocalSize(Q, &odofs); CHKERRQ(ierr);
    gnodes = gdofs/ncompq;
    ierr = MPI_Comm_size(comm, &comm_size); CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL, NULL, "-dm_plex_box_faces", box_faces_str,
                                 sizeof(box_faces_str), NULL); CHKERRQ(ierr);
    const char *usedresource;
    CeedGetResource(ceed, &usedresource);

    ierr = PetscPrintf(comm,
                       "\n-- Navier-Stokes solver - libCEED + PETSc --\n"
                       "  rank(s)                              : %d\n"
                       "  Problem:\n"
                       "    Problem Name                       : %s\n"
                       "    Stabilization                      : %s\n"
                       "  PETSc:\n"
                       "    Box Faces                          : %s\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "    libCEED User Requested MemType     : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (P)       : %d\n"
                       "    Number of 1D Quadrature Points (Q) : %d\n"
                       "    Global DoFs                        : %D\n"
                       "    Owned DoFs                         : %D\n"
                       "    DoFs per node                      : %D\n"
                       "    Global nodes                       : %D\n"
                       "    Owned nodes                        : %D\n",
                       comm_size, problemTypes[problemChoice],
                       StabilizationTypes[stab], box_faces_str, usedresource,
                       CeedMemTypes[memtypebackend],
                       (setmemtyperequest) ?
                       CeedMemTypes[memtyperequested] : "none",
                       numP, numQ, gdofs, odofs, ncompq, gnodes, lnodes);
    CHKERRQ(ierr);
  }

  // Set up global mass vector
  ierr = VecDuplicate(Q, &user->M); CHKERRQ(ierr);

  // Set up libCEED
  // CEED Bases
  CeedInit(ceedresource, &ceed);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompq, numP, numQ, CEED_GAUSS,
                                  &basisq);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, numQ, CEED_GAUSS,
                                  &basisx);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, numP,
                                  CEED_GAUSS_LOBATTO, &basisxc);
  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  // CEED Restrictions
  ierr = GetRestrictionForDomain(ceed, dm, 0, 0, 0, numP, numQ,
                                 qdatasizeVol, &restrictq, &restrictx,
                                 &restrictqdi); CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dm, &Xloc); CHKERRQ(ierr);
  ierr = CreateVectorFromPetscVec(ceed, Xloc, &xcorners); CHKERRQ(ierr);

  // Create the CEED vectors that will be needed in setup
  CeedInt NqptsVol;
  CeedBasisGetNumQuadraturePoints(basisq, &NqptsVol);
  CeedElemRestrictionGetNumElements(restrictq, &localNelemVol);
  CeedVectorCreate(ceed, qdatasizeVol*localNelemVol*NqptsVol, &qdata);
  CeedElemRestrictionCreateVector(restrictq, &q0ceed, NULL);

  // Create the Q-function that builds the quadrature data for the NS operator
  CeedQFunctionCreateInterior(ceed, 1, problem->setupVol, problem->setupVol_loc,
                              &qf_setupVol);
  CeedQFunctionAddInput(qf_setupVol, "dx", ncompx*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setupVol, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setupVol, "qdata", qdatasizeVol, CEED_EVAL_NONE);

  // Create the Q-function that sets the ICs of the operator
  CeedQFunctionCreateInterior(ceed, 1, problem->ics, problem->ics_loc, &qf_ics);
  CeedQFunctionAddInput(qf_ics, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ics, "q0", ncompq, CEED_EVAL_NONE);

  qf_rhsVol = NULL;
  if (problem->applyVol_rhs) { // Create the Q-function that defines the action of the RHS operator
    CeedQFunctionCreateInterior(ceed, 1, problem->applyVol_rhs,
                                problem->applyVol_rhs_loc, &qf_rhsVol);
    CeedQFunctionAddInput(qf_rhsVol, "q", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_rhsVol, "dq", ncompq*dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_rhsVol, "qdata", qdatasizeVol, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_rhsVol, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_rhsVol, "v", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_rhsVol, "dv", ncompq*dim, CEED_EVAL_GRAD);
  }

  qf_ifunctionVol = NULL;
  if (problem->applyVol_ifunction) { // Create the Q-function that defines the action of the IFunction
    CeedQFunctionCreateInterior(ceed, 1, problem->applyVol_ifunction,
                                problem->applyVol_ifunction_loc, &qf_ifunctionVol);
    CeedQFunctionAddInput(qf_ifunctionVol, "q", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_ifunctionVol, "dq", ncompq*dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_ifunctionVol, "qdot", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_ifunctionVol, "qdata", qdatasizeVol, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_ifunctionVol, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_ifunctionVol, "v", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_ifunctionVol, "dv", ncompq*dim, CEED_EVAL_GRAD);
  }

  // Create the operator that builds the quadrature data for the NS operator
  CeedOperatorCreate(ceed, qf_setupVol, NULL, NULL, &op_setupVol);
  CeedOperatorSetField(op_setupVol, "dx", restrictx, basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setupVol, "weight", CEED_ELEMRESTRICTION_NONE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setupVol, "qdata", restrictqdi,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the operator that sets the ICs
  CeedOperatorCreate(ceed, qf_ics, NULL, NULL, &op_ics);
  CeedOperatorSetField(op_ics, "x", restrictx, basisxc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ics, "q0", restrictq,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedElemRestrictionCreateVector(restrictq, &user->qceed, NULL);
  CeedElemRestrictionCreateVector(restrictq, &user->qdotceed, NULL);
  CeedElemRestrictionCreateVector(restrictq, &user->gceed, NULL);

  if (qf_rhsVol) { // Create the RHS physics operator
    CeedOperator op;
    CeedOperatorCreate(ceed, qf_rhsVol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdata", restrictqdi,
                         CEED_BASIS_COLLOCATED, qdata);
    CeedOperatorSetField(op, "x", restrictx, basisx, xcorners);
    CeedOperatorSetField(op, "v", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", restrictq, basisq, CEED_VECTOR_ACTIVE);
    user->op_rhs_vol = op;
  }

  if (qf_ifunctionVol) { // Create the IFunction operator
    CeedOperator op;
    CeedOperatorCreate(ceed, qf_ifunctionVol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdot", restrictq, basisq, user->qdotceed);
    CeedOperatorSetField(op, "qdata", restrictqdi,
                         CEED_BASIS_COLLOCATED, qdata);
    CeedOperatorSetField(op, "x", restrictx, basisx, xcorners);
    CeedOperatorSetField(op, "v", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", restrictq, basisq, CEED_VECTOR_ACTIVE);
    user->op_ifunction_vol = op;
  }

  // Set up CEED for the boundaries
  CeedInt height = 1;
  CeedInt dimSur = dim - height;
  CeedInt numP_Sur = degree + 1;
  CeedInt numQ_Sur = numP_Sur + qextraSur;
  const CeedInt qdatasizeSur = problem->qdatasizeSur;
  CeedBasis basisxSur, basisxcSur, basisqSur;
  CeedInt NqptsSur;
  CeedQFunction qf_setupSur, qf_applySur;

  // CEED bases for the boundaries
  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, ncompq, numP_Sur, numQ_Sur,
                                  CEED_GAUSS,
                                  &basisqSur);
  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, ncompx, 2, numQ_Sur, CEED_GAUSS,
                                  &basisxSur);
  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, ncompx, 2, numP_Sur,
                                  CEED_GAUSS_LOBATTO, &basisxcSur);
  CeedBasisGetNumQuadraturePoints(basisqSur, &NqptsSur);

  // Create the Q-function that builds the quadrature data for the Surface operator
  CeedQFunctionCreateInterior(ceed, 1, problem->setupSur, problem->setupSur_loc,
                              &qf_setupSur);
  CeedQFunctionAddInput(qf_setupSur, "dx", ncompx*dimSur, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setupSur, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setupSur, "qdataSur", qdatasizeSur, CEED_EVAL_NONE);

  // Creat Q-Function for Boundaries
  qf_applySur = NULL;
  if (problem->applySur) {
    CeedQFunctionCreateInterior(ceed, 1, problem->applySur,
                                problem->applySur_loc, &qf_applySur);
    CeedQFunctionAddInput(qf_applySur, "q", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_applySur, "qdataSur", qdatasizeSur, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_applySur, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_applySur, "v", ncompq, CEED_EVAL_INTERP);
  }

  // Create CEED Operator for the whole domain
  if (!implicit)
    ierr = CreateOperatorForDomain(ceed, dm, &bc, wind_type, user->op_rhs_vol,
                                   qf_applySur, qf_setupSur,
                                   height, numP_Sur, numQ_Sur, qdatasizeSur,
                                   NqptsSur, basisxSur, basisqSur,
                                   &user->op_rhs); CHKERRQ(ierr);
  if (implicit)
    ierr = CreateOperatorForDomain(ceed, dm, &bc, wind_type,
                                   user->op_ifunction_vol,
                                   qf_applySur, qf_setupSur,
                                   height, numP_Sur, numQ_Sur, qdatasizeSur,
                                   NqptsSur, basisxSur, basisqSur,
                                   &user->op_ifunction); CHKERRQ(ierr);
  // Set up contex for QFunctions
  CeedQFunctionContextCreate(ceed, &ctxSetup);
  CeedQFunctionContextSetData(ctxSetup, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof ctxSetupData, &ctxSetupData);
  CeedQFunctionSetContext(qf_ics, ctxSetup);

  CeedScalar ctxNSData[8] = {lambda, mu, k, cv, cp, g, Rd};
  CeedQFunctionContextCreate(ceed, &ctxNS);
  CeedQFunctionContextSetData(ctxNS, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof ctxNSData, &ctxNSData);

  struct Advection2dContext_ ctxAdvection2dData = {
    .CtauS = CtauS,
    .strong_form = strong_form,
    .stabilization = stab,
  };
  CeedQFunctionContextCreate(ceed, &ctxAdvection2d);
  CeedQFunctionContextSetData(ctxAdvection2d, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof ctxAdvection2dData, &ctxAdvection2dData);

  struct SurfaceContext_ ctxSurfaceData = {
    .E_wind = E_wind,
    .strong_form = strong_form,
    .implicit = implicit,
  };
  CeedQFunctionContextCreate(ceed, &ctxSurface);
  CeedQFunctionContextSetData(ctxSurface, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof ctxSurfaceData, &ctxSurfaceData);

  switch (problemChoice) {
  case NS_DENSITY_CURRENT:
    if (qf_rhsVol) CeedQFunctionSetContext(qf_rhsVol, ctxNS);
    if (qf_ifunctionVol) CeedQFunctionSetContext(qf_ifunctionVol, ctxNS);
    break;
  case NS_ADVECTION:
  case NS_ADVECTION2D:
    if (qf_rhsVol) CeedQFunctionSetContext(qf_rhsVol, ctxAdvection2d);
    if (qf_ifunctionVol) CeedQFunctionSetContext(qf_ifunctionVol, ctxAdvection2d);
    if (qf_applySur) CeedQFunctionSetContext(qf_applySur, ctxSurface);
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
  units->Joule = Joule;

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
  CeedOperatorApply(op_setupVol, xcorners, qdata, CEED_REQUEST_IMMEDIATE);
  ierr = ComputeLumpedMassMatrix(ceed, dm, restrictq, basisq, restrictqdi, qdata,
                                 user->M); CHKERRQ(ierr);

  ierr = ICs_FixMultiplicity(op_ics, xcorners, q0ceed, dm, Qloc, Q, restrictq,
                             ctxSetup, 0.0); CHKERRQ(ierr);
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
                                      "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_NS);
    CHKERRQ(ierr);
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
    if (!user->op_rhs) SETERRQ(comm, PETSC_ERR_ARG_NULL,
                                 "Problem does not provide RHSFunction");
    ierr = TSSetType(ts, TSRK); CHKERRQ(ierr);
    ierr = TSRKSetType(ts, TSRK5F); CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(ts, 500. * units->second); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 1.e-2 * units->second); CHKERRQ(ierr);
  if (testChoice != TEST_NONE) {ierr = TSSetMaxSteps(ts, 10); CHKERRQ(ierr);}
  ierr = TSGetAdapt(ts, &adapt); CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,
                              1.e-12 * units->second,
                              1.e2 * units->second); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  if (!contsteps) { // print initial condition
    if (testChoice == TEST_NONE) {
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
  if (testChoice == TEST_NONE) {
    ierr = TSMonitorSet(ts, TSMonitor_NS, user, NULL); CHKERRQ(ierr);
  }

  // Solve
  start = MPI_Wtime();
  ierr = PetscBarrier((PetscObject)ts); CHKERRQ(ierr);
  ierr = TSSolve(ts, Q); CHKERRQ(ierr);
  cpu_time_used = MPI_Wtime() - start;
  ierr = TSGetSolveTime(ts, &ftime); CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE, &cpu_time_used, 1, MPI_DOUBLE, MPI_MIN,
                       comm); CHKERRQ(ierr);
  if (testChoice == TEST_NONE) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time taken for solution (sec): %g\n",
                       (double)cpu_time_used); CHKERRQ(ierr);
  }

  // Get error
  if (problem->non_zero_time && testChoice == TEST_NONE) {
    Vec Qexact, Qexactloc;
    PetscReal norm;
    ierr = DMCreateGlobalVector(dm, &Qexact); CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm, &Qexactloc); CHKERRQ(ierr);
    ierr = VecGetSize(Qexactloc, &lnodes); CHKERRQ(ierr);

    ierr = ICs_FixMultiplicity(op_ics, xcorners, q0ceed, dm, Qexactloc, Qexact,
                               restrictq, ctxSetup, ftime); CHKERRQ(ierr);

    ierr = VecAXPY(Q, -1.0, Qexact);  CHKERRQ(ierr);
    ierr = VecNorm(Q, NORM_MAX, &norm); CHKERRQ(ierr);
    CeedVectorDestroy(&q0ceed);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Max Error: %g\n",
                       (double)norm); CHKERRQ(ierr);
    // Clean up vectors
    ierr = DMRestoreLocalVector(dm, &Qexactloc); CHKERRQ(ierr);
    ierr = VecDestroy(&Qexact); CHKERRQ(ierr);
  }

  // Output Statistics
  ierr = TSGetStepNumber(ts, &steps); CHKERRQ(ierr);
  if (testChoice == TEST_NONE) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time integrator took %D time steps to reach final time %g\n",
                       steps, (double)ftime); CHKERRQ(ierr);
  }
  // Output numerical values from command line
  ierr = VecViewFromOptions(Q, NULL, "-vec_view"); CHKERRQ(ierr);

  // Compare reference solution values with current test run for CI
  if (testChoice != TEST_NONE) {
    PetscViewer viewer;
    // Read reference file
    Vec Qref;
    PetscReal error, Qrefnorm;
    ierr = VecDuplicate(Q, &Qref); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, test->filepath, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = VecLoad(Qref, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    // Compute error with respect to reference solution
    ierr = VecAXPY(Q, -1.0, Qref);  CHKERRQ(ierr);
    ierr = VecNorm(Qref, NORM_MAX, &Qrefnorm); CHKERRQ(ierr);
    ierr = VecScale(Q, 1./Qrefnorm); CHKERRQ(ierr);
    ierr = VecNorm(Q, NORM_MAX, &error); CHKERRQ(ierr);
    ierr = VecDestroy(&Qref); CHKERRQ(ierr);
    // Check error
    if (error > test->testtol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
                         "Test failed with error norm %g\n",
                         (double)error); CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
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
  CeedQFunctionDestroy(&qf_setupVol);
  CeedQFunctionDestroy(&qf_ics);
  CeedQFunctionDestroy(&qf_rhsVol);
  CeedQFunctionDestroy(&qf_ifunctionVol);
  CeedQFunctionContextDestroy(&ctxSetup);
  CeedQFunctionContextDestroy(&ctxNS);
  CeedQFunctionContextDestroy(&ctxAdvection2d);
  CeedQFunctionContextDestroy(&ctxSurface);
  CeedOperatorDestroy(&op_setupVol);
  CeedOperatorDestroy(&op_ics);
  CeedOperatorDestroy(&user->op_rhs_vol);
  CeedOperatorDestroy(&user->op_ifunction_vol);
  CeedDestroy(&ceed);
  CeedBasisDestroy(&basisqSur);
  CeedBasisDestroy(&basisxSur);
  CeedBasisDestroy(&basisxcSur);
  CeedQFunctionDestroy(&qf_setupSur);
  CeedQFunctionDestroy(&qf_applySur);
  CeedOperatorDestroy(&user->op_rhs);
  CeedOperatorDestroy(&user->op_ifunction);

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

