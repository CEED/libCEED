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

#include <stdbool.h>
#include <string.h>

#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <petscfe.h>

#include <ceed.h>

#ifndef PHYSICS_STRUCT
#define PHYSICS_STRUCT
typedef struct Physics_private *Physics;
struct Physics_private {
  CeedScalar   nu;      // Poisson's ratio
  CeedScalar   E;       // Young's Modulus
};
#endif

// -----------------------------------------------------------------------------
// Command Line Options
// -----------------------------------------------------------------------------
// Problem options
typedef enum {
  ELAS_LIN = 0, ELAS_HYPER_SS = 1, ELAS_HYPER_FS = 2
} problemType;
static const char *const problemTypes[] = {"linElas",
                                           "hyperSS",
                                           "hyperFS",
                                           "problemType","ELAS_",0
                                          };
static const char *const problemTypesForDisp[] = {"Linear elasticity",
                                                  "Hyper elasticity small strain",
                                                  "Hyper elasticity finite strain"
                                                 };

// Forcing function options
typedef enum {
  FORCE_NONE = 0, FORCE_CONST = 1, FORCE_MMS = 2
} forcingType;
static const char *const forcingTypes[] = {"none",
                                           "constant",
                                           "mms",
                                           "forcingType","FORCE_",0
                                          };
static const char *const forcingTypesForDisp[] = {"None",
                                                  "Constant",
                                                  "Manufactured solution"
                                                 };

// Multigrid options
typedef enum {
  MULTIGRID_LOGARITHMIC = 0, MULTIGRID_UNIFORM = 1, MULTIGRID_NONE = 2
} multigridType;
static const char *const multigridTypes [] = {"logarithmic",
                                              "uniform",
                                              "none",
                                              "multigridType","MULTIGRID",0
                                             };
static const char *const multigridTypesForDisp[] = {"P-multigrid, logarithmic coarsening",
                                                    "P-multigrind, uniform coarsening",
                                                    "No multigrid"
                                                   };

typedef PetscErrorCode BCFunc(PetscInt, PetscReal, const PetscReal *, PetscInt,
                              PetscScalar *, void *);
// Note: These variables should be updated if additional boundary conditions
//         are added to boundary.c.
BCFunc BCMMS, BCZero, BCClamp;

// -----------------------------------------------------------------------------
// Structs
// -----------------------------------------------------------------------------
// Units
typedef struct Units_private *Units;
struct Units_private {
  // Fundamental units
  PetscScalar meter;
  PetscScalar kilogram;
  PetscScalar second;
  // Derived unit
  PetscScalar Pascal;
};

// Application context from user command line options
typedef struct AppCtx_private *AppCtx;
struct AppCtx_private {
  char          ceedResource[PETSC_MAX_PATH_LEN];     // libCEED backend
  char          ceedResourceFine[PETSC_MAX_PATH_LEN]; // libCEED for fine grid
  char          meshFile[PETSC_MAX_PATH_LEN];         // exodusII mesh file
  PetscBool     testMode;
  PetscBool     viewSoln;
  problemType   problemChoice;
  forcingType   forcingChoice;
  multigridType multigridChoice;
  PetscInt      degree;
  PetscInt      numLevels;
  PetscInt      *levelDegrees;
  PetscInt      numIncrements;                         // Number of steps
  PetscInt      bcZeroFaces[16], bcClampFaces[16];
  PetscInt      bcZeroCount, bcClampCount;
};

// Problem specific data
typedef struct {
  CeedInt           qdatasize;
  CeedQFunctionUser setupgeo, apply, jacob;
  const char        *setupgeofname, *applyfname, *jacobfname;
  CeedQuadMode      qmode;
} problemData;

// Data specific to each problem option
problemData problemOptions[3];

// Forcing function data
typedef struct {
  CeedQFunctionUser setupforcing;
  const char        *setupforcingfname;
} forcingData;

forcingData forcingOptions[3];

// Data for PETSc Matshell
typedef struct UserMult_private *UserMult;
struct UserMult_private {
  MPI_Comm     comm;
  DM           dm;
  Vec          Xloc, Yloc;
  CeedVector   Xceed, Yceed;
  CeedOperator op;
  Ceed         ceed;
  PetscScalar  loadIncrement;
};

// Data for Jacobian setup routine
typedef struct FormJacobCtx_private *FormJacobCtx;
struct FormJacobCtx_private {
  UserMult     *jacobCtx;
  PetscInt     numLevels;
  SNES         snesCoarse;
  Mat          *jacobMat, jacobMatCoarse;
  Vec          Ucoarse;
};

// Data for PETSc Prolongation/Restriction Matshell
typedef struct UserMultProlongRestr_private *UserMultProlongRestr;
struct UserMultProlongRestr_private {
  MPI_Comm     comm;
  DM           dmC, dmF;
  Vec          locVecC, locVecF, multVec;
  CeedVector   ceedVecC, ceedVecF;
  CeedOperator opProlong, opRestrict;
  Ceed         ceed;
};

// libCEED data struct for level
typedef struct CeedData_private *CeedData;
struct CeedData_private {
  Ceed                ceed;
  CeedBasis           basisx, basisu, basisCtoF;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictqdi, ErestrictGradui;
  CeedQFunction       qfApply, qfJacob;
  CeedOperator        opApply, opJacob, opRestrict, opProlong;
  CeedVector          qdata, gradu, xceed, yceed, truesoln;
};

// -----------------------------------------------------------------------------
// Process command line options
// -----------------------------------------------------------------------------
// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx appCtx);

// Process physics options
PetscErrorCode ProcessPhysics(MPI_Comm comm, Physics phys, Units units);

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
PetscErrorCode CreateBCLabel(DM dm, const char name[]);

// Create FE by degree
PetscErrorCode PetscFECreateByDegree(DM dm, PetscInt dim, PetscInt Nc,
                                     PetscBool isSimplex, const char prefix[],
                                     PetscInt order, PetscFE *fem);

// Read mesh and distribute DM in parallel
PetscErrorCode CreateDistributedDM(MPI_Comm comm, AppCtx appCtx, DM *dm);

// Setup DM with FE space of appropriate degree
PetscErrorCode SetupDMByDegree(DM dm, AppCtx appCtx, PetscInt order,
                               PetscInt ncompu);

// -----------------------------------------------------------------------------
// libCEED Functions
// -----------------------------------------------------------------------------
// Destroy libCEED objects
PetscErrorCode CeedDataDestroy(CeedInt level, CeedData data);

// Get libCEED restriction data from DMPlex
PetscErrorCode CreateRestrictionPlex(Ceed ceed, CeedInterlaceMode imode,
                                     CeedInt P, CeedInt ncomp,
                                     CeedElemRestriction *Erestrict, DM dm);

// Set up libCEED for a given degree
PetscErrorCode SetupLibceedFineLevel(DM dm, Ceed ceed, AppCtx appCtx,
                                     Physics phys, CeedData *data,
                                     PetscInt fineLevel, PetscInt ncompu,
                                     PetscInt Ugsz, PetscInt Ulocsz,
                                     CeedVector forceCeed,
                                     CeedQFunction qfRestrict,
                                     CeedQFunction qfProlong);

// Set up libCEED for a given degree
PetscErrorCode SetupLibceedLevel(DM dm, Ceed ceed, AppCtx appCtx, Physics phys,
                                 CeedData *data, PetscInt level,
                                 PetscInt ncompu, PetscInt Ugsz,
                                 PetscInt Ulocsz, CeedVector forceCeed,
                                 CeedQFunction qfRestrict,
                                 CeedQFunction qfProlong);

// Setup context data for Jacobian evaluation
PetscErrorCode SetupJacobianCtx(MPI_Comm comm, AppCtx appCtx, DM dm, Vec V,
                                Vec Vloc, CeedData ceedData, Ceed ceed,
                                UserMult jacobianCtx);

// Setup context data for prolongation and restriction operators
PetscErrorCode SetupProlongRestrictCtx(MPI_Comm comm, DM dmC, DM dmF, Vec VF,
                                       Vec VlocC, Vec VlocF, CeedData ceedDataC,
                                       CeedData ceedDataF, Ceed ceed,
                                       UserMultProlongRestr prolongRestrCtx);

// -----------------------------------------------------------------------------
// Jacobian setup
// -----------------------------------------------------------------------------
PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat Jpre, void *ctx);

// -----------------------------------------------------------------------------
// SNES Monitor
// -----------------------------------------------------------------------------
PetscErrorCode ViewSolution(MPI_Comm comm, Vec U, PetscInt increment,
                            PetscScalar loadIncrement);

// -----------------------------------------------------------------------------
// libCEED Operators for MatShell
// -----------------------------------------------------------------------------
// This function uses libCEED to compute the local action of an operator
PetscErrorCode ApplyLocalCeedOp(Vec X, Vec Y, UserMult user);

// This function uses libCEED to compute the non-linear residual
PetscErrorCode FormResidual_Ceed(SNES snes, Vec X, Vec Y, void *ctx);

// This function uses libCEED to apply the Jacobian for assembly via a SNES
PetscErrorCode ApplyJacobianCoarse_Ceed(SNES snes, Vec X, Vec Y, void *ctx);

// This function uses libCEED to compute the action of the Jacobian
PetscErrorCode ApplyJacobian_Ceed(Mat A, Vec X, Vec Y);

// This function uses libCEED to compute the action of the prolongation operator
PetscErrorCode Prolong_Ceed(Mat A, Vec X, Vec Y);

// This function uses libCEED to compute the action of the restriction operator
PetscErrorCode Restrict_Ceed(Mat A, Vec X, Vec Y);
// This function returns the computed diagonal of the operator
PetscErrorCode GetDiag_Ceed(Mat A, Vec D);

// -----------------------------------------------------------------------------
// Boundary Functions
// -----------------------------------------------------------------------------
// Note: If additional boundary conditions are added, an update is needed in
//         elasticity.h for the boundaryOptions variable.

// BCMMS - boundary function
// Values on all points of the mesh is set based on given solution below
// for u[0], u[1], u[2]
PetscErrorCode BCMMS(PetscInt dim, PetscReal loadIncrement,
                     const PetscReal coords[], PetscInt ncompu,
                     PetscScalar *u, void *ctx);

// BCZero - fix boundary values at zero
PetscErrorCode BCZero(PetscInt dim, PetscReal loadIncrement,
                      const PetscReal coords[], PetscInt ncompu,
                      PetscScalar *u, void *ctx);

// BCClamp - fix boundary values at fraction of load increment
PetscErrorCode BCBend1_ss(PetscInt dim, PetscReal loadIncrement,
                          const PetscReal coords[], PetscInt ncompu,
                          PetscScalar *u, void *ctx);

#endif //setup_h
