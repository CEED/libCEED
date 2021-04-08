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

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>
#include <petscksp.h>
#include <stdbool.h>
#include <string.h>

#if PETSC_VERSION_LT(3,14,0)
#  define DMAddBoundary(a,b,c,d,e,f,g,h,i,j,k,l) DMAddBoundary(a,b,c,d,e,f,g,h,j,k,l)
#endif

#ifndef PHYSICS_STRUCT
#define PHYSICS_STRUCT
typedef struct Physics_private *Physics;
struct Physics_private {
  CeedScalar   nu;      // Poisson's ratio
  CeedScalar   E;       // Young's Modulus
};
#endif

// Mooney-Rivlin context
#ifndef PHYSICS_STRUCT_MR
#define PHYSICS_STRUCT_MR
typedef struct Physics_private_MR *Physics_MR;

struct Physics_private_MR { 
  //material properties for MR
  CeedScalar mu_1; // 
  CeedScalar mu_2; // 
  CeedScalar k_1; // 
};
#endif

// -----------------------------------------------------------------------------
// Generalized Polynomial context
#ifndef PHYSICS_STRUCT_GP
#define PHYSICS_STRUCT_GP
typedef struct Physics_private_GP *Physics_GP;

struct Physics_private_GP { 
  CeedScalar   nu;      // Poisson's ratio rm
  CeedScalar   E;       // Young's Modulus rm
  //material properties for GP
  CeedScalar C_mat[6][6]; // 2D matrix
  CeedScalar K[6]; // 1D array
  CeedScalar N; // max value of the sum; usually 1 or 2
};

#endif

// -----------------------------------------------------------------------------
// Command Line Options
// -----------------------------------------------------------------------------
// Problem options
typedef enum {
  ELAS_LIN = 0, ELAS_HYPER_SS = 1, ELAS_HYPER_FS = 2, ELAS_HYPER_FS_MR = 3, ELAS_HYPER_FS_GP = 4 
} problemType;
static const char *const problemTypes[] = {"linElas",
                                           "hyperSS",
                                           "hyperFS",
                                           "hyperFS-MR",
                                           "hyperFS-GP","problemType","ELAS_",0 
};
static const char *const problemTypesForDisp[] = {"Linear elasticity",
                                                  "Hyper elasticity small strain",
                                                  "Hyper elasticity finite strain",
                                                  "Hyper elasticity finite strain - Mooney Rivlin",
                                                  "Hyper elasticity finite strain - Generalized Polynomial"
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
  char          meshFile[PETSC_MAX_PATH_LEN];         // exodusII mesh file
  PetscBool     testMode;
  PetscBool     viewSoln;
  PetscBool     viewFinalSoln;
  problemType   problemChoice;
  forcingType   forcingChoice;
  multigridType multigridChoice;
  PetscScalar   nuSmoother;
  PetscInt      degree;
  PetscInt      qextra;
  PetscInt      numLevels;
  PetscInt      *levelDegrees;
  PetscInt      numIncrements;                        // Number of steps
  PetscInt      bcClampCount;
  PetscInt      bcClampFaces[16];
  PetscScalar   bcClampMax[16][7];
  PetscInt      bcTractionCount;
  PetscInt      bcTractionFaces[16];
  PetscScalar   bcTractionVector[16][3];
  PetscScalar   forcingVector[3];
};

// Problem specific data
// *INDENT-OFF*
typedef struct {
  CeedInt           qdatasize;
  CeedQFunctionUser setupgeo, apply, jacob, energy, diagnostic;
  const char        *setupgeofname, *applyfname, *jacobfname, *energyfname,
                    *diagnosticfname;
  CeedQuadMode      qmode;
} problemData;
// *INDENT-ON*

// Data specific to each problem option
extern problemData problemOptions[5];

// Forcing function data
typedef struct {
  CeedQFunctionUser setupforcing;
  const char        *setupforcingfname;
} forcingData;

extern forcingData forcingOptions[3];

// Data for PETSc Matshell
typedef struct UserMult_private *UserMult;
struct UserMult_private {
  MPI_Comm        comm;
  DM              dm;
  Vec             Xloc, Yloc, NBCs;
  CeedVector      Xceed, Yceed;
  CeedOperator    op;
  CeedQFunction   qf;
  Ceed            ceed;
  PetscScalar     loadIncrement;
  CeedQFunctionContext ctxPhys, ctxPhysSmoother;
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
  Vec          locVecC, locVecF;
  CeedVector   ceedVecC, ceedVecF;
  CeedOperator opProlong, opRestrict;
  Ceed         ceed;
};

// libCEED data struct for level
typedef struct CeedData_private *CeedData;
struct CeedData_private {
  Ceed                ceed;
  CeedBasis           basisx, basisu, basisCtoF, basisEnergy, basisDiagnostic;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictqdi,
                      ErestrictGradui, ErestrictEnergy, ErestrictDiagnostic,
                      ErestrictqdDiagnostici;
  CeedQFunction       qfApply, qfJacob, qfEnergy, qfDiagnostic;
  CeedOperator        opApply, opJacob, opRestrict, opProlong, opEnergy,
                      opDiagnostic;
  CeedVector          qdata, qdataDiagnostic, gradu, xceed, yceed, truesoln;
};

// Translate PetscMemType to CeedMemType
static inline CeedMemType MemTypeP2C(PetscMemType mtype) {
  return PetscMemTypeDevice(mtype) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}

// -----------------------------------------------------------------------------
// Process command line options
// -----------------------------------------------------------------------------
// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx appCtx);

// Process physics options
PetscErrorCode ProcessPhysics(MPI_Comm comm, Physics phys, Units units);
PetscErrorCode ProcessPhysics_MR(MPI_Comm comm, Physics_MR phys_MR, Units units);

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
                               PetscBool boundary, PetscInt ncompu);

// -----------------------------------------------------------------------------
// libCEED Functions
// -----------------------------------------------------------------------------
// Destroy libCEED objects
PetscErrorCode CeedDataDestroy(CeedInt level, CeedData data);

// Utility function - essential BC dofs are encoded in closure indices as -(i+1)
PetscInt Involute(PetscInt i);

// Utility function to create local CEED restriction from DMPlex
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt height, DMLabel domainLabel, CeedInt value,
    CeedElemRestriction *Erestrict);

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height,
                                       DMLabel domainLabel, PetscInt value, CeedInt P, CeedInt Q, CeedInt qdatasize,
                                       CeedElemRestriction *restrictq, CeedElemRestriction *restrictx,
                                       CeedElemRestriction *restrictqdi);

// Set up libCEED for a given degree
PetscErrorCode SetupLibceedFineLevel(DM dm, DM dmEnergy, DM dmDiagnostic,
                                     Ceed ceed, AppCtx appCtx,
                                     CeedQFunctionContext physCtx,
                                     CeedData *data, PetscInt fineLevel,
                                     PetscInt ncompu, PetscInt Ugsz,
                                     PetscInt Ulocsz, CeedVector forceCeed,
                                     CeedVector neumannCeed);

// Set up libCEED multigrid level for a given degree
PetscErrorCode SetupLibceedLevel(DM dm, Ceed ceed, AppCtx appCtx,
                                 CeedData *data, PetscInt level,
                                 PetscInt ncompu, PetscInt Ugsz,
                                 PetscInt Ulocsz, CeedVector fineMult);

// Setup context data for Jacobian evaluation
PetscErrorCode SetupJacobianCtx(MPI_Comm comm, AppCtx appCtx, DM dm, Vec V,
                                Vec Vloc, CeedData ceedData, Ceed ceed,
                                CeedQFunctionContext ctxPhys,
                                CeedQFunctionContext ctxPhysSmoother,
                                UserMult jacobianCtx);

// Setup context data for prolongation and restriction operators
PetscErrorCode SetupProlongRestrictCtx(MPI_Comm comm, AppCtx appCtx, DM dmC,
                                       DM dmF, Vec VF, Vec VlocC, Vec VlocF,
                                       CeedData ceedDataC, CeedData ceedDataF,
                                       Ceed ceed,
                                       UserMultProlongRestr prolongRestrCtx);

// -----------------------------------------------------------------------------
// Jacobian setup
// -----------------------------------------------------------------------------
PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat Jpre, void *ctx);

// -----------------------------------------------------------------------------
// Solution output
// -----------------------------------------------------------------------------
PetscErrorCode ViewSolution(MPI_Comm comm, Vec U, PetscInt increment,
                            PetscScalar loadIncrement);

PetscErrorCode ViewDiagnosticQuantities(MPI_Comm comm, DM dmU,
                                        UserMult user, Vec U,
                                        CeedElemRestriction ErestrictDiagnostic);

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

// This function calculates the strain energy in the final solution
PetscErrorCode ComputeStrainEnergy(DM dmEnergy, UserMult user,
                                   CeedOperator opEnergy, Vec X,
                                   PetscReal *energy);

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

// BCClamp - fix boundary values with affine transformation at fraction of load
//   increment
PetscErrorCode BCClamp(PetscInt dim, PetscReal loadIncrement,
                       const PetscReal coords[], PetscInt ncompu,
                       PetscScalar *u, void *ctx);

#endif //setup_h
