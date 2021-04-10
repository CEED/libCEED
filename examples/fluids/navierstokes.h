#ifndef navierstokes_h
#define navierstokes_h

#include <ceed.h>
#include <petscdm.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <petscts.h>
#include <stdbool.h>

#include "qfunctions/common.h"
#include "qfunctions/setupboundary.h"
#include "qfunctions/advection.h"
#include "qfunctions/advection2d.h"
#include "qfunctions/eulervortex.h"
#include "qfunctions/densitycurrent.h"

// -----------------------------------------------------------------------------
// PETSc Macros
// -----------------------------------------------------------------------------

#if PETSC_VERSION_LT(3,14,0)
#  define DMPlexGetClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexGetClosureIndices(a,b,c,d,f,g,i)
#  define DMPlexRestoreClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexRestoreClosureIndices(a,b,c,d,f,g,i)
#endif

#if PETSC_VERSION_LT(3,14,0)
#  define DMAddBoundary(a,b,c,d,e,f,g,h,i,j,k,l) DMAddBoundary(a,b,c,d,e,f,g,h,j,k,l)
#endif

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------

// MemType Options
static const char *const memTypes[] = {
  "host",
  "device",
  "memType", "CEED_MEM_", NULL
};

// Advection - Wind Options
typedef enum {
  ADVECTION_WIND_ROTATION = 0,
  ADVECTION_WIND_TRANSLATION = 1,
} WindType;
static const char *const WindTypes[] = {
  "rotation",
  "translation",
  "WindType", "ADVECTION_WIND_", NULL
};

// Euler - test cases
typedef enum {
  EULER_TEST_NONE = 0,
  EULER_TEST_1 = 1,
  EULER_TEST_2 = 2,
  EULER_TEST_3 = 3,
  EULER_TEST_4 = 4,
} EulerTestType;
static const char *const EulerTestTypes[] = {
  "none",
  "t1",
  "t2",
  "t3",
  "t4",
  "EulerTestType", "EULER_TEST_", NULL
};

// Stabilization methods
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

// -----------------------------------------------------------------------------
// Structs
// -----------------------------------------------------------------------------

typedef struct User_ *User;
typedef struct Units_ *Units;
typedef struct AppCtx_ *AppCtx;
typedef struct Physics_ *Physics;
typedef struct SimpleBC_ *SimpleBC;

// Boundary conditions
struct SimpleBC_ {
  PetscInt nwall, nslip[3];
  PetscInt walls[6], slips[3][6];
  PetscBool userbc;
};

// Problem specific data
typedef struct {
  CeedInt dim, qdatasizeVol, qdatasizeSur;
  CeedQFunctionUser setupVol, setupSur, ics, applyVol_rhs, applyVol_ifunction,
                    applySur;
  const char *setupVol_loc, *setupSur_loc, *ics_loc, *applyVol_rhs_loc,
        *applyVol_ifunction_loc, *applySur_loc;
  bool non_zero_time;
  PetscErrorCode (*bc)(PetscInt, PetscReal, const PetscReal[], PetscInt,
                       PetscScalar[], void *);
  PetscErrorCode (*bc_fnc)(DM, SimpleBC, Physics, void *);
} problemData;

// PETSc user data
struct User_ {
  MPI_Comm comm;
  DM dm;
  DM dmviz;
  Mat interpviz;
  Ceed ceed;
  Units units;
  CeedVector qceed, qdotceed, gceed;
  CeedOperator op_rhs_vol, op_rhs, op_ifunction_vol, op_ifunction;
  Vec M;
  Physics phys;
  AppCtx app_ctx;
};

// Units
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

// Setup Context for QFunctions
struct Physics_ {
  NSContext ctxNSData;
  EulerContext ctxEulerData;
  AdvectionContext ctxAdvectionData;
  WindType wind_type;
  EulerTestType eulertest;
  StabilizationType stab;
  PetscBool implicit;
  PetscBool hasCurrentTime;
  PetscBool hasNeumann;
};

// Application context from user command line options
struct AppCtx_ {
  char                ceed_resource[PETSC_MAX_PATH_LEN];     // libCEED backend
  char                output_dir[PETSC_MAX_PATH_LEN];
  char                problem_name[PETSC_MAX_PATH_LEN];
  PetscFunctionList   problems;
  PetscInt            output_freq;
  PetscInt            viz_refine;
  PetscInt            cont_steps;
  PetscInt            degree;
  // todo: degree_sur
  PetscInt            q_extra;
  PetscInt            q_extra_sur;
  // Test mode arguments
  PetscBool           test_mode;
  PetscScalar         test_tol;
  char                file_path[PETSC_MAX_PATH_LEN];
};

// -----------------------------------------------------------------------------
// Setup function for each problem
// -----------------------------------------------------------------------------

extern PetscErrorCode NS_DENSITY_CURRENT(problemData *problem,
    void *ctxSetupData, void *ctx, void *ctxPhys);
extern PetscErrorCode NS_EULER_VORTEX(problemData *problem,
                                      void *ctxSetupData, void *ctx, void *ctxPhys);
extern PetscErrorCode NS_ADVECTION(problemData *problem,
                                   void *ctxSetupData, void *ctx, void *ctxPhys);
extern PetscErrorCode NS_ADVECTION2D(problemData *problem,
                                     void *ctxSetupData, void *ctx, void *ctxPhys);

// -----------------------------------------------------------------------------
// Boundary condition function for each problem
// -----------------------------------------------------------------------------

extern PetscErrorCode BC_DENSITY_CURRENT(DM dm, SimpleBC bc, Physics phys,
    void *ctxSetupData);
extern PetscErrorCode BC_EULER_VORTEX(DM dm, SimpleBC bc, Physics phys,
                                      void *ctxSetupData);
extern PetscErrorCode BC_ADVECTION(DM dm, SimpleBC bc, Physics phys,
                                   void *ctxSetupData);
extern PetscErrorCode BC_ADVECTION2D(DM dm, SimpleBC bc, Physics phys,
                                     void *ctxSetupData);

// -----------------------------------------------------------------------------
// libCEED functions
// -----------------------------------------------------------------------------

// Utility function - essential BC dofs are encoded in closure indices as -(i+1).
PetscInt Involute(PetscInt i);

// Utility function to create local CEED restriction
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt height, DMLabel domainLabel,
    CeedInt value, CeedElemRestriction *Erestrict);

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height,
                                       DMLabel domainLabel, PetscInt value,
                                       CeedInt P, CeedInt Q, CeedInt qdatasize,
                                       CeedElemRestriction *restrictq,
                                       CeedElemRestriction *restrictx,
                                       CeedElemRestriction *restrictqdi);

// Utility function to create CEED Composite Operator for the entire domain
PetscErrorCode CreateOperatorForDomain(Ceed ceed, DM dm, SimpleBC bc,
                                       Physics phys, CeedOperator op_applyVol, CeedQFunction qf_applySur,
                                       CeedQFunction qf_setupSur, CeedInt height, CeedInt numP_Sur, CeedInt numQ_Sur,
                                       CeedInt qdatasizeSur, CeedInt NqptsSur, CeedBasis basisxSur,
                                       CeedBasis basisqSur, CeedOperator *op_apply);

// -----------------------------------------------------------------------------
// Time-stepping functions
// -----------------------------------------------------------------------------

PetscErrorCode ComputeLumpedMassMatrix(Ceed ceed, DM dm,
                                       CeedElemRestriction restrictq, CeedBasis basisq,
                                       CeedElemRestriction restrictqdi, CeedVector qdata, Vec M);

// RHS (Explicit time-stepper) function setup
//   This is the RHS of the ODE, given as u_t = G(t,u)
//   This function takes in a state vector Q and writes into G
PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *userData);

// Implicit time-stepper function setup
PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Qdot, Vec G,
                            void *userData);

// User provided TS Monitor
PetscErrorCode TSMonitor_NS(TS ts, PetscInt stepno, PetscReal time, Vec Q,
                            void *ctx);

PetscErrorCode ICs_FixMultiplicity(CeedOperator op_ics, CeedVector xcorners,
                                   CeedVector q0ceed, DM dm, Vec Qloc, Vec Q,
                                   CeedElemRestriction restrictq,
                                   CeedQFunctionContext ctxSetup, CeedScalar time);

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------

PetscErrorCode SetUpDM(DM dm, problemData *problem, PetscInt degree,
                       SimpleBC bc, Physics phys, void *ctxSetupData);

// -----------------------------------------------------------------------------
// Miscellaneous utility functions
// -----------------------------------------------------------------------------

int VectorPlacePetscVec(CeedVector c, Vec p);

PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm,
    PetscBool insertEssential, Vec Qloc, PetscReal time, Vec faceGeomFVM,
    Vec cellGeomFVM, Vec gradFVM);

PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx);

#endif
