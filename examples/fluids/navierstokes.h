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
  ADVECTION_WIND_ROTATION    = 0,
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
  STAB_SU   = 1,   // Streamline Upwind
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
// Structs declarations
typedef struct User_private     *User;
typedef struct Units_private    *Units;
typedef struct AppCtx_private   *AppCtx;
typedef struct Physics_private  *Physics;
typedef struct SimpleBC_private *SimpleBC;
typedef struct CeedData_private *CeedData;

// Boundary conditions
struct SimpleBC_private {
  PetscInt      nwall, nslip[3];
  PetscInt      walls[6], slips[3][6];
  PetscBool     userbc;
};

// Problem specific data
// *INDENT-OFF*
typedef struct {
  CeedInt           dim, qdatasizeVol, qdatasizeSur;
  CeedQFunctionUser setupVol, setupSur, ics, applyVol_rhs, applyVol_ifunction,
                    applySur;
  const char        *setupVol_loc, *setupSur_loc, *ics_loc,
                    *applyVol_rhs_loc, *applyVol_ifunction_loc, *applySur_loc;
  bool              non_zero_time;
  PetscErrorCode    (*bc)(PetscInt, PetscReal, const PetscReal[], PetscInt,
                          PetscScalar[], void *);
  PetscErrorCode    (*bc_fnc)(DM, SimpleBC, Physics, void *);
} problemData;
// *INDENT-ON*

// PETSc user data
struct User_private {
  MPI_Comm     comm;
  DM           dm;
  DM           dmviz;
  Mat          interpviz;
  Ceed         ceed;
  Units        units;
  Vec          M;
  Physics      phys;
  AppCtx       app_ctx;
  CeedVector   qceed, qdotceed, gceed;
  CeedOperator op_rhs_vol, op_rhs, op_ifunction_vol, op_ifunction;
};

// Units
struct Units_private {
  // fundamental units
  PetscScalar meter;
  PetscScalar kilogram;
  PetscScalar second;
  PetscScalar Kelvin;
  // derived units
  PetscScalar Pascal;
  PetscScalar J_per_kg_K;
  PetscScalar m_per_squared_s;
  PetscScalar W_per_m_K;
  PetscScalar Joule;
};

// Setup Context for QFunctions
struct Physics_private {
  DCContext         dc_ctx_data;
  EulerContext      euler_ctx_data;
  AdvectionContext  advection_ctx_data;
  WindType          wind_type;
  EulerTestType     euler_test;
  StabilizationType stab;
  PetscBool         implicit;
  PetscBool         has_current_time;
  PetscBool         has_neumann;
};

// Application context from user command line options
struct AppCtx_private {
  // libCEED arguments
  char              ceed_resource[PETSC_MAX_PATH_LEN];     // libCEED backend
  PetscInt          degree;                                // todo: degree_sur
  PetscInt          q_extra;
  PetscInt          q_extra_sur;
  // Post-processing arguments
  PetscInt          output_freq;
  PetscInt          viz_refine;
  PetscInt          cont_steps;
  char              output_dir[PETSC_MAX_PATH_LEN];
  // Problem type arguments
  PetscFunctionList problems;
  char              problem_name[PETSC_MAX_PATH_LEN];
  // Test mode arguments
  PetscBool         test_mode;
  PetscScalar       test_tol;
  char              file_path[PETSC_MAX_PATH_LEN];
};

// libCEED data struct
struct CeedData_private {
  CeedBasis            basisx, basisxc, basisq, basisxSur, basisxcSur, basisqSur;
  CeedElemRestriction  restrictx, restrictq, restrictqdi;
  CeedQFunction        qf_setupVol, qf_ics, qf_rhsVol, qf_ifunctionVol,
                       qf_setupSur, qf_applySur;
  CeedOperator         op_setupVol, op_ics;
  CeedVector           xcorners, qdata, q0ceed;
  CeedQFunctionContext ctxSetup, ctxNS, ctxAdvection, ctxEuler;
};

// -----------------------------------------------------------------------------
// Set up problems
// -----------------------------------------------------------------------------
// Set up function for each problem
extern PetscErrorCode NS_DENSITY_CURRENT(problemData *problem,
    void *ctxSetupData, void *ctx, void *ctxPhys);

extern PetscErrorCode NS_EULER_VORTEX(problemData *problem,
                                      void *ctxSetupData, void *ctx, void *ctxPhys);

extern PetscErrorCode NS_ADVECTION(problemData *problem,
                                   void *ctxSetupData, void *ctx, void *ctxPhys);

extern PetscErrorCode NS_ADVECTION2D(problemData *problem,
                                     void *ctxSetupData, void *ctx, void *ctxPhys);

// Boundary condition function for each problem
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

PetscErrorCode SetupLibceed(Ceed ceed, CeedData ceed_data, DM dm, User user,
                            AppCtx app_ctx, problemData *problem, SimpleBC bc);

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
// Read mesh and distribute DM in parallel
PetscErrorCode CreateDistributedDM(MPI_Comm comm, problemData *problem,
                                   SetupContext setup_ctx, DM *dm);

// Set up DM
PetscErrorCode SetUpDM(DM dm, problemData *problem, PetscInt degree,
                       SimpleBC bc, Physics phys, void *ctxSetupData);

// Refine DM for high-order viz
PetscErrorCode VizRefineDM(DM dm, User user, problemData *problem,
                           SimpleBC bc, Physics phys, void *ctxSetupData);

// -----------------------------------------------------------------------------
// Process command line options
// -----------------------------------------------------------------------------
// Register problems to be available on the command line
PetscErrorCode RegisterProblems_NS(AppCtx app_ctx);

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx);

// -----------------------------------------------------------------------------
// Miscellaneous utility functions
// -----------------------------------------------------------------------------
int VectorPlacePetscVec(CeedVector c, Vec p);

PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm,
    PetscBool insertEssential, Vec Qloc, PetscReal time, Vec faceGeomFVM,
    Vec cellGeomFVM, Vec gradFVM);

// -----------------------------------------------------------------------------
#endif
