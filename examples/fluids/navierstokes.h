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
static const char *const MemTypes[] = {
  "host",
  "device",
  "MemType", "CEED_MEM_", NULL
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
  PetscInt      num_wall, num_slip[3];
  PetscInt      walls[6], slips[3][6];
  PetscBool     user_bc;
};

// Problem specific data
// *INDENT-OFF*
typedef struct {
  CeedInt           dim, q_data_size_vol, q_data_size_sur;
  CeedQFunctionUser setup_vol, setup_sur, ics, apply_vol_rhs, apply_vol_ifunction,
                    apply_sur;
  const char        *setup_vol_loc, *setup_sur_loc, *ics_loc,
                    *apply_vol_rhs_loc, *apply_vol_ifunction_loc, *apply_sur_loc;
  bool              non_zero_time;
  PetscErrorCode    (*bc)(PetscInt, PetscReal, const PetscReal[], PetscInt,
                          PetscScalar[], void *);
  PetscErrorCode    (*bc_fnc)(DM, SimpleBC, Physics, void *);
} ProblemData;
// *INDENT-ON*

// PETSc user data
struct User_private {
  MPI_Comm     comm;
  DM           dm;
  DM           dm_viz;
  Mat          interp_viz;
  Ceed         ceed;
  Units        units;
  Vec          M;
  Physics      phys;
  AppCtx       app_ctx;
  CeedVector   q_ceed, q_dot_ceed, g_ceed;
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
  DCContext         dc_ctx;
  EulerContext      euler_ctx;
  AdvectionContext  advection_ctx;
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
  CeedVector           x_corners, q_data, q0_ceed;
  CeedQFunctionContext setup_context, dc_context, advection_context,
                       euler_context;
  CeedQFunction        qf_setup_vol, qf_ics, qf_rhs_vol, qf_ifunction_vol,
                       qf_setup_sur, qf_apply_sur;
  CeedBasis            basis_x, basis_xc, basis_q, basis_x_sur, basis_xc_sur,
                       basis_q_sur;
  CeedElemRestriction  elem_restr_x, elem_restr_q, elem_restr_qd_i;
  CeedOperator         op_setup_vol, op_ics;
};

// -----------------------------------------------------------------------------
// Set up problems
// -----------------------------------------------------------------------------
// Set up function for each problem
extern PetscErrorCode NS_DENSITY_CURRENT(ProblemData *problem,
    void *setup_ctx, void *ctx, void *phys);

extern PetscErrorCode NS_EULER_VORTEX(ProblemData *problem,
                                      void *setup_ctx, void *ctx, void *phys);

extern PetscErrorCode NS_ADVECTION(ProblemData *problem,
                                   void *setup_ctx, void *ctx, void *phys);

extern PetscErrorCode NS_ADVECTION2D(ProblemData *problem,
                                     void *setup_ctx, void *ctx, void *phys);

// Boundary condition function for each problem
extern PetscErrorCode BC_DENSITY_CURRENT(DM dm, SimpleBC bc, Physics phys,
    void *setup_ctx);

extern PetscErrorCode BC_EULER_VORTEX(DM dm, SimpleBC bc, Physics phys,
                                      void *setup_ctx);

extern PetscErrorCode BC_ADVECTION(DM dm, SimpleBC bc, Physics phys,
                                   void *setup_ctx);

extern PetscErrorCode BC_ADVECTION2D(DM dm, SimpleBC bc, Physics phys,
                                     void *setup_ctx);

// -----------------------------------------------------------------------------
// libCEED functions
// -----------------------------------------------------------------------------
// Utility function - essential BC dofs are encoded in closure indices as -(i+1).
PetscInt Involute(PetscInt i);

// Utility function to create local CEED restriction
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt height, DMLabel domain_label,
    CeedInt value, CeedElemRestriction *elem_restr);

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height,
                                       DMLabel domain_label, PetscInt value,
                                       CeedInt P, CeedInt Q, CeedInt q_data_size,
                                       CeedElemRestriction *elem_restr_q,
                                       CeedElemRestriction *elem_restr_x,
                                       CeedElemRestriction *elem_restr_qd_i);

// Utility function to create CEED Composite Operator for the entire domain
PetscErrorCode CreateOperatorForDomain(Ceed ceed, DM dm, SimpleBC bc,
                                       CeedData ceed_data, Physics phys,
                                       CeedOperator op_apply_vol, CeedInt height,
                                       CeedInt P_sur, CeedInt Q_sur, CeedInt q_data_size_sur,
                                       CeedOperator *op_apply);

PetscErrorCode SetupLibceed(Ceed ceed, CeedData ceed_data, DM dm, User user,
                            AppCtx app_ctx, ProblemData *problem, SimpleBC bc);

// Set up contex for QFunctions
PetscErrorCode SetupContextForProblems(Ceed ceed, CeedData ceed_data,
                                       AppCtx app_ctx, SetupContext setup_ctx, Physics phys);

// -----------------------------------------------------------------------------
// Time-stepping functions
// -----------------------------------------------------------------------------
PetscErrorCode ComputeLumpedMassMatrix(Ceed ceed, DM dm,
                                       CeedElemRestriction elem_restr_q, CeedBasis basis_q,
                                       CeedElemRestriction elem_restr_qd_i, CeedVector q_data, Vec M);

// RHS (Explicit time-stepper) function setup
//   This is the RHS of the ODE, given as u_t = G(t,u)
//   This function takes in a state vector Q and writes into G
PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *user_data);

// Implicit time-stepper function setup
PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, Vec G,
                            void *user_data);

// User provided TS Monitor
PetscErrorCode TSMonitor_NS(TS ts, PetscInt step_no, PetscReal time, Vec Q,
                            void *ctx);

// TS: Create, setup, and solve
PetscErrorCode TSSolve_NS(DM dm, User user, AppCtx app_ctx, Physics phys,
                          Vec *Q, PetscScalar *f_time, TS *ts);

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
// Read mesh and distribute DM in parallel
PetscErrorCode CreateDistributedDM(MPI_Comm comm, ProblemData *problem,
                                   SetupContext setup_ctx, DM *dm);

// Set up DM
PetscErrorCode SetUpDM(DM dm, ProblemData *problem, PetscInt degree,
                       SimpleBC bc, Physics phys, void *setup_ctx);

// Refine DM for high-order viz
PetscErrorCode VizRefineDM(DM dm, User user, ProblemData *problem,
                           SimpleBC bc, Physics phys, void *setup_ctx);

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
PetscErrorCode ICs_FixMultiplicity(CeedOperator op_ics, CeedVector x_corners,
                                   CeedVector q0_ceed, DM dm, Vec Q_loc, Vec Q,
                                   CeedElemRestriction elem_restr_q,
                                   CeedQFunctionContext setup_context, CeedScalar time);

PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm,
    PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM,
    Vec cell_geom_FVM, Vec grad_FVM);

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTests_NS(AppCtx app_ctx, Vec Q);

// Get error for problems with exact solutions
PetscErrorCode GetError_NS(CeedData ceed_data, DM dm, AppCtx app_ctx, Vec Q,
                           PetscScalar final_time);

// Post-processing
PetscErrorCode PostProcess_NS(TS ts, CeedData ceed_data, DM dm,
                              ProblemData *problem, AppCtx app_ctx,
                              Vec Q, PetscScalar final_time);

// -- Gather initial Q values in case of continuation of simulation
PetscErrorCode SetupICsFromBinary(MPI_Comm comm, AppCtx app_ctx, Vec Q);

// Record boundary values from initial condition
PetscErrorCode SetBCsFromICs_NS(DM dm, Vec Q, Vec Q_loc);

// -----------------------------------------------------------------------------
#endif
