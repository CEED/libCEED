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

#ifndef navierstokes_h
#define navierstokes_h

#include <ceed.h>
#include <petscdm.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <petscts.h>
#include <stdbool.h>

// -----------------------------------------------------------------------------
// PETSc Macros
// -----------------------------------------------------------------------------
#if PETSC_VERSION_LT(3,14,0)
#  define DMPlexGetClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexGetClosureIndices(a,b,c,d,f,g,i)
#  define DMPlexRestoreClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexRestoreClosureIndices(a,b,c,d,f,g,i)
#endif

#if PETSC_VERSION_LT(3,14,0)
#  define DMAddBoundary(a,b,c,d,e,f,g,h,i,j,k,l,m,n) DMAddBoundary(a,b,c,e,h,i,j,k,f,g,m)
#elif PETSC_VERSION_LT(3,16,0)
#  define DMAddBoundary(a,b,c,d,e,f,g,h,i,j,k,l,m,n) DMAddBoundary(a,b,c,e,h,i,j,k,l,f,g,m)
#else
#  define DMAddBoundary(a,b,c,d,e,f,g,h,i,j,k,l,m,n) DMAddBoundary(a,b,c,d,f,g,h,i,j,k,l,m,n)
#endif

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------
// Translate PetscMemType to CeedMemType
static inline CeedMemType MemTypeP2C(PetscMemType mem_type) {
  return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}

// Advection - Wind Options
typedef enum {
  WIND_ROTATION    = 0,
  WIND_TRANSLATION = 1,
} WindType;
static const char *const WindTypes[] = {
  "rotation",
  "translation",
  "WindType", "WIND_", NULL
};

// Advection - Bubble Types
typedef enum {
  BUBBLE_SPHERE   = 0, // dim=3
  BUBBLE_CYLINDER = 1, // dim=2
} BubbleType;
static const char *const BubbleTypes[] = {
  "sphere",
  "cylinder",
  "BubbleType", "BUBBLE_", NULL
};

// Advection - Bubble Continuity Types
typedef enum {
  BUBBLE_CONTINUITY_SMOOTH     = 0,  // Original continuous, smooth shape
  BUBBLE_CONTINUITY_BACK_SHARP = 1,  // Discontinuous, sharp back half shape
  BUBBLE_CONTINUITY_THICK      = 2,  // Define a finite thickness
} BubbleContinuityType;
static const char *const BubbleContinuityTypes[] = {
  "smooth",
  "back_sharp",
  "thick",
  "BubbleContinuityType", "BUBBLE_CONTINUITY_", NULL
};

// Euler - test cases
typedef enum {
  EULER_TEST_NONE = 0,
  EULER_TEST_1    = 1,
  EULER_TEST_2    = 2,
  EULER_TEST_3    = 3,
  EULER_TEST_4    = 4,
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
  STAB_SU   = 1, // Streamline Upwind
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
typedef struct AppCtx_private    *AppCtx;
typedef struct CeedData_private  *CeedData;
typedef struct User_private      *User;
typedef struct Units_private     *Units;
typedef struct SimpleBC_private  *SimpleBC;
typedef struct SetupContext_     *SetupContext;
typedef struct Physics_private   *Physics;

// Application context from user command line options
struct AppCtx_private {
  // libCEED arguments
  char              ceed_resource[PETSC_MAX_PATH_LEN]; // libCEED backend
  PetscInt          degree;
  PetscInt          q_extra;
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
  CeedVector           x_coord, q_data;
  CeedQFunctionContext setup_context, dc_context, advection_context,
                       euler_context;
  CeedQFunction        qf_setup_vol, qf_ics, qf_rhs_vol, qf_ifunction_vol,
                       qf_setup_sur, qf_apply_sur;
  CeedBasis            basis_x, basis_xc, basis_q, basis_x_sur, basis_xc_sur,
                       basis_q_sur;
  CeedElemRestriction  elem_restr_x, elem_restr_q, elem_restr_qd_i;
  CeedOperator         op_setup_vol, op_ics;
};

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

// Boundary conditions
struct SimpleBC_private {
  PetscInt  num_wall, num_slip[3];
  PetscInt  walls[6], slips[3][6];
  PetscBool user_bc;
};

// Initial conditions
#ifndef setup_context_struct
#define setup_context_struct
typedef struct SetupContext_ *SetupContext;
struct SetupContext_ {
  CeedScalar theta0;
  CeedScalar thetaC;
  CeedScalar P0;
  CeedScalar N;
  CeedScalar cv;
  CeedScalar cp;
  CeedScalar Rd;
  CeedScalar g;
  CeedScalar rc;
  CeedScalar lx;
  CeedScalar ly;
  CeedScalar lz;
  CeedScalar center[3];
  CeedScalar dc_axis[3];
  CeedScalar wind[3];
  CeedScalar time;
  int wind_type;              // See WindType: 0=ROTATION, 1=TRANSLATION
  int bubble_type;            // See BubbleType: 0=SPHERE, 1=CYLINDER
  int bubble_continuity_type; // See BubbleContinuityType: 0=SMOOTH, 1=BACK_SHARP 2=THICK
};
#endif

// DENSITY_CURRENT
#ifndef dc_context_struct
#define dc_context_struct
typedef struct DCContext_ *DCContext;
struct DCContext_ {
  CeedScalar lambda;
  CeedScalar mu;
  CeedScalar k;
  CeedScalar cv;
  CeedScalar cp;
  CeedScalar g;
  CeedScalar Rd;
  int stabilization; // See StabilizationType: 0=none, 1=SU, 2=SUPG
};
#endif

// EULER_VORTEX
#ifndef euler_context_struct
#define euler_context_struct
typedef struct EulerContext_ *EulerContext;
struct EulerContext_ {
  CeedScalar center[3];
  CeedScalar curr_time;
  CeedScalar vortex_strength;
  CeedScalar mean_velocity[3];
  int euler_test;
  bool implicit;
};
#endif

// ADVECTION and ADVECTION2D
#ifndef advection_context_struct
#define advection_context_struct
typedef struct AdvectionContext_ *AdvectionContext;
struct AdvectionContext_ {
  CeedScalar CtauS;
  CeedScalar strong_form;
  CeedScalar E_wind;
  bool implicit;
  int stabilization; // See StabilizationType: 0=none, 1=SU, 2=SUPG
};
#endif

// Struct that contains all enums and structs used for the physics of all problems
struct Physics_private {
  DCContext            dc_ctx;
  EulerContext         euler_ctx;
  AdvectionContext     advection_ctx;
  WindType             wind_type;
  BubbleType           bubble_type;
  BubbleContinuityType bubble_continuity_type;
  EulerTestType        euler_test;
  StabilizationType    stab;
  PetscBool            implicit;
  PetscBool            has_curr_time;
  PetscBool            has_neumann;
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
  PetscErrorCode    (*bc_func)(DM, SimpleBC, Physics, void *);
  PetscErrorCode    (*print_info)(Physics, SetupContext, AppCtx);
} ProblemData;
// *INDENT-ON*

// -----------------------------------------------------------------------------
// Set up problems
// -----------------------------------------------------------------------------
// Set up function for each problem
extern PetscErrorCode NS_DENSITY_CURRENT(ProblemData *problem, void *setup_ctx,
    void *ctx);

extern PetscErrorCode NS_EULER_VORTEX(ProblemData *problem, void *setup_ctx,
                                      void *ctx);

extern PetscErrorCode NS_ADVECTION(ProblemData *problem, void *setup_ctx,
                                   void *ctx);

extern PetscErrorCode NS_ADVECTION2D(ProblemData *problem, void *setup_ctx,
                                     void *ctx);

// Boundary condition function for each problem
extern PetscErrorCode BC_DENSITY_CURRENT(DM dm, SimpleBC bc, Physics phys,
    void *setup_ctx);

extern PetscErrorCode BC_EULER_VORTEX(DM dm, SimpleBC bc, Physics phys,
                                      void *setup_ctx);

extern PetscErrorCode BC_ADVECTION(DM dm, SimpleBC bc, Physics phys,
                                   void *setup_ctx);

extern PetscErrorCode BC_ADVECTION2D(DM dm, SimpleBC bc, Physics phys,
                                     void *setup_ctx);

// Print function for each problem
extern PetscErrorCode PRINT_DENSITY_CURRENT(Physics phys,
    SetupContext setup_ctx, AppCtx app_ctx);

extern PetscErrorCode PRINT_EULER_VORTEX(Physics phys, SetupContext setup_ctx,
    AppCtx app_ctx);

extern PetscErrorCode PRINT_ADVECTION(Physics phys, SetupContext setup_ctx,
                                      AppCtx app_ctx);

extern PetscErrorCode PRINT_ADVECTION2D(Physics phys, SetupContext setup_ctx,
                                        AppCtx app_ctx);

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
// Compute mass matrix for explicit scheme
PetscErrorCode ComputeLumpedMassMatrix(Ceed ceed, DM dm, CeedData ceed_data,
                                       Vec M);

// RHS (Explicit time-stepper) function setup
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
PetscErrorCode ICs_FixMultiplicity(DM dm, CeedData ceed_data, Vec Q_loc, Vec Q,
                                   CeedScalar time);

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
