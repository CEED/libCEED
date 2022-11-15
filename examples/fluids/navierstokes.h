// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef libceed_fluids_examples_navier_stokes_h
#define libceed_fluids_examples_navier_stokes_h

#include <ceed.h>
#include <petscdm.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <petscts.h>
#include <stdbool.h>

#include "qfunctions/newtonian_types.h"
#include "qfunctions/stabilization_types.h"

// -----------------------------------------------------------------------------
// PETSc Version
// -----------------------------------------------------------------------------
#if PETSC_VERSION_LT(3, 17, 0)
#error "PETSc v3.17 or later is required"
#endif

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------
// Translate PetscMemType to CeedMemType
static inline CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

// Advection - Wind Options
typedef enum {
  WIND_ROTATION    = 0,
  WIND_TRANSLATION = 1,
} WindType;
static const char *const WindTypes[] = {"rotation", "translation", "WindType", "WIND_", NULL};

// Advection - Bubble Types
typedef enum {
  BUBBLE_SPHERE   = 0,  // dim=3
  BUBBLE_CYLINDER = 1,  // dim=2
} BubbleType;
static const char *const BubbleTypes[] = {"sphere", "cylinder", "BubbleType", "BUBBLE_", NULL};

// Advection - Bubble Continuity Types
typedef enum {
  BUBBLE_CONTINUITY_SMOOTH     = 0,  // Original continuous, smooth shape
  BUBBLE_CONTINUITY_BACK_SHARP = 1,  // Discontinuous, sharp back half shape
  BUBBLE_CONTINUITY_THICK      = 2,  // Define a finite thickness
} BubbleContinuityType;
static const char *const BubbleContinuityTypes[] = {"smooth", "back_sharp", "thick", "BubbleContinuityType", "BUBBLE_CONTINUITY_", NULL};

// Euler - test cases
typedef enum {
  EULER_TEST_ISENTROPIC_VORTEX = 0,
  EULER_TEST_1                 = 1,
  EULER_TEST_2                 = 2,
  EULER_TEST_3                 = 3,
  EULER_TEST_4                 = 4,
  EULER_TEST_5                 = 5,
} EulerTestType;
static const char *const EulerTestTypes[] = {"isentropic_vortex", "test_1",      "test_2", "test_3", "test_4", "test_5",
                                             "EulerTestType",     "EULER_TEST_", NULL};

// Stabilization methods
static const char *const StabilizationTypes[] = {"none", "SU", "SUPG", "StabilizationType", "STAB_", NULL};

// -----------------------------------------------------------------------------
// Structs
// -----------------------------------------------------------------------------
// Structs declarations
typedef struct AppCtx_private   *AppCtx;
typedef struct CeedData_private *CeedData;
typedef struct User_private     *User;
typedef struct Units_private    *Units;
typedef struct SimpleBC_private *SimpleBC;
typedef struct Physics_private  *Physics;

// Application context from user command line options
struct AppCtx_private {
  // libCEED arguments
  char     ceed_resource[PETSC_MAX_PATH_LEN];  // libCEED backend
  PetscInt degree;
  PetscInt q_extra;
  // Solver arguments
  MatType   amat_type;
  PetscBool pmat_pbdiagonal;
  // Post-processing arguments
  PetscInt  output_freq;
  PetscInt  viz_refine;
  PetscInt  cont_steps;
  char      cont_file[PETSC_MAX_PATH_LEN];
  char      cont_time_file[PETSC_MAX_PATH_LEN];
  char      output_dir[PETSC_MAX_PATH_LEN];
  PetscBool add_stepnum2bin;
  // Problem type arguments
  PetscFunctionList problems;
  char              problem_name[PETSC_MAX_PATH_LEN];
  // Test mode arguments
  PetscBool   test_mode;
  PetscScalar test_tol;
  char        file_path[PETSC_MAX_PATH_LEN];
};

// libCEED data struct
struct CeedData_private {
  CeedVector    x_coord, q_data;
  CeedQFunction qf_setup_vol, qf_ics, qf_rhs_vol, qf_ifunction_vol, qf_setup_sur, qf_apply_inflow, qf_apply_inflow_jacobian, qf_apply_outflow,
      qf_apply_outflow_jacobian, qf_apply_freestream, qf_apply_freestream_jacobian;
  CeedBasis           basis_x, basis_xc, basis_q, basis_x_sur, basis_q_sur, basis_xc_sur;
  CeedElemRestriction elem_restr_x, elem_restr_q, elem_restr_qd_i;
  CeedOperator        op_setup_vol, op_ics;
};

// PETSc user data
struct User_private {
  MPI_Comm     comm;
  DM           dm;
  DM           dm_viz;
  Mat          interp_viz;
  Ceed         ceed;
  Units        units;
  Vec          M, Q_loc, Q_dot_loc;
  Physics      phys;
  AppCtx       app_ctx;
  CeedVector   q_ceed, q_dot_ceed, g_ceed, coo_values_amat, coo_values_pmat, x_ceed;
  CeedOperator op_rhs_vol, op_rhs, op_ifunction_vol, op_ifunction, op_ijacobian, op_dirichlet;
  bool         matrices_set_up;
  CeedScalar   time, dt;
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
  PetscInt num_wall,  // Number of faces with wall BCs
      wall_comps[5],  // An array of constrained component numbers
      num_comps,
      num_slip[3],  // Number of faces with slip BCs
      num_inflow, num_outflow, num_freestream;
  PetscInt  walls[16], slips[3][16], inflows[16], outflows[16], freestreams[16];
  PetscBool user_bc;
};

// Struct that contains all enums and structs used for the physics of all problems
struct Physics_private {
  WindType              wind_type;
  BubbleType            bubble_type;
  BubbleContinuityType  bubble_continuity_type;
  EulerTestType         euler_test;
  StabilizationType     stab;
  PetscBool             implicit;
  StateVariable         state_var;
  PetscBool             has_curr_time;
  PetscBool             has_neumann;
  CeedContextFieldLabel solution_time_label;
  CeedContextFieldLabel stg_solution_time_label;
  CeedContextFieldLabel timestep_size_label;
  CeedContextFieldLabel ics_time_label;
  CeedContextFieldLabel ijacobian_time_shift_label;
};

typedef struct {
  CeedQFunctionUser    qfunction;
  const char          *qfunction_loc;
  CeedQFunctionContext qfunction_context;
} ProblemQFunctionSpec;

// Problem specific data
// *INDENT-OFF*
typedef struct ProblemData_private ProblemData;
struct ProblemData_private {
  CeedInt              dim, q_data_size_vol, q_data_size_sur, jac_data_size_sur;
  CeedScalar           dm_scale;
  ProblemQFunctionSpec setup_vol, setup_sur, ics, apply_vol_rhs, apply_vol_ifunction, apply_vol_ijacobian, apply_inflow, apply_outflow,
      apply_freestream, apply_inflow_jacobian, apply_outflow_jacobian, apply_freestream_jacobian;
  bool non_zero_time;
  PetscErrorCode (*bc)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
  void     *bc_ctx;
  PetscBool bc_from_ics, use_dirichlet_ceed;
  PetscErrorCode (*print_info)(ProblemData *, AppCtx);
};
// *INDENT-ON*

extern int FreeContextPetsc(void *);

// -----------------------------------------------------------------------------
// Set up problems
// -----------------------------------------------------------------------------
// Set up function for each problem
extern PetscErrorCode NS_NEWTONIAN_WAVE(ProblemData *problem, DM dm, void *ctx);
extern PetscErrorCode NS_CHANNEL(ProblemData *problem, DM dm, void *ctx);
extern PetscErrorCode NS_BLASIUS(ProblemData *problem, DM dm, void *ctx);
extern PetscErrorCode NS_NEWTONIAN_IG(ProblemData *problem, DM dm, void *ctx);
extern PetscErrorCode NS_DENSITY_CURRENT(ProblemData *problem, DM dm, void *ctx);

extern PetscErrorCode NS_EULER_VORTEX(ProblemData *problem, DM dm, void *ctx);
extern PetscErrorCode NS_SHOCKTUBE(ProblemData *problem, DM dm, void *ctx);
extern PetscErrorCode NS_ADVECTION(ProblemData *problem, DM dm, void *ctx);
extern PetscErrorCode NS_ADVECTION2D(ProblemData *problem, DM dm, void *ctx);

// Print function for each problem
extern PetscErrorCode PRINT_NEWTONIAN(ProblemData *problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_EULER_VORTEX(ProblemData *problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_SHOCKTUBE(ProblemData *problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_ADVECTION(ProblemData *problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_ADVECTION2D(ProblemData *problem, AppCtx app_ctx);

// -----------------------------------------------------------------------------
// libCEED functions
// -----------------------------------------------------------------------------
// Utility function - essential BC dofs are encoded in closure indices as -(i+1).
PetscInt Involute(PetscInt i);

// Utility function to create local CEED restriction
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr);

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, PetscInt value, CeedInt Q, CeedInt q_data_size,
                                       CeedElemRestriction *elem_restr_q, CeedElemRestriction *elem_restr_x, CeedElemRestriction *elem_restr_qd_i);

// Utility function to create CEED Composite Operator for the entire domain
PetscErrorCode CreateOperatorForDomain(Ceed ceed, DM dm, SimpleBC bc, CeedData ceed_data, Physics phys, CeedOperator op_apply_vol,
                                       CeedOperator op_apply_ijacobian_vol, CeedInt height, CeedInt P_sur, CeedInt Q_sur, CeedInt q_data_size_sur,
                                       CeedInt jac_data_size_sur, CeedOperator *op_apply, CeedOperator *op_apply_ijacobian);

PetscErrorCode SetupLibceed(Ceed ceed, CeedData ceed_data, DM dm, User user, AppCtx app_ctx, ProblemData *problem, SimpleBC bc);

// -----------------------------------------------------------------------------
// Time-stepping functions
// -----------------------------------------------------------------------------
// Compute mass matrix for explicit scheme
PetscErrorCode ComputeLumpedMassMatrix(Ceed ceed, DM dm, CeedData ceed_data, Vec M);

// RHS (Explicit time-stepper) function setup
PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *user_data);

// Implicit time-stepper function setup
PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, Vec G, void *user_data);

// User provided TS Monitor
PetscErrorCode TSMonitor_NS(TS ts, PetscInt step_no, PetscReal time, Vec Q, void *ctx);

// TS: Create, setup, and solve
PetscErrorCode TSSolve_NS(DM dm, User user, AppCtx app_ctx, Physics phys, Vec *Q, PetscScalar *f_time, TS *ts);

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
// Create mesh
PetscErrorCode CreateDM(MPI_Comm comm, ProblemData *problem, MatType, VecType, DM *dm);

// Set up DM
PetscErrorCode SetUpDM(DM dm, ProblemData *problem, PetscInt degree, SimpleBC bc, Physics phys);

// Refine DM for high-order viz
PetscErrorCode VizRefineDM(DM dm, User user, ProblemData *problem, SimpleBC bc, Physics phys);

// -----------------------------------------------------------------------------
// Process command line options
// -----------------------------------------------------------------------------
// Register problems to be available on the command line
PetscErrorCode RegisterProblems_NS(AppCtx app_ctx);

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx, SimpleBC bc);

// -----------------------------------------------------------------------------
// Miscellaneous utility functions
// -----------------------------------------------------------------------------
PetscErrorCode ICs_FixMultiplicity(DM dm, CeedData ceed_data, User user, Vec Q_loc, Vec Q, CeedScalar time);

PetscErrorCode DMPlexInsertBoundaryValues_NS(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM, Vec cell_geom_FVM,
                                             Vec grad_FVM);

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTests_NS(AppCtx app_ctx, Vec Q);

// Get error for problems with exact solutions
PetscErrorCode GetError_NS(CeedData ceed_data, DM dm, User user, Vec Q, PetscScalar final_time);

// Post-processing
PetscErrorCode PostProcess_NS(TS ts, CeedData ceed_data, DM dm, ProblemData *problem, User user, Vec Q, PetscScalar final_time);

// -- Gather initial Q values in case of continuation of simulation
PetscErrorCode SetupICsFromBinary(MPI_Comm comm, AppCtx app_ctx, Vec Q);

// Record boundary values from initial condition
PetscErrorCode SetBCsFromICs_NS(DM dm, Vec Q, Vec Q_loc);

// -----------------------------------------------------------------------------
// Boundary Condition Related Functions
// -----------------------------------------------------------------------------

// Setup StrongBCs that use QFunctions
PetscErrorCode SetupStrongBC_Ceed(Ceed ceed, CeedData ceed_data, DM dm, User user, AppCtx app_ctx, ProblemData *problem, SimpleBC bc, CeedInt Q_sur,
                                  CeedInt q_data_size_sur);

PetscErrorCode FreestreamBCSetup(ProblemData *problem, DM dm, void *ctx);

#endif  // libceed_fluids_examples_navier_stokes_h
