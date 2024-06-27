// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <bc_definition.h>
#include <log_events.h>
#include <mat-ceed.h>
#include <petsc-ceed-utils.h>
#include <petscts.h>
#include <stdbool.h>

#include "./include/petsc_ops.h"
#include "qfunctions/newtonian_types.h"

#if PETSC_VERSION_LT(3, 21, 0)
#error "PETSc v3.21 or later is required"
#endif

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------

// Euler - test cases
typedef enum {
  EULER_TEST_ISENTROPIC_VORTEX = 0,
  EULER_TEST_1                 = 1,
  EULER_TEST_2                 = 2,
  EULER_TEST_3                 = 3,
  EULER_TEST_4                 = 4,
  EULER_TEST_5                 = 5,
} EulerTestType;
static const char *const EulerTestTypes[] = {"ISENTROPIC_VORTEX", "1", "2", "3", "4", "5", "EulerTestType", "EULER_TEST_", NULL};

// Advection - Wind types
static const char *const WindTypes[] = {"ROTATION", "TRANSLATION", "WindType", "WIND_", NULL};

// Advection - Initial Condition Types
static const char *const AdvectionICTypes[] = {"SPHERE", "CYLINDER", "COSINE_HILL", "SKEW", "AdvectionICType", "ADVECTIONIC_", NULL};

// Advection - Bubble Continuity Types
static const char *const BubbleContinuityTypes[] = {"SMOOTH", "BACK_SHARP", "THICK", "COSINE", "BubbleContinuityType", "BUBBLE_CONTINUITY_", NULL};

// Stabilization methods
static const char *const StabilizationTypes[] = {"NONE", "SU", "SUPG", "StabilizationType", "STAB_", NULL};

// Stabilization tau constants
static const char *const StabilizationTauTypes[] = {"CTAU", "ADVDIFF_SHAKIB", "ADVDIFF_SHAKIB_P", "StabilizationTauType", "STAB_TAU_", NULL};

// Test mode type
typedef enum {
  TESTTYPE_NONE           = 0,
  TESTTYPE_SOLVER         = 1,
  TESTTYPE_TURB_SPANSTATS = 2,
  TESTTYPE_DIFF_FILTER    = 3,
} TestType;
static const char *const TestTypes[] = {"NONE", "SOLVER", "TURB_SPANSTATS", "DIFF_FILTER", "TestType", "TESTTYPE_", NULL};

// Mesh transformation type
typedef enum {
  MESH_TRANSFORM_NONE      = 0,
  MESH_TRANSFORM_PLATEMESH = 1,
} MeshTransformType;
static const char *const MeshTransformTypes[] = {"NONE", "PLATEMESH", "MeshTransformType", "MESH_TRANSFORM_", NULL};

static const char *const DifferentialFilterDampingFunctions[] = {
    "NONE", "VAN_DRIEST", "MMS", "DifferentialFilterDampingFunction", "DIFF_FILTER_DAMP_", NULL};

// -----------------------------------------------------------------------------
// Structs
// -----------------------------------------------------------------------------
// Structs declarations
typedef struct AppCtx_private      *AppCtx;
typedef struct CeedData_private    *CeedData;
typedef struct User_private        *User;
typedef struct Units_private       *Units;
typedef struct SimpleBC_private    *SimpleBC;
typedef struct Physics_private     *Physics;
typedef struct ProblemData_private *ProblemData;

// Application context from user command line options
struct AppCtx_private {
  // libCEED arguments
  char     ceed_resource[PETSC_MAX_PATH_LEN];  // libCEED backend
  PetscInt degree;
  PetscInt q_extra;
  // Solver arguments
  MatType amat_type;
  // Post-processing arguments
  PetscInt  checkpoint_interval;
  PetscInt  viz_refine;
  PetscInt  cont_steps;
  PetscReal cont_time;
  char      cont_file[PETSC_MAX_PATH_LEN];
  char      cont_time_file[PETSC_MAX_PATH_LEN];
  char      output_dir[PETSC_MAX_PATH_LEN];
  PetscBool add_stepnum2bin;
  PetscBool checkpoint_vtk;
  // Problem type arguments
  PetscFunctionList problems;
  char              problem_name[PETSC_MAX_PATH_LEN];
  // Test mode arguments
  TestType    test_type;
  PetscScalar test_tol;
  char        test_file_path[PETSC_MAX_PATH_LEN];
  // Turbulent spanwise statistics
  PetscBool         turb_spanstats_enable;
  PetscInt          turb_spanstats_collect_interval;
  PetscInt          turb_spanstats_viewer_interval;
  PetscViewer       turb_spanstats_viewer;
  PetscViewerFormat turb_spanstats_viewer_format;
  // Wall forces
  struct {
    PetscInt          num_wall;
    PetscInt         *walls;
    PetscViewer       viewer;
    PetscViewerFormat viewer_format;
    PetscBool         header_written;
  } wall_forces;
  // Differential Filtering
  PetscBool         diff_filter_monitor;
  MeshTransformType mesh_transform_type;
};

// libCEED data struct
struct CeedData_private {
  CeedVector           x_coord, q_data;
  CeedBasis            basis_x, basis_q;
  CeedElemRestriction  elem_restr_x, elem_restr_q, elem_restr_qd_i;
  OperatorApplyContext op_ics_ctx;
};

typedef struct {
  DM                    dm;
  PetscSF               sf;  // For communicating child data to parents
  OperatorApplyContext  op_stats_collect_ctx, op_proj_rhs_ctx;
  PetscInt              num_comp_stats;
  Vec                   Child_Stats_loc, Parent_Stats_loc;
  KSP                   ksp;         // For the L^2 projection solve
  CeedScalar            span_width;  // spanwise width of the child domain
  PetscBool             do_mms_test;
  OperatorApplyContext  mms_error_ctx;
  CeedContextFieldLabel solution_time_label, previous_time_label;
} SpanStatsData;

typedef struct {
  DM                   dm;
  PetscInt             num_comp;
  OperatorApplyContext l2_rhs_ctx;
  KSP                  ksp;
} *NodalProjectionData;

typedef struct {
  DM                    dm_filter;
  PetscInt              num_filtered_fields;
  CeedInt              *num_field_components;
  PetscInt              field_prim_state, field_velo_prod;
  OperatorApplyContext  op_rhs_ctx;
  KSP                   ksp;
  PetscObjectState      X_loc_state;
  PetscBool             do_mms_test;
  CeedContextFieldLabel filter_width_scaling_label;
} *DiffFilterData;

// PETSc user data
struct User_private {
  MPI_Comm             comm;
  DM                   dm;
  DM                   dm_viz;
  Mat                  interp_viz;
  Ceed                 ceed;
  Units                units;
  Vec                  Q_loc, Q_dot_loc;
  Physics              phys;
  AppCtx               app_ctx;
  CeedVector           q_ceed, q_dot_ceed, g_ceed, x_ceed;
  CeedOperator         op_ifunction;
  Mat                  mat_ijacobian;
  KSP                  mass_ksp;
  OperatorApplyContext op_rhs_ctx, op_strong_bc_ctx;
  CeedScalar           time_bc_set;
  SpanStatsData        spanstats;
  NodalProjectionData  grad_velo_proj;
  DiffFilterData       diff_filter;
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
  PetscInt num_inflow, num_outflow, num_freestream, num_slip;
  PetscInt inflows[16], outflows[16], freestreams[16], slips[16];
};

// Struct that contains all enums and structs used for the physics of all problems
struct Physics_private {
  PetscBool             implicit;
  StateVariable         state_var;
  CeedContextFieldLabel solution_time_label;
  CeedContextFieldLabel stg_solution_time_label;
  CeedContextFieldLabel timestep_size_label;
  CeedContextFieldLabel ics_time_label;
};

PetscErrorCode BoundaryConditionSetUp(User user, ProblemData problem, AppCtx app_ctx, SimpleBC bc);

typedef struct {
  CeedQFunctionUser    qfunction;
  const char          *qfunction_loc;
  CeedQFunctionContext qfunction_context;
} ProblemQFunctionSpec;

// Problem specific data
struct ProblemData_private {
  CeedInt              dim, q_data_size_vol, q_data_size_sur, jac_data_size_sur;
  CeedScalar           dm_scale;
  ProblemQFunctionSpec ics, apply_vol_rhs, apply_vol_ifunction, apply_vol_ijacobian, apply_inflow, apply_outflow, apply_freestream, apply_slip,
      apply_inflow_jacobian, apply_outflow_jacobian, apply_freestream_jacobian, apply_slip_jacobian;
  bool          compute_exact_solution_error;
  PetscBool     set_bc_from_ics, use_strong_bc_ceed, uses_newtonian;
  size_t        num_bc_defs;
  BCDefinition *bc_defs;
  PetscErrorCode (*print_info)(User, ProblemData, AppCtx);
  PetscErrorCode (*create_mass_operator)(User, CeedOperator *);
};

extern int FreeContextPetsc(void *);

// -----------------------------------------------------------------------------
// Set up problems
// -----------------------------------------------------------------------------
// Set up function for each problem
extern PetscErrorCode NS_TAYLOR_GREEN(ProblemData problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_GAUSSIAN_WAVE(ProblemData problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_CHANNEL(ProblemData problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_BLASIUS(ProblemData problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_NEWTONIAN_IG(ProblemData problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_DENSITY_CURRENT(ProblemData problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_EULER_VORTEX(ProblemData problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_SHOCKTUBE(ProblemData problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_ADVECTION(ProblemData problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_ADVECTION2D(ProblemData problem, DM dm, void *ctx, SimpleBC bc);

// Print function for each problem
extern PetscErrorCode PRINT_NEWTONIAN(User user, ProblemData problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_EULER_VORTEX(User user, ProblemData problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_SHOCKTUBE(User user, ProblemData problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_ADVECTION(User user, ProblemData problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_ADVECTION2D(User user, ProblemData problem, AppCtx app_ctx);

PetscErrorCode PrintRunInfo(User user, Physics phys_ctx, ProblemData problem, TS ts);

// -----------------------------------------------------------------------------
// libCEED functions
// -----------------------------------------------------------------------------
// Utility function to create local CEED restriction
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, CeedInt label_value, PetscInt dm_field,
                                         CeedElemRestriction *elem_restr);

PetscErrorCode DMPlexCeedElemRestrictionCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field,
                                               CeedElemRestriction *restriction);
PetscErrorCode DMPlexCeedElemRestrictionCoordinateCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                         CeedElemRestriction *restriction);
PetscErrorCode DMPlexCeedElemRestrictionQDataCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                    PetscInt q_data_size, CeedElemRestriction *restriction);
PetscErrorCode DMPlexCeedElemRestrictionCollocatedCreate(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height,
                                                         PetscInt q_data_size, CeedElemRestriction *restriction);

PetscErrorCode CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label, CeedInt label_value, CeedInt height, CeedInt dm_field, CeedBasis *basis);

PetscErrorCode SetupLibceed(Ceed ceed, CeedData ceed_data, DM dm, User user, AppCtx app_ctx, ProblemData problem, SimpleBC bc);

PetscErrorCode QDataGet(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, CeedElemRestriction elem_restr_x, CeedBasis basis_x,
                        CeedVector x_coord, CeedElemRestriction *elem_restr_qd, CeedVector *q_data, CeedInt *q_data_size);
PetscErrorCode QDataGetNumComponents(DM dm, CeedInt *q_data_size);
PetscErrorCode QDataBoundaryGet(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, CeedElemRestriction elem_restr_x, CeedBasis basis_x,
                                CeedVector x_coord, CeedElemRestriction *elem_restr_qd, CeedVector *q_data, CeedInt *q_data_size);
PetscErrorCode QDataBoundaryGetNumComponents(DM dm, CeedInt *q_data_size);
// -----------------------------------------------------------------------------
// Time-stepping functions
// -----------------------------------------------------------------------------
// RHS (Explicit time-stepper) function setup
PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *user_data);

// Implicit time-stepper function setup
PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, Vec G, void *user_data);

// User provided TS Monitor
PetscErrorCode TSMonitor_NS(TS ts, PetscInt step_no, PetscReal time, Vec Q, void *ctx);

// TS: Create, setup, and solve
PetscErrorCode TSSolve_NS(DM dm, User user, AppCtx app_ctx, Physics phys, ProblemData problem, Vec *Q, PetscScalar *f_time, TS *ts);

// Update Boundary Values when time has changed
PetscErrorCode UpdateBoundaryValues(User user, Vec Q_loc, PetscReal t);

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
// Create mesh
PetscErrorCode CreateDM(MPI_Comm comm, ProblemData problem, MatType, VecType, DM *dm);

// Set up DM
PetscErrorCode SetUpDM(DM dm, ProblemData problem, PetscInt degree, PetscInt q_extra, SimpleBC bc, Physics phys);
PetscErrorCode DMSetupByOrderBegin_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                       PetscInt num_fields, const PetscInt *field_sizes, DM dm);
PetscErrorCode DMSetupByOrderEnd_FEM(PetscBool setup_coords, DM dm);
PetscErrorCode DMSetupByOrder_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                  PetscInt num_fields, const PetscInt *field_sizes, DM dm);

// Refine DM for high-order viz
PetscErrorCode VizRefineDM(DM dm, User user, ProblemData problem, SimpleBC bc, Physics phys);

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
PetscErrorCode GetInverseMultiplicity(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field,
                                      PetscBool get_global_multiplicity, CeedElemRestriction *elem_restr_inv_multiplicity,
                                      CeedVector *inv_multiplicity);
PetscErrorCode ICs_FixMultiplicity(DM dm, CeedData ceed_data, User user, Vec Q_loc, Vec Q, CeedScalar time);

PetscErrorCode DMPlexInsertBoundaryValues_FromICs(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM, Vec cell_geom_FVM,
                                                  Vec grad_FVM);

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTest(AppCtx app_ctx, Vec Q);

// Get error for problems with exact solutions
PetscErrorCode PrintError(CeedData ceed_data, DM dm, User user, Vec Q, PetscScalar final_time);

// Post-processing
PetscErrorCode PostProcess(TS ts, CeedData ceed_data, DM dm, ProblemData problem, User user, Vec Q, PetscScalar final_time);

// -- Gather initial Q values in case of continuation of simulation
PetscErrorCode SetupICsFromBinary(MPI_Comm comm, AppCtx app_ctx, Vec Q);

// Record boundary values from initial condition
PetscErrorCode SetBCsFromICs(DM dm, Vec Q, Vec Q_loc);

// Versioning token for binary checkpoints
extern const PetscInt32 FLUIDS_FILE_TOKEN;  // for backwards compatibility
extern const PetscInt32 FLUIDS_FILE_TOKEN_32;
extern const PetscInt32 FLUIDS_FILE_TOKEN_64;

// Create appropriate mass qfunction based on number of components N
PetscErrorCode CreateMassQFunction(Ceed ceed, CeedInt N, CeedInt q_data_size, CeedQFunction *qf);

PetscErrorCode NodalProjectionDataDestroy(NodalProjectionData context);

PetscErrorCode PhastaDatFileOpen(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], const PetscInt char_array_len, PetscInt dims[2],
                                 FILE **fp);

PetscErrorCode PhastaDatFileGetNRows(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], PetscInt *nrows);

PetscErrorCode PhastaDatFileReadToArrayReal(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], PetscReal array[]);

// -----------------------------------------------------------------------------
// Turbulence Statistics Collection Functions
// -----------------------------------------------------------------------------

PetscErrorCode TurbulenceStatisticsSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData problem);
PetscErrorCode TSMonitor_TurbulenceStatistics(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, void *ctx);
PetscErrorCode TurbulenceStatisticsDestroy(User user, CeedData ceed_data);

// -----------------------------------------------------------------------------
// Data-Driven Subgrid Stress (DD-SGS) Modeling Functions
// -----------------------------------------------------------------------------
PetscErrorCode VelocityGradientProjectionSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData problem, StateVariable state_var_input,
                                               CeedElemRestriction elem_restr_input, CeedBasis basis_input, NodalProjectionData *pgrad_velo_proj);
PetscErrorCode VelocityGradientProjectionApply(NodalProjectionData grad_velo_proj, Vec Q_loc, Vec VelocityGradient);
PetscErrorCode GridAnisotropyTensorProjectionSetupApply(Ceed ceed, User user, CeedData ceed_data, CeedElemRestriction *elem_restr_grid_aniso,
                                                        CeedVector *grid_aniso_vector);
PetscErrorCode GridAnisotropyTensorCalculateCollocatedVector(Ceed ceed, User user, CeedData ceed_data, CeedElemRestriction *elem_restr_grid_aniso,
                                                             CeedVector *aniso_colloc_ceed, PetscInt *num_comp_aniso);

// -----------------------------------------------------------------------------
// Boundary Condition Related Functions
// -----------------------------------------------------------------------------

// Setup StrongBCs that use QFunctions
PetscErrorCode SetupStrongBC_Ceed(Ceed ceed, CeedData ceed_data, DM dm, User user, ProblemData problem, SimpleBC bc);

PetscErrorCode FreestreamBCSetup(ProblemData problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx, const StatePrimitive *reference);
PetscErrorCode OutflowBCSetup(ProblemData problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx, const StatePrimitive *reference);
PetscErrorCode SlipBCSetup(ProblemData problem, DM dm, void *ctx, CeedQFunctionContext newtonian_ig_qfctx);

// -----------------------------------------------------------------------------
// Differential Filtering Functions
// -----------------------------------------------------------------------------

PetscErrorCode DifferentialFilterSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData problem);
PetscErrorCode DifferentialFilterDataDestroy(DiffFilterData diff_filter);
PetscErrorCode TSMonitor_DifferentialFilter(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, void *ctx);
PetscErrorCode DifferentialFilterApply(User user, const PetscReal solution_time, const Vec Q, Vec Filtered_Solution);
PetscErrorCode DifferentialFilterMmsICSetup(ProblemData problem);
