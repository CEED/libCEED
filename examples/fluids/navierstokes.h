// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef libceed_fluids_examples_navier_stokes_h
#define libceed_fluids_examples_navier_stokes_h

#include <ceed.h>
#include <mat-ceed.h>
#include <petscts.h>
#include <stdbool.h>

#include "./include/petsc_ops.h"
#include "qfunctions/newtonian_types.h"
#include "qfunctions/stabilization_types.h"

#if PETSC_VERSION_LT(3, 20, 0)
#error "PETSc v3.20 or later is required"
#endif

#if PETSC_VERSION_LT(3, 21, 0)
#define DMSetCoordinateDisc(a, b, c) DMProjectCoordinates(a, b)
#endif

#define PetscCallCeed(ceed, ...)                                    \
  do {                                                              \
    int ierr = __VA_ARGS__;                                         \
    if (ierr != CEED_ERROR_SUCCESS) {                               \
      const char *error_message;                                    \
      CeedGetErrorMessage(ceed, &error_message);                    \
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", error_message); \
    }                                                               \
  } while (0)

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------
// Translate PetscMemType to CeedMemType
static inline CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

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

// Advection - Wind types
static const char *const WindTypes[] = {"rotation", "translation", "WindType", "WIND_", NULL};

// Advection - Initial Condition Types
static const char *const AdvectionICTypes[] = {"sphere", "cylinder", "cosine_hill", "skew", "AdvectionICType", "ADVECTIONIC_", NULL};

// Advection - Bubble Continuity Types
static const char *const BubbleContinuityTypes[] = {"smooth", "back_sharp", "thick", "cosine", "BubbleContinuityType", "BUBBLE_CONTINUITY_", NULL};

// Stabilization methods
static const char *const StabilizationTypes[] = {"none", "SU", "SUPG", "StabilizationType", "STAB_", NULL};

// Stabilization tau constants
static const char *const StabilizationTauTypes[] = {"Ctau", "AdvDiff_Shakib", "AdvDiff_Shakib_P", "StabilizationTauType", "STAB_TAU_", NULL};

// Test mode type
typedef enum {
  TESTTYPE_NONE           = 0,
  TESTTYPE_SOLVER         = 1,
  TESTTYPE_TURB_SPANSTATS = 2,
  TESTTYPE_DIFF_FILTER    = 3,
} TestType;
static const char *const TestTypes[] = {"none", "solver", "turb_spanstats", "diff_filter", "TestType", "TESTTYPE_", NULL};

// Subgrid-Stress mode type
typedef enum {
  SGS_MODEL_NONE        = 0,
  SGS_MODEL_DATA_DRIVEN = 1,
} SGSModelType;
static const char *const SGSModelTypes[] = {"none", "data_driven", "SGSModelType", "SGS_MODEL_", NULL};

// Mesh transformation type
typedef enum {
  MESH_TRANSFORM_NONE      = 0,
  MESH_TRANSFORM_PLATEMESH = 1,
} MeshTransformType;
static const char *const MeshTransformTypes[] = {"none", "platemesh", "MeshTransformType", "MESH_TRANSFORM_", NULL};

static const char *const DifferentialFilterDampingFunctions[] = {
    "none", "van_driest", "mms", "DifferentialFilterDampingFunction", "DIFF_FILTER_DAMP_", NULL};

// -----------------------------------------------------------------------------
// Log Events
// -----------------------------------------------------------------------------
extern PetscLogEvent FLUIDS_CeedOperatorApply;
extern PetscLogEvent FLUIDS_CeedOperatorAssemble;
extern PetscLogEvent FLUIDS_CeedOperatorAssembleDiagonal;
extern PetscLogEvent FLUIDS_CeedOperatorAssemblePointBlockDiagonal;
extern PetscLogEvent FLUIDS_SmartRedis_Init;
extern PetscLogEvent FLUIDS_SmartRedis_Meta;
extern PetscLogEvent FLUIDS_SmartRedis_Train;
extern PetscLogEvent FLUIDS_TrainDataCompute;
extern PetscLogEvent FLUIDS_DifferentialFilter;
extern PetscLogEvent FLUIDS_VelocityGradientProjection;
PetscErrorCode       RegisterLogEvents();

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
  // Subgrid Stress Model
  SGSModelType sgs_model_type;
  PetscBool    sgs_train_enable;
  // Differential Filtering
  PetscBool         diff_filter_monitor;
  MeshTransformType mesh_transform_type;
};

// libCEED data struct
struct CeedData_private {
  CeedVector           x_coord, q_data;
  CeedBasis            basis_x, basis_xc, basis_q, basis_x_sur, basis_q_sur, basis_xc_sur;
  CeedElemRestriction  elem_restr_x, elem_restr_q, elem_restr_qd_i;
  CeedOperator         op_setup_vol;
  OperatorApplyContext op_ics_ctx;
  CeedQFunction        qf_setup_vol, qf_ics, qf_rhs_vol, qf_ifunction_vol, qf_setup_sur, qf_apply_inflow, qf_apply_inflow_jacobian, qf_apply_outflow,
      qf_apply_outflow_jacobian, qf_apply_freestream, qf_apply_freestream_jacobian, qf_apply_slip, qf_apply_slip_jacobian;
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

typedef PetscErrorCode (*SgsDDNodalStressEval)(User user, Vec Q_loc, Vec VelocityGradient, Vec SGSNodal_loc);
typedef PetscErrorCode (*SgsDDNodalStressInference)(Vec DD_Inputs_loc, Vec DD_Outputs_loc, void *ctx);
typedef struct {
  DM                        dm_sgs, dm_dd_inputs, dm_dd_outputs;
  PetscInt                  num_comp_sgs, num_comp_inputs, num_comp_outputs;
  OperatorApplyContext      op_nodal_evaluation_ctx, op_nodal_dd_inputs_ctx, op_nodal_dd_outputs_ctx, op_sgs_apply_ctx;
  CeedVector                sgs_nodal_ceed, grad_velo_ceed;
  SgsDDNodalStressEval      sgs_nodal_eval;
  SgsDDNodalStressInference sgs_nodal_inference;
  void                     *sgs_nodal_inference_ctx;
  PetscErrorCode (*sgs_nodal_inference_ctx_destroy)(void *ctx);
} *SgsDDData;

typedef struct {
  DM                   dm_dd_training;
  PetscInt             num_comp_dd_inputs, write_data_interval;
  OperatorApplyContext op_training_data_calc_ctx;
  NodalProjectionData  filtered_grad_velo_proj;
  size_t               training_data_array_dims[2];
  PetscBool            overwrite_training_data;
} *SGS_DD_TrainingData;

typedef struct {
  DM                   dm_filter;
  PetscInt             num_filtered_fields;
  CeedInt             *num_field_components;
  PetscInt             field_prim_state, field_velo_prod;
  OperatorApplyContext op_rhs_ctx;
  KSP                  ksp;
  PetscBool            do_mms_test;
} *DiffFilterData;

typedef struct {
  void    *client;
  char     rank_id_name[16];
  PetscInt collocated_database_num_ranks;
} *SmartSimData;

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
  CeedOperator         op_rhs_vol, op_ifunction_vol, op_ifunction;
  Mat                  mat_ijacobian;
  KSP                  mass_ksp;
  OperatorApplyContext op_rhs_ctx, op_strong_bc_ctx;
  CeedScalar           time_bc_set;
  SpanStatsData        spanstats;
  NodalProjectionData  grad_velo_proj;
  SgsDDData            sgs_dd_data;
  DiffFilterData       diff_filter;
  SmartSimData         smartsim;
  SGS_DD_TrainingData  sgs_dd_train;
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
      num_symmetry[3],  // Number of faces with symmetry BCs
      num_inflow, num_outflow, num_freestream, num_slip;
  PetscInt walls[16], symmetries[3][16], inflows[16], outflows[16], freestreams[16], slips[16];
};

// Struct that contains all enums and structs used for the physics of all problems
struct Physics_private {
  PetscBool             implicit;
  StateVariable         state_var;
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
typedef struct ProblemData_private ProblemData;
struct ProblemData_private {
  CeedInt              dim, q_data_size_vol, q_data_size_sur, jac_data_size_sur;
  CeedScalar           dm_scale;
  ProblemQFunctionSpec setup_vol, setup_sur, ics, apply_vol_rhs, apply_vol_ifunction, apply_vol_ijacobian, apply_inflow, apply_outflow,
      apply_freestream, apply_slip, apply_inflow_jacobian, apply_outflow_jacobian, apply_freestream_jacobian, apply_slip_jacobian;
  bool      non_zero_time;
  PetscBool bc_from_ics, use_strong_bc_ceed, uses_newtonian;
  PetscErrorCode (*print_info)(User, ProblemData *, AppCtx);
};

extern int FreeContextPetsc(void *);

// -----------------------------------------------------------------------------
// Set up problems
// -----------------------------------------------------------------------------
// Set up function for each problem
extern PetscErrorCode NS_TAYLOR_GREEN(ProblemData *problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_GAUSSIAN_WAVE(ProblemData *problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_CHANNEL(ProblemData *problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_BLASIUS(ProblemData *problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_NEWTONIAN_IG(ProblemData *problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_DENSITY_CURRENT(ProblemData *problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_EULER_VORTEX(ProblemData *problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_SHOCKTUBE(ProblemData *problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_ADVECTION(ProblemData *problem, DM dm, void *ctx, SimpleBC bc);
extern PetscErrorCode NS_ADVECTION2D(ProblemData *problem, DM dm, void *ctx, SimpleBC bc);

// Print function for each problem
extern PetscErrorCode PRINT_NEWTONIAN(User user, ProblemData *problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_EULER_VORTEX(User user, ProblemData *problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_SHOCKTUBE(User user, ProblemData *problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_ADVECTION(User user, ProblemData *problem, AppCtx app_ctx);

extern PetscErrorCode PRINT_ADVECTION2D(User user, ProblemData *problem, AppCtx app_ctx);

PetscErrorCode PrintRunInfo(User user, Physics phys_ctx, ProblemData *problem, MPI_Comm comm);

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

// Utility function to create CEED Composite Operator for the entire domain
PetscErrorCode CreateOperatorForDomain(Ceed ceed, DM dm, SimpleBC bc, CeedData ceed_data, Physics phys, CeedOperator op_apply_vol,
                                       CeedOperator op_apply_ijacobian_vol, CeedInt height, CeedInt P_sur, CeedInt Q_sur, CeedInt q_data_size_sur,
                                       CeedInt jac_data_size_sur, CeedOperator *op_apply, CeedOperator *op_apply_ijacobian);

PetscErrorCode SetupLibceed(Ceed ceed, CeedData ceed_data, DM dm, User user, AppCtx app_ctx, ProblemData *problem, SimpleBC bc);

// -----------------------------------------------------------------------------
// Time-stepping functions
// -----------------------------------------------------------------------------
// Create KSP to solve the inverse mass operator for explicit time stepping schemes
PetscErrorCode CreateKSPMassOperator(User user, CeedData ceed_data);

// RHS (Explicit time-stepper) function setup
PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *user_data);

// Implicit time-stepper function setup
PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Q_dot, Vec G, void *user_data);

// User provided TS Monitor
PetscErrorCode TSMonitor_NS(TS ts, PetscInt step_no, PetscReal time, Vec Q, void *ctx);

// TS: Create, setup, and solve
PetscErrorCode TSSolve_NS(DM dm, User user, AppCtx app_ctx, Physics phys, Vec *Q, PetscScalar *f_time, TS *ts);

// Update Boundary Values when time has changed
PetscErrorCode UpdateBoundaryValues(User user, Vec Q_loc, PetscReal t);

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
// Create mesh
PetscErrorCode CreateDM(MPI_Comm comm, ProblemData *problem, MatType, VecType, DM *dm);

// Set up DM
PetscErrorCode SetUpDM(DM dm, ProblemData *problem, PetscInt degree, PetscInt q_extra, SimpleBC bc, Physics phys);
PetscErrorCode DMSetupByOrderBegin_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                       PetscInt num_fields, const PetscInt *field_sizes, DM dm);
PetscErrorCode DMSetupByOrderEnd_FEM(PetscBool setup_coords, DM dm);
PetscErrorCode DMSetupByOrder_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                  PetscInt num_fields, const PetscInt *field_sizes, DM dm);

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

PetscErrorCode DMPlexInsertBoundaryValues_FromICs(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM, Vec cell_geom_FVM,
                                                  Vec grad_FVM);

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTest(AppCtx app_ctx, Vec Q);

// Get error for problems with exact solutions
PetscErrorCode PrintError(CeedData ceed_data, DM dm, User user, Vec Q, PetscScalar final_time);

// Post-processing
PetscErrorCode PostProcess(TS ts, CeedData ceed_data, DM dm, ProblemData *problem, User user, Vec Q, PetscScalar final_time);

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

PetscErrorCode IntArrayC2P(PetscInt num_entries, CeedInt **array_ceed, PetscInt **array_petsc);
PetscErrorCode IntArrayP2C(PetscInt num_entries, PetscInt **array_petsc, CeedInt **array_ceed);

// -----------------------------------------------------------------------------
// Turbulence Statistics Collection Functions
// -----------------------------------------------------------------------------

PetscErrorCode TurbulenceStatisticsSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem);
PetscErrorCode TSMonitor_TurbulenceStatistics(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, void *ctx);
PetscErrorCode TurbulenceStatisticsDestroy(User user, CeedData ceed_data);

// -----------------------------------------------------------------------------
// Data-Driven Subgrid Stress (DD-SGS) Modeling Functions
// -----------------------------------------------------------------------------

PetscErrorCode SgsDDSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem);
PetscErrorCode SgsDDDataDestroy(SgsDDData sgs_dd_data);
PetscErrorCode SgsDDApplyIFunction(User user, const Vec Q_loc, Vec G_loc);
PetscErrorCode VelocityGradientProjectionSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem, StateVariable state_var_input,
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
PetscErrorCode SetupStrongBC_Ceed(Ceed ceed, CeedData ceed_data, DM dm, User user, ProblemData *problem, SimpleBC bc);

PetscErrorCode FreestreamBCSetup(ProblemData *problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx, const StatePrimitive *reference);
PetscErrorCode OutflowBCSetup(ProblemData *problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx, const StatePrimitive *reference);
PetscErrorCode SlipBCSetup(ProblemData *problem, DM dm, void *ctx, CeedQFunctionContext newtonian_ig_qfctx);

// -----------------------------------------------------------------------------
// Differential Filtering Functions
// -----------------------------------------------------------------------------

PetscErrorCode DifferentialFilterSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem);
PetscErrorCode DifferentialFilterDataDestroy(DiffFilterData diff_filter);
PetscErrorCode TSMonitor_DifferentialFilter(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, void *ctx);
PetscErrorCode DifferentialFilterApply(User user, const PetscReal solution_time, const Vec Q, Vec Filtered_Solution);
PetscErrorCode DifferentialFilterMmsICSetup(ProblemData *problem);

// -----------------------------------------------------------------------------
// SGS Data-Driven Training via SmartSim
// -----------------------------------------------------------------------------
PetscErrorCode SmartSimSetup(User user);
PetscErrorCode SmartSimDataDestroy(SmartSimData smartsim);
PetscErrorCode SGS_DD_TrainingSetup(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem);
PetscErrorCode TSMonitor_SGS_DD_Training(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx);
PetscErrorCode TSPostStep_SGS_DD_Training(TS ts);
PetscErrorCode SGS_DD_TrainingDataDestroy(SGS_DD_TrainingData sgs_dd_train);

#endif  // libceed_fluids_examples_navier_stokes_h
