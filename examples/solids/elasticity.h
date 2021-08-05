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
#  define DMAddBoundary(a,b,c,d,e,f,g,h,i,j,k,l,m,n) DMAddBoundary(a,b,c,e,h,i,j,k,f,g,m)
#elif PETSC_VERSION_LT(3,16,0)
#  define DMAddBoundary(a,b,c,d,e,f,g,h,i,j,k,l,m,n) DMAddBoundary(a,b,c,e,h,i,j,k,l,f,g,m)
#else
#  define DMAddBoundary(a,b,c,d,e,f,g,h,i,j,k,l,m,n) DMAddBoundary(a,b,c,d,f,g,h,i,j,k,l,m,n)
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
  CeedScalar lambda; //
};
#endif

// -----------------------------------------------------------------------------
// Command Line Options
// -----------------------------------------------------------------------------
// Problem options
typedef enum {
  ELAS_LINEAR = 0, ELAS_SS_NH = 1, ELAS_FSInitial_NH1 = 2, ELAS_FSInitial_NH2 = 3,
  ELAS_FSCurrent_NH1 = 4, ELAS_FSCurrent_NH2 = 5, ELAS_FSInitial_MR1 = 6
} problemType;
static const char *const problemTypes[] = {"Linear",
                                           "SS-NH",
                                           "FSInitial-NH1",
                                           "FSInitial-NH2",
                                           "FSCurrent-NH1",
                                           "FSCurrent-NH2",
                                           "FSInitial-MR1",
                                           "problemType","ELAS_",0
                                          };
static const char *const problemTypesForDisp[] = {"Linear elasticity",
                                                  "Hyperelasticity small strain, Neo-Hookean",
                                                  "Hyperelasticity finite strain Initial configuration Neo-Hookean w/ dXref_dxinit, Grad(u) storage",
                                                  "Hyperelasticity finite strain Initial configuration Neo-Hookean w/ dXref_dxinit, Grad(u), C_inv, constant storage",
                                                  "Hyperelasticity finite strain Current configuration Neo-Hookean w/ dXref_dxinit, Grad(u) storage",
                                                  "Hyperelasticity finite strain Current configuration Neo-Hookean w/ dXref_dxcurr, tau, constant storage",
                                                  "Hyperelasticity finite strain Initial configuration Moony-Rivlin w/ dXref_dxinit, Grad(u) storage"
                                                 };

// Forcing function options
typedef enum {
  FORCE_NONE = 0, FORCE_CONST = 1, FORCE_MMS = 2
} forcingType;
static const char *const forcing_types[] = {"none",
                                            "constant",
                                            "mms",
                                            "forcingType","FORCE_",0
                                           };
static const char *const forcing_types_for_disp[] = {"None",
                                                     "Constant",
                                                     "Manufactured solution"
                                                    };

// Multigrid options
typedef enum {
  MULTIGRID_LOGARITHMIC = 0, MULTIGRID_UNIFORM = 1, MULTIGRID_NONE = 2
} multigridType;
static const char *const multigrid_types [] = {"logarithmic",
                                               "uniform",
                                               "none",
                                               "multigridType","MULTIGRID",0
                                              };
static const char *const multigrid_types_for_disp[] = {"P-multigrid, logarithmic coarsening",
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
  char          ceed_resource[PETSC_MAX_PATH_LEN];     // libCEED backend
  char          mesh_file[PETSC_MAX_PATH_LEN];         // exodusII mesh file
  char          output_dir[PETSC_MAX_PATH_LEN];
  PetscBool     test_mode;
  PetscBool     view_soln;
  PetscBool     view_final_soln;
  PetscViewer   energy_viewer;
  problemType   problem_choice;
  forcingType   forcing_choice;
  multigridType multigrid_choice;
  PetscScalar   nu_smoother;
  PetscInt      degree;
  PetscInt      q_extra;
  PetscInt      num_levels;
  PetscInt      *level_degrees;
  PetscInt      num_increments;                        // Number of steps
  PetscInt      bc_clamp_count;
  PetscInt      bc_clamp_faces[16];
  // [translation; 3] [rotation axis; 3] [rotation magnitude c_0, c_1]
  // The rotations are (c_0 + c_1 s) \pi, where s = x · axis
  PetscScalar   bc_clamp_max[16][8];
  PetscInt      bc_traction_count;
  PetscInt      bc_traction_faces[16];
  PetscScalar   bc_traction_vector[16][3];
  PetscScalar   forcing_vector[3];
  PetscReal     test_tol;
  PetscReal     expect_final_strain;
};

// Problem specific data
// *INDENT-OFF*
typedef struct {
  CeedInt           q_data_size;
  CeedQFunctionUser setup_geo, apply, jacob, energy, diagnostic;
  const char        *setup_geo_loc, *apply_loc, *jacob_loc, *energy_loc,
                    *diagnostic_loc;
  CeedQuadMode      quad_mode;
} problemData;
// *INDENT-ON*

// Data specific to each problem option
extern problemData problem_options[7];

// Forcing function data
typedef struct {
  CeedQFunctionUser setup_forcing;
  const char        *setup_forcing_loc;
} forcingData;

extern forcingData forcing_options[3];

// Data for PETSc Matshell
typedef struct UserMult_private *UserMult;
struct UserMult_private {
  MPI_Comm        comm;
  DM              dm;
  Vec             X_loc, Y_loc, neumann_bcs;
  CeedVector      x_ceed, y_ceed;
  CeedOperator    op;
  CeedQFunction   qf;
  Ceed            ceed;
  PetscScalar     load_increment;
  CeedQFunctionContext ctx_phys, ctx_phys_smoother;
};

// Data for Jacobian setup routine
typedef struct FormJacobCtx_private *FormJacobCtx;
struct FormJacobCtx_private {
  UserMult     *jacob_ctx;
  PetscInt     num_levels;
  SNES         snes_coarse;
  Mat          *jacob_mat, jacob_mat_coarse;
  Vec          u_coarse;
};

// Data for PETSc Prolongation/Restriction Matshell
typedef struct UserMultProlongRestr_private *UserMultProlongRestr;
struct UserMultProlongRestr_private {
  MPI_Comm     comm;
  DM           dm_c, dm_f;
  Vec          loc_vec_c, loc_vec_f;
  CeedVector   ceed_vec_c, ceed_vec_f;
  CeedOperator op_prolong, op_restrict;
  Ceed         ceed;
};

// libCEED data struct for level
typedef struct CeedData_private *CeedData;
struct CeedData_private {
  Ceed                ceed;
  CeedBasis           basis_x, basis_u, basis_c_to_f, basis_energy,
                      basis_diagnostic;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_qd_i,
                      elem_restr_gradu_i,
                      elem_restr_energy, elem_restr_diagnostic,
                      elem_restr_dXdx, elem_restr_tau,
                      elem_restr_C_inv, elem_restr_lam_log_J, elem_restr_qd_diagnostic_i;
  CeedQFunction       qf_apply, qf_jacob, qf_energy, qf_diagnostic;
  CeedOperator        op_apply, op_jacob, op_restrict, op_prolong, op_energy,
                      op_diagnostic;
  CeedVector          q_data, q_data_diagnostic, grad_u, x_ceed,
                      y_ceed, true_soln, dXdx, tau, C_inv, lam_log_J;
};

// Translate PetscMemType to CeedMemType
static inline CeedMemType MemTypeP2C(PetscMemType mem_type) {
  return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}

// -----------------------------------------------------------------------------
// Process command line options
// -----------------------------------------------------------------------------
// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx);

// Process physics options; fix this to be problem specific
PetscErrorCode ProcessPhysics_General(MPI_Comm comm, AppCtx app_ctx,
                                      Physics phys, Physics_MR phys_MR, Units units);

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
PetscErrorCode CreateBCLabel(DM dm, const char name[]);

// Create FE by degree
PetscErrorCode PetscFECreateByDegree(DM dm, PetscInt dim, PetscInt Nc,
                                     PetscBool is_simplex, const char prefix[],
                                     PetscInt order, PetscFE *fem);

// Read mesh and distribute DM in parallel
PetscErrorCode CreateDistributedDM(MPI_Comm comm, AppCtx app_ctx, DM *dm);

// Setup DM with FE space of appropriate degree
PetscErrorCode SetupDMByDegree(DM dm, AppCtx app_ctx, PetscInt order,
                               PetscBool boundary, PetscInt num_comp_u);

// -----------------------------------------------------------------------------
// libCEED Functions
// -----------------------------------------------------------------------------
// Destroy libCEED objects
PetscErrorCode CeedDataDestroy(CeedInt level, CeedData data);

// Utility function - essential BC dofs are encoded in closure indices as -(i+1)
PetscInt Involute(PetscInt i);

// Utility function to create local CEED restriction from DMPlex
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt height, DMLabel domain_label, CeedInt value,
    CeedElemRestriction *elem_restr);

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height,
                                       DMLabel domain_label, PetscInt value, CeedInt P, CeedInt Q, CeedInt q_data_size,
                                       CeedElemRestriction *elem_restr_q, CeedElemRestriction *elem_restr_x,
                                       CeedElemRestriction *elem_restr_qd_i);

// Set up libCEED for a given degree
PetscErrorCode SetupLibceedFineLevel(DM dm, DM dm_energy, DM dm_diagnostic,
                                     Ceed ceed, AppCtx app_ctx,
                                     CeedQFunctionContext phys_ctx,
                                     CeedData *data, PetscInt fine_level,
                                     PetscInt num_comp_u, PetscInt U_g_size,
                                     PetscInt u_loc_size, CeedVector force_ceed,
                                     CeedVector neumann_ceed);

// Set up libCEED multigrid level for a given degree
PetscErrorCode SetupLibceedLevel(DM dm, Ceed ceed, AppCtx app_ctx,
                                 CeedData *data, PetscInt level,
                                 PetscInt num_comp_u, PetscInt U_g_size,
                                 PetscInt u_loc_size, CeedVector fine_mult);

// Setup context data for Jacobian evaluation
PetscErrorCode SetupJacobianCtx(MPI_Comm comm, AppCtx app_ctx, DM dm, Vec V,
                                Vec V_loc, CeedData ceed_data, Ceed ceed,
                                CeedQFunctionContext ctx_phys,
                                CeedQFunctionContext ctx_phys_smoother,
                                UserMult jacobian_ctx);

// Setup context data for prolongation and restriction operators
PetscErrorCode SetupProlongRestrictCtx(MPI_Comm comm, AppCtx app_ctx, DM dm_c,
                                       DM dm_f, Vec V_f, Vec V_loc_c, Vec V_loc_f,
                                       CeedData ceed_data_c, CeedData ceed_data_f,
                                       Ceed ceed,
                                       UserMultProlongRestr prolong_restr_ctx);

// -----------------------------------------------------------------------------
// Jacobian setup
// -----------------------------------------------------------------------------
PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat J_pre, void *ctx);

// -----------------------------------------------------------------------------
// Solution output
// -----------------------------------------------------------------------------
PetscErrorCode ViewSolution(MPI_Comm comm, AppCtx app_ctx, Vec U,
                            PetscInt increment,
                            PetscScalar load_increment);

PetscErrorCode ViewDiagnosticQuantities(MPI_Comm comm, DM dm_U,
                                        UserMult user, AppCtx app_ctx, Vec U,
                                        CeedElemRestriction elem_restr_diagnostic);

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
PetscErrorCode ComputeStrainEnergy(DM dm_energy, UserMult user,
                                   CeedOperator op_energy, Vec X,
                                   PetscReal *energy);

// this function checks to see if the computed energy is close enough to reference file energy.
PetscErrorCode RegressionTests_solids(AppCtx app_ctx, PetscReal energy);

// -----------------------------------------------------------------------------
// Boundary Functions
// -----------------------------------------------------------------------------
// Note: If additional boundary conditions are added, an update is needed in
//         elasticity.h for the boundaryOptions variable.

// BCMMS - boundary function
// Values on all points of the mesh is set based on given solution below
// for u[0], u[1], u[2]
PetscErrorCode BCMMS(PetscInt dim, PetscReal load_increment,
                     const PetscReal coords[], PetscInt num_comp_u,
                     PetscScalar *u, void *ctx);

// BCClamp - fix boundary values with affine transformation at fraction of load
//   increment
PetscErrorCode BCClamp(PetscInt dim, PetscReal load_increment,
                       const PetscReal coords[], PetscInt num_comp_u,
                       PetscScalar *u, void *ctx);

#endif //setup_h
