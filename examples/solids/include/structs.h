// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef libceed_solids_examples_structs_h
#define libceed_solids_examples_structs_h

#include <ceed.h>
#include <petsc.h>

#include "../problems/cl-problems.h"

// -----------------------------------------------------------------------------
// Command Line Options
// -----------------------------------------------------------------------------
// Forcing function options
typedef enum { FORCE_NONE = 0, FORCE_CONST = 1, FORCE_MMS = 2 } forcingType;
static const char *const forcing_types[]          = {"none", "constant", "mms", "forcingType", "FORCE_", 0};
static const char *const forcing_types_for_disp[] = {"None", "Constant", "Manufactured solution"};

// Multigrid options
typedef enum { MULTIGRID_LOGARITHMIC = 0, MULTIGRID_UNIFORM = 1, MULTIGRID_NONE = 2 } multigridType;
static const char *const multigrid_types[]          = {"logarithmic", "uniform", "none", "multigridType", "MULTIGRID", 0};
static const char *const multigrid_types_for_disp[] = {"P-multigrid, logarithmic coarsening", "P-multigrind, uniform coarsening", "No multigrid"};

// -----------------------------------------------------------------------------
// Application data structs
// -----------------------------------------------------------------------------
// Units
typedef struct Units_ *Units;
struct Units_ {
  // Fundamental units
  PetscScalar meter;
  PetscScalar kilogram;
  PetscScalar second;
  // Derived unit
  PetscScalar Pascal;
};

// Application context from user command line options
typedef struct AppCtx_ *AppCtx;
struct AppCtx_ {
  const char   *name, *name_for_disp;               // problem name
  char          ceed_resource[PETSC_MAX_PATH_LEN];  // libCEED backend
  char          mesh_file[PETSC_MAX_PATH_LEN];      // exodusII mesh file
  char          output_dir[PETSC_MAX_PATH_LEN];
  PetscBool     test_mode;
  PetscBool     view_soln;
  PetscBool     view_final_soln;
  PetscViewer   energy_viewer;
  problemType   problem_choice;
  forcingType   forcing_choice;
  multigridType multigrid_choice;
  PetscInt      degree;
  PetscInt      q_extra;
  PetscInt      num_levels;
  PetscInt     *level_degrees;
  PetscInt      num_increments;  // Number of steps
  PetscInt      bc_clamp_count;
  PetscInt      bc_clamp_faces[16];
  // [translation; 3] [rotation axis; 3] [rotation magnitude c_0, c_1]
  // The rotations are (c_0 + c_1 s) \pi, where s = x · axis
  PetscScalar bc_clamp_max[16][8];
  PetscInt    bc_traction_count;
  PetscInt    bc_traction_faces[16];
  PetscScalar bc_traction_vector[16][3];
  PetscScalar forcing_vector[3];
  PetscReal   test_tol;
  PetscReal   expect_final_strain;
};

// Forcing function data
typedef struct {
  CeedQFunctionUser setup_forcing;
  const char       *setup_forcing_loc;
} forcingData;

extern forcingData forcing_options[3];

// Data for PETSc Matshell
typedef struct UserMult_ *UserMult;
struct UserMult_ {
  MPI_Comm             comm;
  DM                   dm;
  Vec                  X_loc, Y_loc, neumann_bcs;
  CeedVector           x_ceed, y_ceed;
  CeedOperator         op;
  CeedQFunction        qf;
  Ceed                 ceed;
  PetscScalar          load_increment;
  CeedQFunctionContext ctx_phys, ctx_phys_smoother;
};

// Data for Jacobian setup routine
typedef struct FormJacobCtx_ *FormJacobCtx;
struct FormJacobCtx_ {
  UserMult    *jacob_ctx;
  PetscInt     num_levels;
  Mat         *jacob_mat, jacob_mat_coarse;
  CeedVector   coo_values;
  CeedOperator op_coarse;
};

// Data for PETSc Prolongation/Restriction Matshell
typedef struct UserMultProlongRestr_ *UserMultProlongRestr;
struct UserMultProlongRestr_ {
  MPI_Comm     comm;
  DM           dm_c, dm_f;
  Vec          loc_vec_c, loc_vec_f;
  CeedVector   ceed_vec_c, ceed_vec_f;
  CeedOperator op_prolong, op_restrict;
  Ceed         ceed;
};

#define SOLIDS_MAX_NUMBER_FIELDS 16

// libCEED data struct for level
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  Ceed                ceed;
  CeedBasis           basis_x, basis_u, basis_c_to_f, basis_energy, basis_diagnostic;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_geo_data_i, elem_restr_energy, elem_restr_diagnostic, elem_restr_geo_data_diagnostic_i,
      elem_restr_stored_fields_i[SOLIDS_MAX_NUMBER_FIELDS];
  CeedQFunction qf_residual, qf_jacobian, qf_energy, qf_diagnostic;
  CeedOperator  op_residual, op_jacobian, op_restrict, op_prolong, op_energy, op_diagnostic;
  CeedVector    geo_data, geo_data_diagnostic, x_ceed, y_ceed, true_soln, stored_fields[SOLIDS_MAX_NUMBER_FIELDS];
};

typedef struct {
  CeedQFunctionUser  setup_geo, residual, jacobian, energy, diagnostic, true_soln;
  const char        *setup_geo_loc, *residual_loc, *jacobian_loc, *energy_loc, *diagnostic_loc, *true_soln_loc;
  CeedQuadMode       quadrature_mode;
  CeedInt            q_data_size, number_fields_stored;
  CeedInt           *field_sizes;
  const char *const *field_names;
} ProblemData;

#endif  // libceed_solids_examples_structs_h
