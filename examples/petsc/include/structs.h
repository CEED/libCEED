// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Data structures for PETSc examples
#pragma once

#include <ceed.h>
#include <petsc.h>

// -----------------------------------------------------------------------------
// libCEED Data Structs
// -----------------------------------------------------------------------------

// libCEED data struct for level
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  Ceed                ceed;
  CeedBasis           basis_x, basis_u;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_u_i, elem_restr_qd_i;
  CeedQFunction       qf_apply;
  CeedOperator        op_apply, op_restrict, op_prolong;
  CeedVector          q_data, x_ceed, y_ceed;
  CeedInt             q_data_size;
};

// libCEED data struct for BDDC
typedef struct CeedDataBDDC_ *CeedDataBDDC;
struct CeedDataBDDC_ {
  CeedBasis           basis_Pi, basis_Pi_r;
  CeedInt             strides[3];
  CeedElemRestriction elem_restr_Pi, elem_restr_Pi_r, elem_restr_r;
  CeedOperator        op_Pi_r, op_r_Pi, op_Pi_Pi, op_r_r, op_r_r_inv, op_inject_Pi, op_inject_Pi_r, op_inject_r, op_restrict_Pi, op_restrict_Pi_r,
      op_restrict_r;
  CeedVector x_ceed, y_ceed, x_Pi_ceed, y_Pi_ceed, x_Pi_r_ceed, y_Pi_r_ceed, x_r_ceed, y_r_ceed, z_r_ceed, mult_ceed, mask_r_ceed, mask_Gamma_ceed,
      mask_I_ceed;
};

// BP specific data
typedef struct {
  CeedInt           num_comp_x, num_comp_u, topo_dim, q_data_size, q_extra;
  CeedQFunctionUser setup_geo, setup_rhs, apply, error;
  const char       *setup_geo_loc, *setup_rhs_loc, *apply_loc, *error_loc;
  CeedEvalMode      in_mode, out_mode;
  CeedQuadMode      q_mode;
  PetscBool         enforce_bc;
} BPData;

// BP options
typedef enum {
  CEED_BP1  = 0,
  CEED_BP2  = 1,
  CEED_BP3  = 2,
  CEED_BP4  = 3,
  CEED_BP5  = 4,
  CEED_BP6  = 5,
  CEED_BP13 = 6,
  CEED_BP24 = 7,
  CEED_BP15 = 8,
  CEED_BP26 = 9,
} BPType;

// -----------------------------------------------------------------------------
// Parameter structure for running problems
// -----------------------------------------------------------------------------
typedef struct RunParams_ *RunParams;
struct RunParams_ {
  MPI_Comm      comm;
  PetscBool     test_mode, read_mesh, user_l_nodes, write_solution, simplex;
  char         *filename, *hostname;
  PetscInt      local_nodes, degree, q_extra, dim, num_comp_u, *mesh_elem;
  PetscInt      ksp_max_it_clip[2];
  PetscMPIInt   ranks_per_node;
  BPType        bp_choice;
  PetscLogStage solve_stage;
};

// -----------------------------------------------------------------------------
// PETSc Operator Structs
// -----------------------------------------------------------------------------

// Data for PETSc Matshell
typedef struct OperatorApplyContext_ *OperatorApplyContext;
struct OperatorApplyContext_ {
  MPI_Comm     comm;
  DM           dm;
  Vec          X_loc, Y_loc, diag;
  CeedVector   x_ceed, y_ceed;
  CeedOperator op;
  Ceed         ceed;
};

// Data for PETSc Prolong/Restrict Matshells
typedef struct ProlongRestrContext_ *ProlongRestrContext;
struct ProlongRestrContext_ {
  MPI_Comm     comm;
  DM           dmc, dmf;
  Vec          loc_vec_c, loc_vec_f, mult_vec;
  CeedVector   ceed_vec_c, ceed_vec_f;
  CeedOperator op_prolong, op_restrict;
  Ceed         ceed;
};

// Data for PETSc PCshell
typedef struct BDDCApplyContext_ *BDDCApplyContext;
struct BDDCApplyContext_ {
  MPI_Comm     comm;
  DM           dm, dm_Pi;
  SNES         snes_Pi, snes_Pi_r;
  KSP          ksp_S_Pi, ksp_S_Pi_r;
  Mat          mat_S_Pi, mat_S_Pi_r;
  Vec          X_loc, Y_loc, X_Pi, Y_Pi, X_Pi_loc, Y_Pi_loc, X_Pi_r_loc, Y_Pi_r_loc;
  PetscBool    is_harmonic;
  CeedDataBDDC ceed_data_bddc;
};
