// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                        libCEED + PETSc Example: CEED BPs
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps.
//
// The code is intentionally "raw", using only low-level communication
// primitives.
//
// Build with:
//
//     make bpsraw [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     ./bpsraw -problem bp1
//     ./bpsraw -problem bp2
//     ./bpsraw -problem bp3
//     ./bpsraw -problem bp4
//     ./bpsraw -problem bp5 -ceed /cpu/self
//     ./bpsraw -problem bp6 -ceed /gpu/cuda
//
//TESTARGS -ceed {ceed_resource} -test -problem bp2 -degree 5 -q_extra 1 -ksp_max_it_clip 15,15

/// @file
/// CEED BPs example using PETSc
/// See bps.c for an implementation using DMPlex unstructured grids.
const char help[] = "Solve CEED BPs using PETSc\n";

#include <ceed.h>
#include <petscksp.h>
#include <petscsys.h>
#include <stdbool.h>
#include <string.h>

#include "qfunctions/bps/bp1.h"
#include "qfunctions/bps/bp2.h"
#include "qfunctions/bps/bp3.h"
#include "qfunctions/bps/bp4.h"
#include "qfunctions/bps/common.h"

#if PETSC_VERSION_LT(3, 12, 0)
#ifdef PETSC_HAVE_CUDA
#include <petsccuda.h>
// Note: With PETSc prior to version 3.12.0, providing the source path to
//       include 'cublas_v2.h' will be needed to use 'petsccuda.h'.
#endif
#endif

static CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

static void Split3(PetscInt size, PetscInt m[3], bool reverse) {
  for (PetscInt d = 0, size_left = size; d < 3; d++) {
    PetscInt try = (PetscInt)PetscCeilReal(PetscPowReal(size_left, 1. / (3 - d)));
    while (try * (size_left / try) != size_left) try++;
    m[reverse ? 2 - d : d] = try;
    size_left /= try;
  }
}

static PetscInt Max3(const PetscInt a[3]) { return PetscMax(a[0], PetscMax(a[1], a[2])); }
static PetscInt Min3(const PetscInt a[3]) { return PetscMin(a[0], PetscMin(a[1], a[2])); }
static void     GlobalNodes(const PetscInt p[3], const PetscInt i_rank[3], PetscInt degree, const PetscInt mesh_elem[3], PetscInt m_nodes[3]) {
      for (int d = 0; d < 3; d++) m_nodes[d] = degree * mesh_elem[d] + (i_rank[d] == p[d] - 1);
}
static PetscInt GlobalStart(const PetscInt p[3], const PetscInt i_rank[3], PetscInt degree, const PetscInt mesh_elem[3]) {
  PetscInt start = 0;
  // Dumb brute-force is easier to read
  for (PetscInt i = 0; i < p[0]; i++) {
    for (PetscInt j = 0; j < p[1]; j++) {
      for (PetscInt k = 0; k < p[2]; k++) {
        PetscInt m_nodes[3], ijk_rank[] = {i, j, k};
        if (i == i_rank[0] && j == i_rank[1] && k == i_rank[2]) return start;
        GlobalNodes(p, ijk_rank, degree, mesh_elem, m_nodes);
        start += m_nodes[0] * m_nodes[1] * m_nodes[2];
      }
    }
  }
  return -1;
}
static PetscErrorCode CreateRestriction(Ceed ceed, const CeedInt mesh_elem[3], CeedInt P, CeedInt num_comp, CeedElemRestriction *elem_restr) {
  const PetscInt num_elem = mesh_elem[0] * mesh_elem[1] * mesh_elem[2];
  PetscInt       m_nodes[3], *idx, *idx_p;

  PetscFunctionBeginUser;
  // Get indicies
  for (int d = 0; d < 3; d++) m_nodes[d] = mesh_elem[d] * (P - 1) + 1;
  idx_p = idx = malloc(num_elem * P * P * P * sizeof idx[0]);
  for (CeedInt i = 0; i < mesh_elem[0]; i++) {
    for (CeedInt j = 0; j < mesh_elem[1]; j++) {
      for (CeedInt k = 0; k < mesh_elem[2]; k++, idx_p += P * P * P) {
        for (CeedInt ii = 0; ii < P; ii++) {
          for (CeedInt jj = 0; jj < P; jj++) {
            for (CeedInt kk = 0; kk < P; kk++) {
              if (0) {  // This is the C-style (i,j,k) ordering that I prefer
                idx_p[(ii * P + jj) * P + kk] = num_comp * (((i * (P - 1) + ii) * m_nodes[1] + (j * (P - 1) + jj)) * m_nodes[2] + (k * (P - 1) + kk));
              } else {  // (k,j,i) ordering for consistency with MFEM example
                idx_p[ii + P * (jj + P * kk)] = num_comp * (((i * (P - 1) + ii) * m_nodes[1] + (j * (P - 1) + jj)) * m_nodes[2] + (k * (P - 1) + kk));
              }
            }
          }
        }
      }
    }
  }

  // Setup CEED restriction
  CeedElemRestrictionCreate(ceed, num_elem, P * P * P, num_comp, 1, m_nodes[0] * m_nodes[1] * m_nodes[2] * num_comp, CEED_MEM_HOST, CEED_OWN_POINTER,
                            idx, elem_restr);

  PetscFunctionReturn(0);
}

// Data for PETSc
typedef struct OperatorApplyContext_ *OperatorApplyContext;
struct OperatorApplyContext_ {
  MPI_Comm     comm;
  VecScatter   l_to_g;    // Scatter for all entries
  VecScatter   l_to_g_0;  // Skip Dirichlet values
  VecScatter   g_to_g_D;  // global-to-global; only Dirichlet values
  Vec          X_loc, Y_loc;
  CeedVector   x_ceed, y_ceed;
  CeedOperator op;
  CeedVector   q_data;
  Ceed         ceed;
};

// BP Options
typedef enum { CEED_BP1 = 0, CEED_BP2 = 1, CEED_BP3 = 2, CEED_BP4 = 3, CEED_BP5 = 4, CEED_BP6 = 5 } BPType;
static const char *const bp_types[] = {"bp1", "bp2", "bp3", "bp4", "bp5", "bp6", "BPType", "CEED_BP", 0};

// BP specific data
typedef struct {
  CeedInt           num_comp_u, q_data_size, q_extra;
  CeedQFunctionUser setup_geo, setup_rhs, apply, error;
  const char       *setup_geo_loc, *setup_rhs_loc, *apply_loc, *error_loc;
  CeedEvalMode      in_mode, out_mode;
  CeedQuadMode      q_mode;
} BPData;

BPData bp_options[6] = {
    [CEED_BP1] = {.num_comp_u    = 1,
                  .q_data_size   = 1,
                  .q_extra       = 1,
                  .setup_geo     = SetupMassGeo,
                  .setup_rhs     = SetupMassRhs,
                  .apply         = Mass,
                  .error         = Error,
                  .setup_geo_loc = SetupMassGeo_loc,
                  .setup_rhs_loc = SetupMassRhs_loc,
                  .apply_loc     = Mass_loc,
                  .error_loc     = Error_loc,
                  .in_mode       = CEED_EVAL_INTERP,
                  .out_mode      = CEED_EVAL_INTERP,
                  .q_mode        = CEED_GAUSS        },
    [CEED_BP2] = {.num_comp_u    = 3,
                  .q_data_size   = 1,
                  .q_extra       = 1,
                  .setup_geo     = SetupMassGeo,
                  .setup_rhs     = SetupMassRhs3,
                  .apply         = Mass3,
                  .error         = Error3,
                  .setup_geo_loc = SetupMassGeo_loc,
                  .setup_rhs_loc = SetupMassRhs3_loc,
                  .apply_loc     = Mass3_loc,
                  .error_loc     = Error3_loc,
                  .in_mode       = CEED_EVAL_INTERP,
                  .out_mode      = CEED_EVAL_INTERP,
                  .q_mode        = CEED_GAUSS        },
    [CEED_BP3] = {.num_comp_u    = 1,
                  .q_data_size   = 7,
                  .q_extra       = 1,
                  .setup_geo     = SetupDiffGeo,
                  .setup_rhs     = SetupDiffRhs,
                  .apply         = Diff,
                  .error         = Error,
                  .setup_geo_loc = SetupDiffGeo_loc,
                  .setup_rhs_loc = SetupDiffRhs_loc,
                  .apply_loc     = Diff_loc,
                  .error_loc     = Error_loc,
                  .in_mode       = CEED_EVAL_GRAD,
                  .out_mode      = CEED_EVAL_GRAD,
                  .q_mode        = CEED_GAUSS        },
    [CEED_BP4] = {.num_comp_u    = 3,
                  .q_data_size   = 7,
                  .q_extra       = 1,
                  .setup_geo     = SetupDiffGeo,
                  .setup_rhs     = SetupDiffRhs3,
                  .apply         = Diff3,
                  .error         = Error3,
                  .setup_geo_loc = SetupDiffGeo_loc,
                  .setup_rhs_loc = SetupDiffRhs3_loc,
                  .apply_loc     = Diff3_loc,
                  .error_loc     = Error3_loc,
                  .in_mode       = CEED_EVAL_GRAD,
                  .out_mode      = CEED_EVAL_GRAD,
                  .q_mode        = CEED_GAUSS        },
    [CEED_BP5] = {.num_comp_u    = 1,
                  .q_data_size   = 7,
                  .q_extra       = 0,
                  .setup_geo     = SetupDiffGeo,
                  .setup_rhs     = SetupDiffRhs,
                  .apply         = Diff,
                  .error         = Error,
                  .setup_geo_loc = SetupDiffGeo_loc,
                  .setup_rhs_loc = SetupDiffRhs_loc,
                  .apply_loc     = Diff_loc,
                  .error_loc     = Error_loc,
                  .in_mode       = CEED_EVAL_GRAD,
                  .out_mode      = CEED_EVAL_GRAD,
                  .q_mode        = CEED_GAUSS_LOBATTO},
    [CEED_BP6] = {.num_comp_u    = 3,
                  .q_data_size   = 7,
                  .q_extra       = 0,
                  .setup_geo     = SetupDiffGeo,
                  .setup_rhs     = SetupDiffRhs3,
                  .apply         = Diff3,
                  .error         = Error3,
                  .setup_geo_loc = SetupDiffGeo_loc,
                  .setup_rhs_loc = SetupDiffRhs3_loc,
                  .apply_loc     = Diff3_loc,
                  .error_loc     = Error3_loc,
                  .in_mode       = CEED_EVAL_GRAD,
                  .out_mode      = CEED_EVAL_GRAD,
                  .q_mode        = CEED_GAUSS_LOBATTO}
};

// This function uses libCEED to compute the action of the mass matrix
static PetscErrorCode MatMult_Mass(Mat A, Vec X, Vec Y) {
  OperatorApplyContext op_apply_ctx;
  PetscScalar         *x, *y;
  PetscMemType         x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &op_apply_ctx));

  // Global-to-local
  PetscCall(VecScatterBegin(op_apply_ctx->l_to_g, X, op_apply_ctx->X_loc, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(op_apply_ctx->l_to_g, X, op_apply_ctx->X_loc, INSERT_VALUES, SCATTER_REVERSE));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x, &x_mem_type));
  PetscCall(VecGetArrayAndMemType(op_apply_ctx->Y_loc, &y, &y_mem_type));
  CeedVectorSetArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(op_apply_ctx->op, op_apply_ctx->x_ceed, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), NULL);
  CeedVectorTakeArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x));
  PetscCall(VecRestoreArrayAndMemType(op_apply_ctx->Y_loc, &y));

  // Local-to-global
  if (Y) {
    PetscCall(VecZeroEntries(Y));
    PetscCall(VecScatterBegin(op_apply_ctx->l_to_g, op_apply_ctx->Y_loc, Y, ADD_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(op_apply_ctx->l_to_g, op_apply_ctx->Y_loc, Y, ADD_VALUES, SCATTER_FORWARD));
  }
  PetscFunctionReturn(0);
}

// This function uses libCEED to compute the action of the Laplacian with
// Dirichlet boundary conditions
static PetscErrorCode MatMult_Diff(Mat A, Vec X, Vec Y) {
  OperatorApplyContext op_apply_ctx;
  PetscScalar         *x, *y;
  PetscMemType         x_mem_type, y_mem_type;

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(A, &op_apply_ctx));

  // Global-to-local
  PetscCall(VecScatterBegin(op_apply_ctx->l_to_g_0, X, op_apply_ctx->X_loc, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(op_apply_ctx->l_to_g_0, X, op_apply_ctx->X_loc, INSERT_VALUES, SCATTER_REVERSE));

  // Setup libCEED vectors
  PetscCall(VecGetArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x, &x_mem_type));
  PetscCall(VecGetArrayAndMemType(op_apply_ctx->Y_loc, &y, &y_mem_type));
  CeedVectorSetArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), CEED_USE_POINTER, x);
  CeedVectorSetArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), CEED_USE_POINTER, y);

  // Apply libCEED operator
  CeedOperatorApply(op_apply_ctx->op, op_apply_ctx->x_ceed, op_apply_ctx->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  CeedVectorTakeArray(op_apply_ctx->x_ceed, MemTypeP2C(x_mem_type), NULL);
  CeedVectorTakeArray(op_apply_ctx->y_ceed, MemTypeP2C(y_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x));
  PetscCall(VecRestoreArrayAndMemType(op_apply_ctx->Y_loc, &y));

  // Local-to-global
  PetscCall(VecZeroEntries(Y));
  PetscCall(VecScatterBegin(op_apply_ctx->g_to_g_D, X, Y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(op_apply_ctx->g_to_g_D, X, Y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(op_apply_ctx->l_to_g_0, op_apply_ctx->Y_loc, Y, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(op_apply_ctx->l_to_g_0, op_apply_ctx->Y_loc, Y, ADD_VALUES, SCATTER_FORWARD));

  PetscFunctionReturn(0);
}

// This function calculates the error in the final solution
static PetscErrorCode ComputeErrorMax(OperatorApplyContext op_apply_ctx, CeedOperator op_error, Vec X, CeedVector target, PetscReal *max_error) {
  PetscScalar *x;
  PetscMemType mem_type;
  CeedVector   collocated_error;
  CeedSize     length;

  PetscFunctionBeginUser;

  CeedVectorGetLength(target, &length);
  CeedVectorCreate(op_apply_ctx->ceed, length, &collocated_error);

  // Global-to-local
  PetscCall(VecScatterBegin(op_apply_ctx->l_to_g, X, op_apply_ctx->X_loc, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(op_apply_ctx->l_to_g, X, op_apply_ctx->X_loc, INSERT_VALUES, SCATTER_REVERSE));

  // Setup libCEED vector
  PetscCall(VecGetArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x, &mem_type));
  CeedVectorSetArray(op_apply_ctx->x_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, x);

  // Apply libCEED operator
  CeedOperatorApply(op_error, op_apply_ctx->x_ceed, collocated_error, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vector
  CeedVectorTakeArray(op_apply_ctx->x_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(op_apply_ctx->X_loc, (const PetscScalar **)&x));

  // Reduce max error
  *max_error = 0;
  const CeedScalar *e;
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &e);
  for (CeedInt i = 0; i < length; i++) {
    *max_error = PetscMax(*max_error, PetscAbsScalar(e[i]));
  }
  CeedVectorRestoreArrayRead(collocated_error, &e);
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, max_error, 1, MPIU_REAL, MPIU_MAX, op_apply_ctx->comm));

  // Cleanup
  CeedVectorDestroy(&collocated_error);

  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  MPI_Comm comm;
  char     ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  double   my_rt_start, my_rt, rt_min, rt_max;
  PetscInt degree, q_extra, local_nodes, local_elem, mesh_elem[3], m_nodes[3], p[3], i_rank[3], l_nodes[3], l_size, num_comp_u = 1,
                                                                                                                    ksp_max_it_clip[2];
  PetscScalar         *r;
  PetscBool            test_mode, benchmark_mode, write_solution;
  PetscMPIInt          size, rank;
  PetscLogStage        solve_stage;
  Vec                  X, X_loc, rhs, rhs_loc;
  Mat                  mat;
  KSP                  ksp;
  VecScatter           l_to_g, l_to_g_0, g_to_g_D;
  PetscMemType         mem_type;
  OperatorApplyContext op_apply_ctx;
  Ceed                 ceed;
  CeedBasis            basis_x, basis_u;
  CeedElemRestriction  elem_restr_x, elem_restr_u, elem_restr_u_i, elem_restr_qd_i;
  CeedQFunction        qf_setup_geo, qf_setup_rhs, qf_apply, qf_error;
  CeedOperator         op_setup_geo, op_setup_rhs, op_apply, op_error;
  CeedVector           x_coord, q_data, rhs_ceed, target;
  CeedInt              P, Q;
  const CeedInt        dim = 3, num_comp_x = 3;
  BPType               bp_choice;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL);
  bp_choice = CEED_BP1;
  PetscCall(PetscOptionsEnum("-problem", "CEED benchmark problem to solve", NULL, bp_types, (PetscEnum)bp_choice, (PetscEnum *)&bp_choice, NULL));
  num_comp_u = bp_options[bp_choice].num_comp_u;
  test_mode  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, test_mode, &test_mode, NULL));
  benchmark_mode = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-benchmark", "Benchmarking mode (prints benchmark statistics)", NULL, benchmark_mode, &benchmark_mode, NULL));
  write_solution = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-write_solution", "Write solution for visualization", NULL, write_solution, &write_solution, NULL));
  degree = test_mode ? 3 : 1;
  PetscCall(PetscOptionsInt("-degree", "Polynomial degree of tensor product basis", NULL, degree, &degree, NULL));
  q_extra = bp_options[bp_choice].q_extra;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, q_extra, &q_extra, NULL));
  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, ceed_resource, ceed_resource, sizeof(ceed_resource), NULL));
  local_nodes = 1000;
  PetscCall(PetscOptionsInt("-local", "Target number of locally owned nodes per process", NULL, local_nodes, &local_nodes, NULL));
  PetscInt two       = 2;
  ksp_max_it_clip[0] = 5;
  ksp_max_it_clip[1] = 20;
  PetscCall(
      PetscOptionsIntArray("-ksp_max_it_clip", "Min and max number of iterations to use during benchmarking", NULL, ksp_max_it_clip, &two, NULL));
  PetscOptionsEnd();
  P = degree + 1;
  Q = P + q_extra;

  // Set up libCEED
  CeedInit(ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  VecType default_vec_type = NULL, vec_type;
  switch (mem_type_backend) {
    case CEED_MEM_HOST:
      default_vec_type = VECSTANDARD;
      break;
    case CEED_MEM_DEVICE: {
      const char *resolved;
      CeedGetResource(ceed, &resolved);
      if (strstr(resolved, "/gpu/cuda")) default_vec_type = VECCUDA;
      else if (strstr(resolved, "/gpu/hip/occa")) default_vec_type = VECSTANDARD;  // https://github.com/CEED/libCEED/issues/678
      else if (strstr(resolved, "/gpu/hip")) default_vec_type = VECHIP;
      else default_vec_type = VECSTANDARD;
    }
  }

  // Determine size of process grid
  PetscCall(MPI_Comm_size(comm, &size));
  Split3(size, p, false);

  // Find a nicely composite number of elements no less than local_nodes
  for (local_elem = PetscMax(1, local_nodes / (degree * degree * degree));; local_elem++) {
    Split3(local_elem, mesh_elem, true);
    if (Max3(mesh_elem) / Min3(mesh_elem) <= 2) break;
  }

  // Find my location in the process grid
  PetscCall(MPI_Comm_rank(comm, &rank));
  for (int d = 0, rank_left = rank; d < dim; d++) {
    const int pstride[3] = {p[1] * p[2], p[2], 1};
    i_rank[d]            = rank_left / pstride[d];
    rank_left -= i_rank[d] * pstride[d];
  }

  GlobalNodes(p, i_rank, degree, mesh_elem, m_nodes);

  // Setup global vector
  PetscCall(VecCreate(comm, &X));
  PetscCall(VecSetType(X, default_vec_type));
  PetscCall(VecSetSizes(X, m_nodes[0] * m_nodes[1] * m_nodes[2] * num_comp_u, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(X));
  PetscCall(VecSetUp(X));

  // Set up libCEED
  CeedInit(ceed_resource, &ceed);

  // Print summary
  CeedInt gsize;
  PetscCall(VecGetSize(X, &gsize));
  if (!test_mode) {
    const char *used_resource;
    CeedGetResource(ceed, &used_resource);

    PetscCall(VecGetType(X, &vec_type));

    PetscCall(PetscPrintf(comm,
                          "\n-- CEED Benchmark Problem %" CeedInt_FMT " -- libCEED + PETSc --\n"
                          "  PETSc:\n"
                          "    PETSc Vec Type                     : %s\n"
                          "  libCEED:\n"
                          "    libCEED Backend                    : %s\n"
                          "    libCEED Backend MemType            : %s\n"
                          "  Mesh:\n"
                          "    Solution Order (P)                 : %" CeedInt_FMT "\n"
                          "    Quadrature  Order (Q)              : %" CeedInt_FMT "\n"
                          "    Global nodes                       : %" PetscInt_FMT "\n"
                          "    Process Decomposition              : %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n"
                          "    Local Elements                     : %" PetscInt_FMT " = %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n"
                          "    Owned nodes                        : %" PetscInt_FMT " = %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n"
                          "    DoF per node                       : %" PetscInt_FMT "\n",
                          bp_choice + 1, vec_type, used_resource, CeedMemTypes[mem_type_backend], P, Q, gsize / num_comp_u, p[0], p[1], p[2],
                          local_elem, mesh_elem[0], mesh_elem[1], mesh_elem[2], m_nodes[0] * m_nodes[1] * m_nodes[2], m_nodes[0], m_nodes[1],
                          m_nodes[2], num_comp_u));
  }

  {
    l_size = 1;
    for (int d = 0; d < dim; d++) {
      l_nodes[d] = mesh_elem[d] * degree + 1;
      l_size *= l_nodes[d];
    }
    PetscCall(VecCreate(PETSC_COMM_SELF, &X_loc));
    PetscCall(VecSetType(X_loc, default_vec_type));
    PetscCall(VecSetSizes(X_loc, l_size * num_comp_u, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(X_loc));
    PetscCall(VecSetUp(X_loc));

    // Create local-to-global scatter
    PetscInt *l_to_g_ind, *l_to_g_ind_0, *loc_ind, l_0_count;
    IS        l_to_g_is, l_to_g_is_0, loc_is;
    PetscInt  g_start[2][2][2], g_m_nodes[2][2][2][dim];

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          PetscInt ijk_rank[3] = {i_rank[0] + i, i_rank[1] + j, i_rank[2] + k};
          g_start[i][j][k]     = GlobalStart(p, ijk_rank, degree, mesh_elem);
          GlobalNodes(p, ijk_rank, degree, mesh_elem, g_m_nodes[i][j][k]);
        }
      }
    }

    PetscCall(PetscMalloc1(l_size, &l_to_g_ind));
    PetscCall(PetscMalloc1(l_size, &l_to_g_ind_0));
    PetscCall(PetscMalloc1(l_size, &loc_ind));
    l_0_count = 0;
    for (PetscInt i = 0, ir, ii; ir = i >= m_nodes[0], ii = i - ir * m_nodes[0], i < l_nodes[0]; i++) {
      for (PetscInt j = 0, jr, jj; jr = j >= m_nodes[1], jj = j - jr * m_nodes[1], j < l_nodes[1]; j++) {
        for (PetscInt k = 0, kr, kk; kr = k >= m_nodes[2], kk = k - kr * m_nodes[2], k < l_nodes[2]; k++) {
          PetscInt here    = (i * l_nodes[1] + j) * l_nodes[2] + k;
          l_to_g_ind[here] = g_start[ir][jr][kr] + (ii * g_m_nodes[ir][jr][kr][1] + jj) * g_m_nodes[ir][jr][kr][2] + kk;
          if ((i_rank[0] == 0 && i == 0) || (i_rank[1] == 0 && j == 0) || (i_rank[2] == 0 && k == 0) ||
              (i_rank[0] + 1 == p[0] && i + 1 == l_nodes[0]) || (i_rank[1] + 1 == p[1] && j + 1 == l_nodes[1]) ||
              (i_rank[2] + 1 == p[2] && k + 1 == l_nodes[2]))
            continue;
          l_to_g_ind_0[l_0_count] = l_to_g_ind[here];
          loc_ind[l_0_count++]    = here;
        }
      }
    }
    PetscCall(ISCreateBlock(comm, num_comp_u, l_size, l_to_g_ind, PETSC_OWN_POINTER, &l_to_g_is));
    PetscCall(VecScatterCreate(X_loc, NULL, X, l_to_g_is, &l_to_g));
    PetscCall(ISCreateBlock(comm, num_comp_u, l_0_count, l_to_g_ind_0, PETSC_OWN_POINTER, &l_to_g_is_0));
    PetscCall(ISCreateBlock(comm, num_comp_u, l_0_count, loc_ind, PETSC_OWN_POINTER, &loc_is));
    PetscCall(VecScatterCreate(X_loc, loc_is, X, l_to_g_is_0, &l_to_g_0));
    {
      // Create global-to-global scatter for Dirichlet values (everything not in
      // l_to_g_is_0, which is the range of l_to_g_0)
      PetscInt           x_start, x_end, *ind_D, count_D = 0;
      IS                 is_D;
      const PetscScalar *x;
      PetscCall(VecZeroEntries(X_loc));
      PetscCall(VecSet(X, 1.0));
      PetscCall(VecScatterBegin(l_to_g_0, X_loc, X, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(l_to_g_0, X_loc, X, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecGetOwnershipRange(X, &x_start, &x_end));
      PetscCall(PetscMalloc1(x_end - x_start, &ind_D));
      PetscCall(VecGetArrayRead(X, &x));
      for (PetscInt i = 0; i < x_end - x_start; i++) {
        if (x[i] == 1.) ind_D[count_D++] = x_start + i;
      }
      PetscCall(VecRestoreArrayRead(X, &x));
      PetscCall(ISCreateGeneral(comm, count_D, ind_D, PETSC_COPY_VALUES, &is_D));
      PetscCall(PetscFree(ind_D));
      PetscCall(VecScatterCreate(X, is_D, X, is_D, &g_to_g_D));
      PetscCall(ISDestroy(&is_D));
    }
    PetscCall(ISDestroy(&l_to_g_is));
    PetscCall(ISDestroy(&l_to_g_is_0));
    PetscCall(ISDestroy(&loc_is));
  }

  // CEED bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_u, P, Q, bp_options[bp_choice].q_mode, &basis_u);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q, bp_options[bp_choice].q_mode, &basis_x);

  // CEED restrictions
  CreateRestriction(ceed, mesh_elem, P, num_comp_u, &elem_restr_u);
  CreateRestriction(ceed, mesh_elem, 2, dim, &elem_restr_x);
  CeedInt num_elem = mesh_elem[0] * mesh_elem[1] * mesh_elem[2];
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * Q * Q, num_comp_u, num_comp_u * num_elem * Q * Q * Q, CEED_STRIDES_BACKEND, &elem_restr_u_i);
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * Q * Q, bp_options[bp_choice].q_data_size,
                                   bp_options[bp_choice].q_data_size * num_elem * Q * Q * Q, CEED_STRIDES_BACKEND, &elem_restr_qd_i);
  {
    CeedScalar *x_loc;
    CeedInt     shape[3] = {mesh_elem[0] + 1, mesh_elem[1] + 1, mesh_elem[2] + 1}, len = shape[0] * shape[1] * shape[2];
    x_loc = malloc(len * num_comp_x * sizeof x_loc[0]);
    for (CeedInt i = 0; i < shape[0]; i++) {
      for (CeedInt j = 0; j < shape[1]; j++) {
        for (CeedInt k = 0; k < shape[2]; k++) {
          x_loc[dim * ((i * shape[1] + j) * shape[2] + k) + 0] = 1. * (i_rank[0] * mesh_elem[0] + i) / (p[0] * mesh_elem[0]);
          x_loc[dim * ((i * shape[1] + j) * shape[2] + k) + 1] = 1. * (i_rank[1] * mesh_elem[1] + j) / (p[1] * mesh_elem[1]);
          x_loc[dim * ((i * shape[1] + j) * shape[2] + k) + 2] = 1. * (i_rank[2] * mesh_elem[2] + k) / (p[2] * mesh_elem[2]);
        }
      }
    }
    CeedVectorCreate(ceed, len * num_comp_x, &x_coord);
    CeedVectorSetArray(x_coord, CEED_MEM_HOST, CEED_OWN_POINTER, x_loc);
  }

  // Create the QFunction that builds the operator quadrature data
  CeedQFunctionCreateInterior(ceed, 1, bp_options[bp_choice].setup_geo, bp_options[bp_choice].setup_geo_loc, &qf_setup_geo);
  CeedQFunctionAddInput(qf_setup_geo, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup_geo, "dx", num_comp_x * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_geo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_geo, "q_data", bp_options[bp_choice].q_data_size, CEED_EVAL_NONE);

  // Create the QFunction that sets up the RHS and true solution
  CeedQFunctionCreateInterior(ceed, 1, bp_options[bp_choice].setup_rhs, bp_options[bp_choice].setup_rhs_loc, &qf_setup_rhs);
  CeedQFunctionAddInput(qf_setup_rhs, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup_rhs, "q_data", bp_options[bp_choice].q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup_rhs, "true_soln", num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup_rhs, "rhs", num_comp_u, CEED_EVAL_INTERP);

  // Set up PDE operator
  CeedQFunctionCreateInterior(ceed, 1, bp_options[bp_choice].apply, bp_options[bp_choice].apply_loc, &qf_apply);
  // Add inputs and outputs
  CeedInt in_scale  = bp_options[bp_choice].in_mode == CEED_EVAL_GRAD ? 3 : 1;
  CeedInt out_scale = bp_options[bp_choice].out_mode == CEED_EVAL_GRAD ? 3 : 1;
  CeedQFunctionAddInput(qf_apply, "u", num_comp_u * in_scale, bp_options[bp_choice].in_mode);
  CeedQFunctionAddInput(qf_apply, "q_data", bp_options[bp_choice].q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", num_comp_u * out_scale, bp_options[bp_choice].out_mode);

  // Create the error qfunction
  CeedQFunctionCreateInterior(ceed, 1, bp_options[bp_choice].error, bp_options[bp_choice].error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_error, "qdata", bp_options[bp_choice].q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", num_comp_u, CEED_EVAL_NONE);

  // Create the persistent vectors that will be needed in setup
  CeedInt num_qpts;
  CeedBasisGetNumQuadraturePoints(basis_u, &num_qpts);
  CeedVectorCreate(ceed, bp_options[bp_choice].q_data_size * num_elem * num_qpts, &q_data);
  CeedVectorCreate(ceed, num_elem * num_qpts * num_comp_u, &target);
  CeedVectorCreate(ceed, l_size * num_comp_u, &rhs_ceed);

  // Create the operator that builds the quadrature data for the ceed operator
  CeedOperatorCreate(ceed, qf_setup_geo, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_geo);
  CeedOperatorSetField(op_setup_geo, "x", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_geo, "q_data", elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the operator that builds the RHS and true solution
  CeedOperatorCreate(ceed, qf_setup_rhs, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_rhs);
  CeedOperatorSetField(op_setup_rhs, "x", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_rhs, "q_data", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_setup_rhs, "true_soln", elem_restr_u_i, CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_setup_rhs, "rhs", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  // Create the mass or diff operator
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);
  CeedOperatorSetField(op_apply, "u", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "q_data", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_apply, "v", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_error);
  CeedOperatorSetField(op_error, "u", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", elem_restr_u_i, CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_error, "error", elem_restr_u_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Set up Mat
  PetscCall(PetscMalloc1(1, &op_apply_ctx));
  op_apply_ctx->comm   = comm;
  op_apply_ctx->l_to_g = l_to_g;
  if (bp_choice != CEED_BP1 && bp_choice != CEED_BP2) {
    op_apply_ctx->l_to_g_0 = l_to_g_0;
    op_apply_ctx->g_to_g_D = g_to_g_D;
  }
  op_apply_ctx->X_loc = X_loc;
  PetscCall(VecDuplicate(X_loc, &op_apply_ctx->Y_loc));
  CeedVectorCreate(ceed, l_size * num_comp_u, &op_apply_ctx->x_ceed);
  CeedVectorCreate(ceed, l_size * num_comp_u, &op_apply_ctx->y_ceed);
  op_apply_ctx->op     = op_apply;
  op_apply_ctx->q_data = q_data;
  op_apply_ctx->ceed   = ceed;

  PetscCall(MatCreateShell(comm, m_nodes[0] * m_nodes[1] * m_nodes[2] * num_comp_u, m_nodes[0] * m_nodes[1] * m_nodes[2] * num_comp_u, PETSC_DECIDE,
                           PETSC_DECIDE, op_apply_ctx, &mat));
  if (bp_choice == CEED_BP1 || bp_choice == CEED_BP2) {
    PetscCall(MatShellSetOperation(mat, MATOP_MULT, (void (*)(void))MatMult_Mass));
  } else {
    PetscCall(MatShellSetOperation(mat, MATOP_MULT, (void (*)(void))MatMult_Diff));
  }
  PetscCall(VecGetType(X, &vec_type));
  PetscCall(MatShellSetVecType(mat, vec_type));

  // Get RHS vector
  PetscCall(VecDuplicate(X, &rhs));
  PetscCall(VecDuplicate(X_loc, &rhs_loc));
  PetscCall(VecZeroEntries(rhs_loc));
  PetscCall(VecGetArrayAndMemType(rhs_loc, &r, &mem_type));
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  // Setup q_data, rhs, and target
  CeedOperatorApply(op_setup_geo, x_coord, q_data, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_rhs, x_coord, rhs_ceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorDestroy(&x_coord);

  // Gather RHS
  PetscCall(CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL));
  PetscCall(VecRestoreArrayAndMemType(rhs_loc, &r));
  PetscCall(VecZeroEntries(rhs));
  PetscCall(VecScatterBegin(l_to_g, rhs_loc, rhs, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(l_to_g, rhs_loc, rhs, ADD_VALUES, SCATTER_FORWARD));
  CeedVectorDestroy(&rhs_ceed);

  PetscCall(KSPCreate(comm, &ksp));
  {
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    if (bp_choice == CEED_BP1 || bp_choice == CEED_BP2) {
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_ROWSUM));
    } else {
      PetscCall(PCSetType(pc, PCNONE));
    }
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  }
  PetscCall(KSPSetOperators(ksp, mat, mat));
  // First run's performance log is not considered for benchmarking purposes
  PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1));
  my_rt_start = MPI_Wtime();
  PetscCall(KSPSolve(ksp, rhs, X));
  my_rt = MPI_Wtime() - my_rt_start;
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, comm));
  // Set maxits based on first iteration timing
  if (my_rt > 0.02) {
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, ksp_max_it_clip[0]));
  } else {
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, ksp_max_it_clip[1]));
  }
  PetscCall(KSPSetFromOptions(ksp));

  // Timed solve
  PetscCall(VecZeroEntries(X));
  PetscCall(PetscBarrier((PetscObject)ksp));

  // -- Performance logging
  PetscCall(PetscLogStageRegister("Solve Stage", &solve_stage));
  PetscCall(PetscLogStagePush(solve_stage));

  // -- Solve
  my_rt_start = MPI_Wtime();
  PetscCall(KSPSolve(ksp, rhs, X));
  my_rt = MPI_Wtime() - my_rt_start;

  // -- Performance logging
  PetscCall(PetscLogStagePop());

  // Output results
  {
    KSPType            ksp_type;
    KSPConvergedReason reason;
    PetscReal          rnorm;
    PetscInt           its;
    PetscCall(KSPGetType(ksp, &ksp_type));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      PetscCall(PetscPrintf(comm,
                            "  KSP:\n"
                            "    KSP Type                           : %s\n"
                            "    KSP Convergence                    : %s\n"
                            "    Total KSP Iterations               : %" PetscInt_FMT "\n"
                            "    Final rnorm                        : %e\n",
                            ksp_type, KSPConvergedReasons[reason], its, (double)rnorm));
    }
    if (!test_mode) {
      PetscCall(PetscPrintf(comm, "  Performance:\n"));
    }
    {
      PetscReal max_error;
      PetscCall(ComputeErrorMax(op_apply_ctx, op_error, X, target, &max_error));
      PetscReal tol = 5e-2;
      if (!test_mode || max_error > tol) {
        PetscCall(MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, comm));
        PetscCall(MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, comm));
        PetscCall(PetscPrintf(comm,
                              "    Pointwise Error (max)              : %e\n"
                              "    CG Solve Time                      : %g (%g) sec\n",
                              (double)max_error, rt_max, rt_min));
      }
    }
    if (!test_mode) {
      PetscCall(
          PetscPrintf(comm, "    DoFs/Sec in CG                     : %g (%g) million\n", 1e-6 * gsize * its / rt_max, 1e-6 * gsize * its / rt_min));
    }
  }

  if (write_solution) {
    PetscViewer vtk_viewer_soln;

    PetscCall(PetscViewerCreate(comm, &vtk_viewer_soln));
    PetscCall(PetscViewerSetType(vtk_viewer_soln, PETSCVIEWERVTK));
    PetscCall(PetscViewerFileSetName(vtk_viewer_soln, "solution.vtu"));
    PetscCall(VecView(X, vtk_viewer_soln));
    PetscCall(PetscViewerDestroy(&vtk_viewer_soln));
  }

  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&rhs_loc));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&op_apply_ctx->X_loc));
  PetscCall(VecDestroy(&op_apply_ctx->Y_loc));
  PetscCall(VecScatterDestroy(&l_to_g));
  PetscCall(VecScatterDestroy(&l_to_g_0));
  PetscCall(VecScatterDestroy(&g_to_g_D));
  PetscCall(MatDestroy(&mat));
  PetscCall(KSPDestroy(&ksp));

  CeedVectorDestroy(&op_apply_ctx->x_ceed);
  CeedVectorDestroy(&op_apply_ctx->y_ceed);
  CeedVectorDestroy(&op_apply_ctx->q_data);
  CeedVectorDestroy(&target);
  CeedOperatorDestroy(&op_setup_geo);
  CeedOperatorDestroy(&op_setup_rhs);
  CeedOperatorDestroy(&op_apply);
  CeedOperatorDestroy(&op_error);
  CeedElemRestrictionDestroy(&elem_restr_u);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_u_i);
  CeedElemRestrictionDestroy(&elem_restr_qd_i);
  CeedQFunctionDestroy(&qf_setup_geo);
  CeedQFunctionDestroy(&qf_setup_rhs);
  CeedQFunctionDestroy(&qf_apply);
  CeedQFunctionDestroy(&qf_error);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedDestroy(&ceed);
  PetscCall(PetscFree(op_apply_ctx));
  return PetscFinalize();
}
