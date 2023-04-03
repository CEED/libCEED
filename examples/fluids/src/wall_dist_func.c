// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// General wall distance functions for Navier-Stokes example using PETSc
/// We do this by solving the Poisson equation ∇^{2} φ  = -1 with weak form ∫ ∇v ⋅ ∇ φ - v 1

#include "../qfunctions/wall_dist_func.h"

#include "../navierstokes.h"
#include "../qfunctions/newtonian_state.h"

// General distance functions
static PetscErrorCode Distance_Function_NS(DM dm, User user) {
  MPI_Comm            comm;
  PetscScalar        *r;
  DM                  dmDist;
  PetscFE             fe;
  PetscInt            xl_size, l_size, g_size, dim = 3;
  SNES                snesDist;
  Mat                 A;
  Vec                 X, Y, q_data, X_loc, rhs, rhs_loc;
  VecType             vec_type;
  PetscMemType        mem_type;
  Ceed                ceed;
  CeedInt             num_elem = 10, P = 3, Q = P + 1;
  CeedInt             num_nodes_x = num_elem + 1, num_nodes_phi = num_elem * (P - 1) + 1;
  CeedInt             ind_x[num_elem * 2], ind_phi[num_elem * P];
  CeedVector          rhs_ceed, X_ceed, Y_ceed, ones_vec;
  CeedOperator        op_distance_function;
  CeedQFunction       qf_distance_function;
  CeedBasis           basis_x, basis_phi;
  CeedElemRestriction elem_restr_x, elem_restr_phi;

  PetscFunctionBeginUser;
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, PETSC_DETERMINE, &fe));
  PetscBool distance_snes_monitor = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-distance_snes_monitor", &distance_snes_monitor));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snesDist));
  PetscObjectSetOptionsPrefix((PetscObject)snesDist, "distance_");
  PetscCall(DMClone(dm, &dmDist));
  PetscCall(DMGetVecType(dm, &vec_type));
  PetscCall(DMSetVecType(dmDist, vec_type));
  PetscCall(DMAddField(dmDist, NULL, (PetscObject)fe));

  // Create Vectors
  PetscCall(DMCreateGlobalVector(dmDist, &X));
  PetscCall(DMCreateGlobalVector(dmDist, &Y));
  PetscCall(VecGetLocalSize(X, &l_size));
  PetscCall(VecGetSize(X, &g_size));
  PetscCall(VecGetSize(q_data, &g_size));
  PetscCall(DMCreateLocalVector(dmDist, &X_loc));
  PetscCall(VecGetSize(X_loc, &xl_size));
  PetscCall(VecDuplicate(X, &rhs));

  // Create Matshell Operator
  PetscCall(MatCreateShell(comm, l_size, l_size, g_size, g_size, op_distance_function, &A));
  PetscCall(MatShellSetOperation(A, MATOP_MULT, (void (*)(void))MatMult_Ceed));
  PetscCall(MatShellSetVecType(A, vec_type));

  // Setup libCEED vector
  PetscCall(VecP2C(X, &mem_type, X_ceed));
  PetscCall(VecP2C(Y, &mem_type, Y_ceed));

  // Create RHS vector
  PetscCall(VecDuplicate(X_loc, &rhs_loc));
  PetscCall(VecZeroEntries(rhs_loc));
  PetscCall(VecGetArrayAndMemType(rhs_loc, &r, &mem_type));
  CeedVectorCreate(ceed, xl_size, &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  // Gather RHS
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(rhs_loc, &r));
  PetscCall(VecZeroEntries(rhs));
  PetscCall(DMLocalToGlobal(dmDist, rhs_loc, ADD_VALUES, rhs));
  CeedVectorDestroy(&rhs_ceed);

  // Create Element Restriction
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_nodes_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_x);
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_nodes_phi, CEED_MEM_HOST, CEED_USE_POINTER, ind_phi, &elem_restr_phi);

  // Create Basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &basis_phi);

  // Create and Add QFunction fields
  CeedQFunctionCreateInterior(ceed, dim, DistanceFunction, DistanceFunction_loc, &qf_distance_function);

  CeedQFunctionAddInput(qf_distance_function, "v", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_distance_function, "dphi", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_distance_function, "q_data", dim * (dim + 1) / 2, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_distance_function, "dv", dim, CEED_EVAL_GRAD);

  // Create Operator
  CeedOperatorCreate(ceed, qf_distance_function, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_distance_function);

  // Operator set
  CeedOperatorSetField(op_distance_function, "phi", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_distance_function, "v", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_distance_function, "output", elem_restr_phi, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedVectorCreate(ceed, num_nodes_phi, &X_ceed);
  CeedVectorSetValue(X_ceed, 0.0);
  CeedVectorCreate(ceed, num_nodes_phi, &Y_ceed);
  CeedVectorSetValue(Y_ceed, 0.0);
  CeedVectorCreate(ceed, num_nodes_phi, &ones_vec);
  CeedVectorSetValue(ones_vec, 1.0);  // vector of ones to multiply mass matrix with

  // Apply Operator
  CeedOperatorApply(op_distance_function, X_ceed, Y_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore PETSc vectors
  PetscCall(VecC2P(X_ceed, mem_type, X));
  PetscCall(VecC2P(Y_ceed, mem_type, Y));

  // Set up SNES
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snesDist));

  // Solve
  PetscCall(SNESSolve(snesDist, rhs, X));
  PetscCall(SNESGetSolution(snesDist, &X));

  // Clean up
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&rhs_loc));
  PetscCall(SNESDestroy(&snesDist));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&X_loc));
  PetscCall(VecDestroy(&q_data));
  PetscCall(MatDestroy(&A));
  CeedVectorDestroy(&ones_vec);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_phi);
  CeedQFunctionDestroy(&qf_distance_function);
  CeedOperatorDestroy(&op_distance_function);
  CeedDestroy(&ceed);
  PetscFunctionReturn(0);
}
