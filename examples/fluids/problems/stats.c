// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up statistics collection

#include "../navierstokes.h"
#include "../qfunctions/mass.h"

PetscErrorCode CreateStatsDM(User user, ProblemData *problem, PetscInt degree, SimpleBC bc) {
  DM dm                      = user->dm;
  user->stats.num_comp_stats = 6;
  PetscFunctionBeginUser;

  PetscCall(DMClone(user->dm, &user->stats.dm));

  {
    PetscFE fe;
    DMLabel label;

    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, problem->dim, user->stats.num_comp_stats, PETSC_FALSE, degree, PETSC_DECIDE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "Q"));
    PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(dm));
    PetscCall(DMGetLabel(dm, "Face Sets", &label));

    // // Set wall BCs
    // if (bc->num_wall > 0) {
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, bc->num_wall, bc->walls, 0, bc->num_comps, bc->wall_comps,
    //                           (void (*)(void))problem->bc, NULL, problem->bc_ctx, NULL));
    // }
    // // Set slip BCs in the x direction
    // if (bc->num_slip[0] > 0) {
    //   PetscInt comps[1] = {1};
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", label, bc->num_slip[0], bc->slips[0], 0, 1, comps, (void (*)(void))NULL, NULL,
    //                           problem->bc_ctx, NULL));
    // }
    // // Set slip BCs in the y direction
    // if (bc->num_slip[1] > 0) {
    //   PetscInt comps[1] = {2};
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", label, bc->num_slip[1], bc->slips[1], 0, 1, comps, (void (*)(void))NULL, NULL,
    //                           problem->bc_ctx, NULL));
    // }
    // // Set slip BCs in the z direction
    // if (bc->num_slip[2] > 0) {
    //   PetscInt comps[1] = {3};
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", label, bc->num_slip[2], bc->slips[2], 0, 1, comps, (void (*)(void))NULL, NULL,
    //                           problem->bc_ctx, NULL));
    // }

    PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
    PetscCall(PetscFEDestroy(&fe));
  }

  PetscSection section;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "Mean Velocity Products XX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "Mean Velocity Products YY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "Mean Velocity Products ZZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "Mean Velocity Products YZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "Mean Velocity Products XZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 5, "Mean Velocity Products XY"));

  PetscFunctionReturn(0);
}

// Compute mass matrix for statistics projection
PetscErrorCode ComputeLumpedMassMatrix(Ceed ceed, DM dm, CeedData ceed_data, Vec M) {
  Vec           M_loc;
  CeedQFunction qf_mass;
  CeedOperator  op_mass;
  CeedVector    m_ceed, ones_vec;
  CeedInt       num_comp_q, q_data_size;
  PetscFunctionBeginUser;

  // CEED Restriction
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &m_ceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &ones_vec, NULL);
  CeedVectorSetValue(ones_vec, 1.0);

  // CEED QFunction
  CeedQFunctionCreateInterior(ceed, 1, Mass, Mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "q", num_comp_q, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_mass, "v", num_comp_q, CEED_EVAL_INTERP);

  // CEED Operator
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_mass, "v", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);

  // Place PETSc vector in CEED vector
  CeedScalar  *m;
  PetscMemType m_mem_type;
  PetscCall(DMGetLocalVector(dm, &M_loc));
  PetscCall(VecGetArrayAndMemType(M_loc, (PetscScalar **)&m, &m_mem_type));
  CeedVectorSetArray(m_ceed, MemTypeP2C(m_mem_type), CEED_USE_POINTER, m);

  // Apply CEED Operator
  CeedOperatorApply(op_mass, ones_vec, m_ceed, CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  CeedVectorTakeArray(m_ceed, MemTypeP2C(m_mem_type), NULL);
  PetscCall(VecRestoreArrayReadAndMemType(M_loc, (const PetscScalar **)&m));

  // Local-to-Global
  PetscCall(VecZeroEntries(M));
  PetscCall(DMLocalToGlobal(dm, M_loc, ADD_VALUES, M));
  PetscCall(DMRestoreLocalVector(dm, &M_loc));

  // Invert diagonally lumped mass vector for RHS function
  PetscCall(VecReciprocal(M));

  // Cleanup
  CeedVectorDestroy(&ones_vec);
  CeedVectorDestroy(&m_ceed);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_mass);

  PetscFunctionReturn(0);
}

// -- Create CeedOperator for statistics collection
PetscErrorCode CreateStatsOperator(Ceed ceed, ProblemQFunctionSpec stats, CeedData ceed_data, User user, CeedInt dim, CeedInt P, CeedInt Q) {
  int       num_comp_q = 5;
  int       num_comp_x = 3;
  PetscInt  q_data_size_vol;
  CeedBasis basis_stats;

  PetscFunctionBeginUser;

  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_qd_i, &q_data_size_vol);
  CeedElemRestrictionGetNumComponents(ceed_data->elem_restr_q, &num_comp_q);

  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->stats.stats_ceedvec, NULL);

  CeedQFunction qf_stats;
  CeedQFunctionCreateInterior(ceed, 1, stats.qfunction, stats.qfunction_loc, &qf_stats);
  CeedQFunctionSetContext(qf_stats, stats.qfunction_context);
  CeedQFunctionContextDestroy(&stats.qfunction_context);
  CeedQFunctionAddInput(qf_stats, "q", num_comp_q, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_stats, "q_data", q_data_size_vol, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_stats, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_stats, "U_prod", user->stats.num_comp_stats, CEED_EVAL_INTERP);

  // -- CEED setup the basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, user->stats.num_comp_stats, P, Q, CEED_GAUSS, &basis_stats);

  // -- CEED operator for stats collection
  CeedOperator op;
  CeedOperatorCreate(ceed, qf_stats, NULL, NULL, &op);
  CeedOperatorSetField(op, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op, "q_data", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
  CeedOperatorSetField(op, "U_prod", ceed_data->elem_restr_q, basis_stats, CEED_VECTOR_ACTIVE);
  CeedQFunctionDestroy(&qf_stats);

  user->stats.op_stats = op;

  PetscFunctionReturn(0);
}
