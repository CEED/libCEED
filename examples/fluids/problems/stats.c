// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up statistics collection

#include "../navierstokes.h"

// -- Create QFunction for Reynolds stress
PetscErrorCode CreateStatsOperator(Ceed ceed, ProblemQFunctionSpec stats, CeedData ceed_data, User user, CeedInt dim, CeedInt P, CeedInt Q) {
  int       num_comp_q      = 5;
  int       q_data_size_vol = 10;
  int       num_comp_x      = 3;
  int       num_comp_stats  = 5;
  CeedBasis basis_stats;

  PetscFunctionBeginUser;

  CeedElemRestrictionCreateVector(ceed_data->elem_restr_q, &user->stats_ceed, NULL);

  CeedQFunction qf_stats;
  CeedQFunctionCreateInterior(ceed, 1, stats.qfunction, stats.qfunction_loc, &qf_stats);
  CeedQFunctionSetContext(qf_stats, stats.qfunction_context);
  CeedQFunctionContextDestroy(&stats.qfunction_context);
  CeedQFunctionAddInput(qf_stats, "q", num_comp_q, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_stats, "q_data", q_data_size_vol, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_stats, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_stats, "U_prod", num_comp_stats, CEED_EVAL_INTERP);

  // -- CEED setup the basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_stats, P, Q, CEED_GAUSS, &basis_stats);

  // -- CEED operator for stats collection
  CeedOperator op;
  CeedOperatorCreate(ceed, qf_stats, NULL, NULL, &op);
  CeedOperatorSetField(op, "q", ceed_data->elem_restr_q, ceed_data->basis_q, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op, "q_data", ceed_data->elem_restr_qd_i, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
  CeedOperatorSetField(op, "U_prod", ceed_data->elem_restr_q, basis_stats, CEED_VECTOR_ACTIVE);
  CeedQFunctionDestroy(&qf_stats);

  user->op_stats = op;

  PetscFunctionReturn(0);
}
