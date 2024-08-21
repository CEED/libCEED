// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
  CeedScalar *colo_grad_1d;
} CeedBasis_Blocked;

typedef struct {
  bool                 is_identity_qf, is_identity_rstr_op;
  bool                *skip_rstr_in, *skip_rstr_out, *apply_add_basis_out;
  CeedInt             *e_data_out_indices;
  uint64_t            *input_states; /* State counter of inputs */
  CeedVector          *e_vecs_full;  /* Full E-vectors, inputs followed by outputs */
  CeedVector          *e_vecs_in;    /* Element block input E-vectors  */
  CeedVector          *e_vecs_out;   /* Element block output E-vectors */
  CeedVector          *q_vecs_in;    /* Element block input Q-vectors  */
  CeedVector          *q_vecs_out;   /* Element block output Q-vectors */
  CeedElemRestriction *block_rstr;   /* Blocked versions of restrictions */
  CeedInt              num_inputs, num_outputs;
  CeedInt              qf_size_in, qf_size_out;
  CeedVector           qf_l_vec;
  CeedElemRestriction  qf_block_rstr;
} CeedOperator_Blocked;

CEED_INTERN int CeedOperatorCreate_Blocked(CeedOperator op);
