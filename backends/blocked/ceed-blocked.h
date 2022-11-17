// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_blocked_h
#define _ceed_blocked_h

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
  CeedScalar *colo_grad_1d;
} CeedBasis_Blocked;

typedef struct {
  bool                 is_identity_qf, is_identity_restr_op;
  CeedElemRestriction *blk_restr;    /* Blocked versions of restrictions */
  CeedVector          *e_vecs_full;  /* Full E-vectors, inputs followed by outputs */
  uint64_t            *input_states; /* State counter of inputs */
  CeedVector          *e_vecs_in;    /* Element block input E-vectors  */
  CeedVector          *e_vecs_out;   /* Element block output E-vectors */
  CeedVector          *q_vecs_in;    /* Element block input Q-vectors  */
  CeedVector          *q_vecs_out;   /* Element block output Q-vectors */
  CeedInt              num_inputs, num_outputs;
  CeedInt              num_active_in, num_active_out;
  CeedVector          *qf_active_in;
  CeedVector           qf_l_vec;
  CeedElemRestriction  qf_blk_rstr;
} CeedOperator_Blocked;

CEED_INTERN int CeedOperatorCreate_Blocked(CeedOperator op);

#endif  // _ceed_blocked_h
