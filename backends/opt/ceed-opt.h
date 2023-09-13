// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OPT_H
#define CEED_OPT_H

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
  CeedInt block_size;
} Ceed_Opt;

typedef struct {
  CeedScalar *colo_grad_1d;
} CeedBasis_Opt;

typedef struct {
  bool                 is_identity_qf, is_identity_rstr_op;
  CeedElemRestriction *block_rstr;   /* Blocked versions of restrictions */
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
  CeedElemRestriction  qf_block_rstr;
} CeedOperator_Opt;

CEED_INTERN int CeedTensorContractCreate_Opt(CeedBasis basis, CeedTensorContract contract);

CEED_INTERN int CeedOperatorCreate_Opt(CeedOperator op);

#endif  // CEED_OPT_H
