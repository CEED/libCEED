// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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

#ifndef _ceed_blocked_h
#define _ceed_blocked_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
  CeedScalar *colo_grad_1d;
} CeedBasis_Blocked;

typedef struct {
  bool is_identity_qf, is_identity_restr_op;
  CeedElemRestriction *blk_restr; /* Blocked versions of restrictions */
  CeedVector *e_vecs_full; /* Full E-vectors, inputs followed by outputs */
  uint64_t *input_states;  /* State counter of inputs */
  CeedVector *e_vecs_in;   /* Element block input E-vectors  */
  CeedVector *e_vecs_out;  /* Element block output E-vectors */
  CeedVector *q_vecs_in;   /* Element block input Q-vectors  */
  CeedVector *q_vecs_out;  /* Element block output Q-vectors */
  CeedInt    num_inputs, num_outputs;
  CeedInt    num_active_in, num_active_out;
  CeedVector *qf_active_in;
  CeedVector qf_l_vec;
  CeedElemRestriction qf_blk_rstr;
} CeedOperator_Blocked;

CEED_INTERN int CeedOperatorCreate_Blocked(CeedOperator op);

#endif // _ceed_blocked_h
