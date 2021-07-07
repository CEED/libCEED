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

#ifndef _ceed_opt_h
#define _ceed_opt_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
  CeedInt blk_size;
} Ceed_Opt;

typedef struct {
  CeedScalar *colo_grad_1d;
} CeedBasis_Opt;

typedef struct {
  bool is_identity_qf, is_identity_restr_op;
  CeedElemRestriction *blk_restr; /* Blocked versions of restrictions */
  CeedVector
  *e_vecs;   /* E-vectors needed to apply operator (input followed by outputs) */
  CeedScalar **e_data;
  uint64_t *input_state;   /* State counter of inputs */
  CeedVector *e_vecs_in;   /* Input E-vectors needed to apply operator */
  CeedVector *e_vecs_out;  /* Output E-vectors needed to apply operator */
  CeedVector *q_vecs_in;   /* Input Q-vectors needed to apply operator */
  CeedVector *q_vecs_out;  /* Output Q-vectors needed to apply operator */
  CeedInt    num_e_vecs_in;
  CeedInt    num_e_vecs_out;
} CeedOperator_Opt;

CEED_INTERN int CeedOperatorCreate_Opt(CeedOperator op);

#endif // _ceed_opt_h
