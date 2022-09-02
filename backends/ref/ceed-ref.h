// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_ref_h
#define _ceed_ref_h

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
  CeedScalar *collo_grad_1d;
  bool        has_collo_interp;
} CeedBasis_Ref;

typedef struct {
  CeedScalar *array;
  CeedScalar *array_borrowed;
  CeedScalar *array_owned;
} CeedVector_Ref;

typedef struct {
  const CeedInt *offsets;
  CeedInt       *offsets_allocated;
  // Orientation, if it exists, is true when the face must be flipped (multiplies by -1.).
  const bool *orient;
  bool       *orient_allocated;
  int (*Apply)(CeedElemRestriction, const CeedInt, const CeedInt, const CeedInt, CeedInt, CeedInt, CeedTransposeMode, CeedVector, CeedVector,
               CeedRequest *);
} CeedElemRestriction_Ref;

typedef struct {
  const CeedScalar **inputs;
  CeedScalar       **outputs;
} CeedQFunction_Ref;

typedef struct {
  void *data;
  void *data_borrowed;
  void *data_owned;
} CeedQFunctionContext_Ref;

typedef struct {
  bool        is_identity_qf, is_identity_restr_op;
  CeedVector *e_vecs_full;  /* Full E-vectors, inputs followed by outputs */
  uint64_t   *input_states; /* State counter of inputs */
  CeedVector *e_vecs_in;    /* Single element input E-vectors  */
  CeedVector *e_vecs_out;   /* Single element output E-vectors */
  CeedVector *q_vecs_in;    /* Single element input Q-vectors  */
  CeedVector *q_vecs_out;   /* Single element output Q-vectors */
  CeedInt     num_inputs, num_outputs;
  CeedInt     num_active_in, num_active_out;
  CeedVector *qf_active_in;
} CeedOperator_Ref;

CEED_INTERN int CeedVectorCreate_Ref(CeedSize n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Ref(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, CeedElemRestriction r);
CEED_INTERN int CeedElemRestrictionCreateOriented_Ref(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orient,
                                                      CeedElemRestriction r);

CEED_INTERN int CeedBasisCreateTensorH1_Ref(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                            const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);
CEED_INTERN int CeedBasisCreateH1_Ref(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                      const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHdiv_Ref(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                        const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);

CEED_INTERN int CeedTensorContractCreate_Ref(CeedBasis basis, CeedTensorContract contract);

CEED_INTERN int CeedQFunctionCreate_Ref(CeedQFunction qf);

CEED_INTERN int CeedQFunctionContextCreate_Ref(CeedQFunctionContext ctx);

CEED_INTERN int CeedOperatorCreate_Ref(CeedOperator op);

#endif  // _ceed_ref_h
