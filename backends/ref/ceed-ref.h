// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_REF_H
#define CEED_REF_H

#include <ceed.h>
#include <ceed/backend.h>
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
  const CeedInt  *offsets;
  const CeedInt  *offsets_borrowed;
  const CeedInt  *offsets_owned;
  const bool     *orients; /* Orientation, if it exists, is true when the dof must be flipped */
  const bool     *orients_borrowed;
  const bool     *orients_owned;
  const CeedInt8 *curl_orients; /* Tridiagonal matrix (row-major) for a general transformation during restriction */
  const CeedInt8 *curl_orients_borrowed;
  const CeedInt8 *curl_orients_owned;
  int (*Apply)(CeedElemRestriction, CeedInt, CeedInt, CeedInt, CeedInt, CeedInt, CeedTransposeMode, bool, bool, CeedVector, CeedVector,
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
  bool        is_identity_qf, is_identity_rstr_op;
  CeedVector *e_vecs_full;  /* Full E-vectors, inputs followed by outputs */
  uint64_t   *input_states; /* State counter of inputs */
  CeedVector *e_vecs_in;    /* Single element input E-vectors  */
  CeedVector *e_vecs_out;   /* Single element output E-vectors */
  CeedVector *q_vecs_in;    /* Single element input Q-vectors  */
  CeedVector *q_vecs_out;   /* Single element output Q-vectors */
  CeedInt     num_inputs, num_outputs;
  CeedInt     num_active_in, num_active_out;
  CeedVector *qf_active_in, point_coords_elem;
} CeedOperator_Ref;

CEED_INTERN int CeedVectorCreate_Ref(CeedSize n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Ref(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orients,
                                              const CeedInt8 *curl_orients, CeedElemRestriction r);

CEED_INTERN int CeedBasisCreateTensorH1_Ref(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                            const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);
CEED_INTERN int CeedBasisCreateH1_Ref(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                      const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHdiv_Ref(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                        const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHcurl_Ref(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                         const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);

CEED_INTERN int CeedTensorContractCreate_Ref(CeedTensorContract contract);

CEED_INTERN int CeedQFunctionCreate_Ref(CeedQFunction qf);

CEED_INTERN int CeedQFunctionContextCreate_Ref(CeedQFunctionContext ctx);

CEED_INTERN int CeedOperatorCreate_Ref(CeedOperator op);
CEED_INTERN int CeedOperatorCreateAtPoints_Ref(CeedOperator op);

#endif  // CEED_REF_H
