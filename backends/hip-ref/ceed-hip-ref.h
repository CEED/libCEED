// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/hip/hip-types.h>
#include <hip/hip_runtime.h>
#if (HIP_VERSION >= 50200000)
#include <hipblas/hipblas.h>  // IWYU pragma: export
#else
#include <hipblas.h>  // IWYU pragma: export
#endif

typedef struct {
  int         has_unified_addressing;
  CeedScalar *h_array;
  CeedScalar *h_array_borrowed;
  CeedScalar *h_array_owned;
  CeedScalar *d_array;
  CeedScalar *d_array_borrowed;
  CeedScalar *d_array_owned;
} CeedVector_Hip;

typedef struct {
  hipModule_t     module;
  hipFunction_t   ApplyNoTranspose, ApplyTranspose;
  hipFunction_t   ApplyUnsignedNoTranspose, ApplyUnsignedTranspose;
  hipFunction_t   ApplyUnorientedNoTranspose, ApplyUnorientedTranspose;
  CeedInt         num_nodes;
  const CeedInt  *h_offsets;
  const CeedInt  *h_offsets_borrowed;
  const CeedInt  *h_offsets_owned;
  const CeedInt  *d_offsets;
  const CeedInt  *d_offsets_borrowed;
  const CeedInt  *d_offsets_owned;
  const CeedInt  *d_t_offsets;
  const CeedInt  *d_t_indices;
  const CeedInt  *d_l_vec_indices;
  const bool     *h_orients;
  const bool     *h_orients_borrowed;
  const bool     *h_orients_owned;
  const bool     *d_orients;
  const bool     *d_orients_borrowed;
  const bool     *d_orients_owned;
  const CeedInt8 *h_curl_orients;
  const CeedInt8 *h_curl_orients_borrowed;
  const CeedInt8 *h_curl_orients_owned;
  const CeedInt8 *d_curl_orients;
  const CeedInt8 *d_curl_orients_borrowed;
  const CeedInt8 *d_curl_orients_owned;
  const CeedInt  *h_offsets_at_points;
  const CeedInt  *h_offsets_at_points_borrowed;
  const CeedInt  *h_offsets_at_points_owned;
  const CeedInt  *d_offsets_at_points;
  const CeedInt  *d_offsets_at_points_borrowed;
  const CeedInt  *d_offsets_at_points_owned;
  const CeedInt  *h_points_per_elem;
  const CeedInt  *h_points_per_elem_borrowed;
  const CeedInt  *h_points_per_elem_owned;
  const CeedInt  *d_points_per_elem;
  const CeedInt  *d_points_per_elem_borrowed;
  const CeedInt  *d_points_per_elem_owned;
} CeedElemRestriction_Hip;

typedef struct {
  hipModule_t   module;
  hipFunction_t Interp;
  hipFunction_t Grad;
  hipFunction_t Weight;
  hipModule_t   moduleAtPoints;
  CeedInt       num_points;
  hipFunction_t InterpAtPoints;
  hipFunction_t InterpTransposeAtPoints;
  hipFunction_t GradAtPoints;
  hipFunction_t GradTransposeAtPoints;
  CeedScalar   *d_interp_1d;
  CeedScalar   *d_grad_1d;
  CeedScalar   *d_q_weight_1d;
  CeedScalar   *d_chebyshev_interp_1d;
  CeedInt       num_elem_at_points;
  CeedInt      *h_points_per_elem;
  CeedInt      *d_points_per_elem;
} CeedBasis_Hip;

typedef struct {
  hipModule_t   module;
  hipFunction_t Interp;
  hipFunction_t InterpTranspose;
  hipFunction_t Deriv;
  hipFunction_t DerivTranspose;
  hipFunction_t Weight;
  CeedScalar   *d_interp;
  CeedScalar   *d_grad;
  CeedScalar   *d_div;
  CeedScalar   *d_curl;
  CeedScalar   *d_q_weight;
} CeedBasisNonTensor_Hip;

typedef struct {
  hipModule_t   module;
  const char   *qfunction_name;
  hipFunction_t QFunction;
  Fields_Hip    fields;
  void         *d_c;
} CeedQFunction_Hip;

typedef struct {
  void *h_data;
  void *h_data_borrowed;
  void *h_data_owned;
  void *d_data;
  void *d_data_borrowed;
  void *d_data_owned;
} CeedQFunctionContext_Hip;

typedef struct {
  hipModule_t         module, module_point_block;
  hipFunction_t       LinearDiagonal;
  hipFunction_t       LinearPointBlock;
  CeedElemRestriction diag_rstr, point_block_diag_rstr;
  CeedVector          elem_diag, point_block_elem_diag;
  CeedEvalMode       *d_eval_modes_in, *d_eval_modes_out;
  CeedScalar         *d_identity, *d_interp_in, *d_grad_in, *d_div_in, *d_curl_in;
  CeedScalar         *d_interp_out, *d_grad_out, *d_div_out, *d_curl_out;
} CeedOperatorDiag_Hip;

typedef struct {
  hipModule_t   module;
  hipFunction_t LinearAssemble;
  CeedInt       block_size_x, block_size_y, elems_per_block;
  CeedScalar   *d_B_in, *d_B_out;
} CeedOperatorAssemble_Hip;

typedef struct {
  bool                     *skip_rstr_in, *skip_rstr_out, *apply_add_basis_out;
  uint64_t                 *input_states;  // State tracking for passive inputs
  CeedVector               *e_vecs_in, *e_vecs_out;
  CeedVector               *q_vecs_in, *q_vecs_out;
  CeedInt                   num_inputs, num_outputs;
  CeedInt                   num_active_in, num_active_out;
  CeedInt                  *input_field_order, *output_field_order;
  CeedSize                  max_active_e_vec_len;
  CeedInt                   max_num_points;
  CeedInt                  *num_points;
  CeedVector               *qf_active_in, point_coords_elem;
  CeedOperatorDiag_Hip     *diag;
  CeedOperatorAssemble_Hip *asmb;
} CeedOperator_Hip;

CEED_INTERN int CeedGetHipblasHandle_Hip(Ceed ceed, hipblasHandle_t *handle);

CEED_INTERN int CeedVectorCreate_Hip(CeedSize n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Hip(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orients,
                                              const CeedInt8 *curl_orients, CeedElemRestriction rstr);

CEED_INTERN int CeedBasisCreateTensorH1_Hip(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                            const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);
CEED_INTERN int CeedBasisCreateH1_Hip(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                      const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHdiv_Hip(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                        const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHcurl_Hip(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                         const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);

CEED_INTERN int CeedQFunctionCreate_Hip(CeedQFunction qf);

CEED_INTERN int CeedQFunctionContextCreate_Hip(CeedQFunctionContext ctx);

CEED_INTERN int CeedOperatorCreate_Hip(CeedOperator op);
CEED_INTERN int CeedOperatorCreateAtPoints_Hip(CeedOperator op);
