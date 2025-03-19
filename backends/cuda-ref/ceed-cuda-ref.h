// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/cuda/cuda-types.h>
#include <cublas_v2.h>
#include <cuda.h>

typedef struct {
  CeedScalar *h_array;
  CeedScalar *h_array_borrowed;
  CeedScalar *h_array_owned;
  CeedScalar *d_array;
  CeedScalar *d_array_borrowed;
  CeedScalar *d_array_owned;
} CeedVector_Cuda;

typedef struct {
  CUmodule        module;
  CUfunction      ApplyNoTranspose, ApplyTranspose;
  CUfunction      ApplyUnsignedNoTranspose, ApplyUnsignedTranspose;
  CUfunction      ApplyUnorientedNoTranspose, ApplyUnorientedTranspose;
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
} CeedElemRestriction_Cuda;

typedef struct {
  CUmodule    module;
  CUfunction  Interp;
  CUfunction  Grad;
  CUfunction  Weight;
  CUmodule    moduleAtPoints;
  CeedInt     num_points;
  CUfunction  InterpAtPoints;
  CUfunction  InterpTransposeAtPoints;
  CUfunction  GradAtPoints;
  CUfunction  GradTransposeAtPoints;
  CeedScalar *d_interp_1d;
  CeedScalar *d_grad_1d;
  CeedScalar *d_q_weight_1d;
  CeedScalar *d_chebyshev_interp_1d;
  CeedInt     num_elem_at_points;
  CeedInt    *h_points_per_elem;
  CeedInt    *d_points_per_elem;
} CeedBasis_Cuda;

typedef struct {
  CUmodule    module;
  CUfunction  Interp;
  CUfunction  InterpTranspose;
  CUfunction  Deriv;
  CUfunction  DerivTranspose;
  CUfunction  Weight;
  CeedScalar *d_interp;
  CeedScalar *d_grad;
  CeedScalar *d_div;
  CeedScalar *d_curl;
  CeedScalar *d_q_weight;
} CeedBasisNonTensor_Cuda;

typedef struct {
  CUmodule    module;
  const char *qfunction_name;
  CUfunction  QFunction;
  Fields_Cuda fields;
  void       *d_c;
} CeedQFunction_Cuda;

typedef struct {
  void *h_data;
  void *h_data_borrowed;
  void *h_data_owned;
  void *d_data;
  void *d_data_borrowed;
  void *d_data_owned;
} CeedQFunctionContext_Cuda;

typedef struct {
  CUmodule            module, module_point_block;
  CUfunction          LinearDiagonal;
  CUfunction          LinearPointBlock;
  CeedElemRestriction diag_rstr, point_block_diag_rstr;
  CeedVector          elem_diag, point_block_elem_diag;
  CeedEvalMode       *d_eval_modes_in, *d_eval_modes_out;
  CeedScalar         *d_identity, *d_interp_in, *d_grad_in, *d_div_in, *d_curl_in;
  CeedScalar         *d_interp_out, *d_grad_out, *d_div_out, *d_curl_out;
} CeedOperatorDiag_Cuda;

typedef struct {
  CUmodule    module;
  CUfunction  LinearAssemble;
  CeedInt     block_size_x, block_size_y, elems_per_block;
  CeedScalar *d_B_in, *d_B_out;
} CeedOperatorAssemble_Cuda;

typedef struct {
  bool                      *skip_rstr_in, *skip_rstr_out, *apply_add_basis_out;
  uint64_t                  *input_states;  // State tracking for passive inputs
  CeedVector                *e_vecs_in, *e_vecs_out;
  CeedVector                *q_vecs_in, *q_vecs_out;
  CeedInt                    num_inputs, num_outputs;
  CeedInt                    num_active_in, num_active_out;
  CeedInt                   *input_field_order, *output_field_order;
  CeedSize                   max_active_e_vec_len;
  CeedInt                    max_num_points;
  CeedInt                   *num_points;
  CeedVector                *qf_active_in, point_coords_elem;
  CeedOperatorDiag_Cuda     *diag;
  CeedOperatorAssemble_Cuda *asmb;
} CeedOperator_Cuda;

CEED_INTERN int CeedGetCublasHandle_Cuda(Ceed ceed, cublasHandle_t *handle);

CEED_INTERN int CeedVectorCreate_Cuda(CeedSize n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Cuda(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orients,
                                               const CeedInt8 *curl_orients, CeedElemRestriction r);

CEED_INTERN int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                             const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);
CEED_INTERN int CeedBasisCreateH1_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                       const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHdiv_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                         const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHcurl_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                          const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);

CEED_INTERN int CeedQFunctionCreate_Cuda(CeedQFunction qf);

CEED_INTERN int CeedQFunctionContextCreate_Cuda(CeedQFunctionContext ctx);

CEED_INTERN int CeedOperatorCreate_Cuda(CeedOperator op);
CEED_INTERN int CeedOperatorCreateAtPoints_Cuda(CeedOperator op);
