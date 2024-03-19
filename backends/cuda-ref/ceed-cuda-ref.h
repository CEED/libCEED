// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_CUDA_REF_H
#define CEED_CUDA_REF_H

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
  CUmodule   module;
  CUfunction ApplyNoTranspose, ApplyTranspose;
  CUfunction ApplyUnsignedNoTranspose, ApplyUnsignedTranspose;
  CUfunction ApplyUnorientedNoTranspose, ApplyUnorientedTranspose;
  CeedInt    num_nodes;
  CeedInt   *h_offsets;
  CeedInt   *h_offsets_borrowed;
  CeedInt   *h_offsets_owned;
  CeedInt   *d_offsets;
  CeedInt   *d_offsets_borrowed;
  CeedInt   *d_offsets_owned;
  CeedInt   *d_t_offsets;
  CeedInt   *d_t_indices;
  CeedInt   *d_l_vec_indices;
  bool      *h_orients;
  bool      *h_orients_borrowed;
  bool      *h_orients_owned;
  bool      *d_orients;
  bool      *d_orients_borrowed;
  bool      *d_orients_owned;
  CeedInt8  *h_curl_orients;
  CeedInt8  *h_curl_orients_borrowed;
  CeedInt8  *h_curl_orients_owned;
  CeedInt8  *d_curl_orients;
  CeedInt8  *d_curl_orients_borrowed;
  CeedInt8  *d_curl_orients_owned;
} CeedElemRestriction_Cuda;

typedef struct {
  CUmodule    module;
  CUfunction  Interp;
  CUfunction  Grad;
  CUfunction  Weight;
  CeedScalar *d_interp_1d;
  CeedScalar *d_grad_1d;
  CeedScalar *d_q_weight_1d;
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
  const char *qfunction_source;
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
  CeedVector                *e_vecs;      // E-vectors, inputs followed by outputs
  CeedVector                *q_vecs_in;   // Input Q-vectors needed to apply operator
  CeedVector                *q_vecs_out;  // Output Q-vectors needed to apply operator
  CeedInt                    num_inputs, num_outputs;
  CeedInt                    num_active_in, num_active_out;
  CeedVector                *qf_active_in;
  CeedOperatorDiag_Cuda     *diag;
  CeedOperatorAssemble_Cuda *asmb;
} CeedOperator_Cuda;

CEED_INTERN int CeedGetCublasHandle_Cuda(Ceed ceed, cublasHandle_t *handle);

CEED_INTERN int CeedVectorCreate_Cuda(CeedSize n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Cuda(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *indices, const bool *orients,
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

#endif  // CEED_CUDA_REF_H
