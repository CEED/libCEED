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
  CUfunction StridedNoTranspose;
  CUfunction StridedTranspose;
  CUfunction OffsetNoTranspose;
  CUfunction OffsetTranspose;
  CUfunction OffsetTransposeDet;
  CeedInt    num_nodes;
  CeedInt   *h_ind;
  CeedInt   *h_ind_allocated;
  CeedInt   *d_ind;
  CeedInt   *d_ind_allocated;
  CeedInt   *d_t_offsets;
  CeedInt   *d_t_indices;
  CeedInt   *d_l_vec_indices;
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
  CUfunction  Grad;
  CUfunction  Weight;
  CeedScalar *d_interp;
  CeedScalar *d_grad;
  CeedScalar *d_q_weight;
} CeedBasisNonTensor_Cuda;

typedef struct {
  CUmodule    module;
  char       *qfunction_name;
  char       *qfunction_source;
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
  CUmodule            module;
  CUfunction          linearDiagonal;
  CUfunction          linearPointBlock;
  CeedBasis           basis_in, basis_out;
  CeedElemRestriction diag_rstr, point_block_diag_rstr;
  CeedVector          elem_diag, point_block_elem_diag;
  CeedInt             num_e_mode_in, num_e_mode_out, num_nodes;
  CeedEvalMode       *h_e_mode_in, *h_e_mode_out;
  CeedEvalMode       *d_e_mode_in, *d_e_mode_out;
  CeedScalar         *d_identity, *d_interp_in, *d_interp_out, *d_grad_in, *d_grad_out;
} CeedOperatorDiag_Cuda;

typedef struct {
  CUmodule    module;
  CUfunction  linearAssemble;
  CeedInt     num_elem, block_size_x, block_size_y, elem_per_block;
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

CEED_INTERN int CeedQFunctionCreate_Cuda(CeedQFunction qf);

CEED_INTERN int CeedQFunctionContextCreate_Cuda(CeedQFunctionContext ctx);

CEED_INTERN int CeedOperatorCreate_Cuda(CeedOperator op);

#endif  // CEED_CUDA_REF_H
