// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_SYCL_REF_HPP
#define CEED_SYCL_REF_HPP

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-common.hpp"
#include "../sycl/ceed-sycl-compile.hpp"

typedef struct {
  CeedScalar *h_array;
  CeedScalar *h_array_borrowed;
  CeedScalar *h_array_owned;
  CeedScalar *d_array;
  CeedScalar *d_array_borrowed;
  CeedScalar *d_array_owned;
  CeedScalar *reduction_norm;
} CeedVector_Sycl;

typedef struct {
  CeedInt  num_nodes;
  CeedInt  num_elem;
  CeedInt  num_comp;
  CeedInt  elem_size;
  CeedInt  comp_stride;
  CeedInt  strides[3];
  CeedInt *h_ind;
  CeedInt *h_ind_allocated;
  CeedInt *d_ind;
  CeedInt *d_ind_allocated;
  CeedInt *d_t_offsets;
  CeedInt *d_t_indices;
  CeedInt *d_l_vec_indices;
} CeedElemRestriction_Sycl;

typedef struct {
  CeedInt       dim;
  CeedInt       P_1d;
  CeedInt       Q_1d;
  CeedInt       num_comp;
  CeedInt       num_nodes;
  CeedInt       num_qpts;
  CeedInt       buf_len;
  CeedInt       op_len;
  SyclModule_t *sycl_module;
  CeedScalar   *d_interp_1d;
  CeedScalar   *d_grad_1d;
  CeedScalar   *d_q_weight_1d;
} CeedBasis_Sycl;

typedef struct {
  CeedInt     dim;
  CeedInt     num_comp;
  CeedInt     num_nodes;
  CeedInt     num_qpts;
  CeedScalar *d_interp;
  CeedScalar *d_grad;
  CeedScalar *d_q_weight;
} CeedBasisNonTensor_Sycl;

typedef struct {
  SyclModule_t *sycl_module;
  sycl::kernel *QFunction;
} CeedQFunction_Sycl;

typedef struct {
  void *h_data;
  void *h_data_borrowed;
  void *h_data_owned;
  void *d_data;
  void *d_data_borrowed;
  void *d_data_owned;
} CeedQFunctionContext_Sycl;

typedef struct {
  CeedBasis           basisin, basisout;
  CeedElemRestriction diagrstr, pbdiagrstr;
  CeedVector          elemdiag, pbelemdiag;
  CeedInt             numemodein, numemodeout, nnodes;
  CeedInt             nqpts, ncomp;  // Kernel parameters
  CeedEvalMode       *h_emodein, *h_emodeout;
  CeedEvalMode       *d_emodein, *d_emodeout;
  CeedScalar         *d_identity, *d_interpin, *d_interpout, *d_gradin, *d_gradout;
} CeedOperatorDiag_Sycl;

typedef struct {
  CeedInt     nelem, block_size_x, block_size_y, elemsPerBlock;
  CeedInt     numemodein, numemodeout, nqpts, nnodes, block_size, ncomp;  // Kernel parameters
  bool        fallback;
  CeedScalar *d_B_in, *d_B_out;
} CeedOperatorAssemble_Sycl;

typedef struct {
  CeedVector                *evecs;     // E-vectors, inputs followed by outputs
  CeedVector                *qvecsin;   // Input Q-vectors needed to apply operator
  CeedVector                *qvecsout;  // Output Q-vectors needed to apply operator
  CeedInt                    numein;
  CeedInt                    numeout;
  CeedInt                    qfnumactivein, qfnumactiveout;
  CeedVector                *qfactivein;
  CeedOperatorDiag_Sycl     *diag;
  CeedOperatorAssemble_Sycl *asmb;
} CeedOperator_Sycl;

CEED_INTERN int CeedVectorCreate_Sycl(CeedSize n, CeedVector vec);

CEED_INTERN int CeedBasisCreateTensorH1_Sycl(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                             const CeedScalar *qref_1d, const CeedScalar *qweight_1d, CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Sycl(CeedElemTopology topo, CeedInt dim, CeedInt ndof, CeedInt nqpts, const CeedScalar *interp,
                                       const CeedScalar *grad, const CeedScalar *qref, const CeedScalar *qweight, CeedBasis basis);

CEED_INTERN int CeedElemRestrictionCreate_Sycl(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *indices, const bool *orients,
                                               const CeedInt8 *curl_orients, CeedElemRestriction r);

CEED_INTERN int CeedQFunctionCreate_Sycl(CeedQFunction qf);

CEED_INTERN int CeedQFunctionContextCreate_Sycl(CeedQFunctionContext ctx);

CEED_INTERN int CeedOperatorCreate_Sycl(CeedOperator op);

#endif  // CEED_SYCL_REF_HPP
