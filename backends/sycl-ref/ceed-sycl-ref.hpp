// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_sycl_hpp
#define _ceed_sycl_hpp

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-common.hpp"

typedef struct {
  CeedScalar *h_array;
  CeedScalar *h_array_borrowed;
  CeedScalar *h_array_owned;
  CeedScalar *d_array;
  CeedScalar *d_array_borrowed;
  CeedScalar *d_array_owned;
} CeedVector_Sycl;

typedef struct {
  // CUmodule module;
  // CUfunction StridedTranspose;
  // CUfunction StridedNoTranspose;
  // CUfunction OffsetTranspose;
  // CUfunction OffsetNoTranspose;
  CeedInt  num_nodes;
  CeedInt *h_ind;
  CeedInt *h_ind_allocated;
  CeedInt *d_ind;
  CeedInt *d_ind_allocated;
  CeedInt *d_t_offsets;
  CeedInt *d_t_indices;
  CeedInt *d_l_vec_indices;
} CeedElemRestriction_Sycl;

typedef struct {
  // CUmodule module;
  // CUfunction Interp;
  // CUfunction Grad;
  // CUfunction Weight;
  CeedScalar *d_interp_1d;
  CeedScalar *d_grad_1d;
  CeedScalar *d_q_weight_1d;
} CeedBasis_Sycl;

typedef struct {
  // CUmodule module;
  // CUfunction Interp;
  // CUfunction Grad;
  // CUfunction Weight;
  CeedScalar *d_interp;
  CeedScalar *d_grad;
  CeedScalar *d_q_weight;
} CeedBasisNonTensor_Sycl;

typedef struct {
  // CUmodule module;
  char *qfunction_name;
  char *qfunction_source;
  // CUfunction QFunction;
  // Fields_Sycl fields;
  void *d_c;
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
  // CUmodule module;
  // CUfunction linearDiagonal;
  // CUfunction linearPointBlock;
  CeedBasis           basisin, basisout;
  CeedElemRestriction diagrstr, pbdiagrstr;
  CeedVector          elemdiag, pbelemdiag;
  CeedInt             numemodein, numemodeout, nnodes;
  CeedEvalMode       *h_emodein, *h_emodeout;
  CeedEvalMode       *d_emodein, *d_emodeout;
  CeedScalar         *d_identity, *d_interpin, *d_interpout, *d_gradin, *d_gradout;
} CeedOperatorDiag_Sycl;

typedef struct {
  // CUmodule module;
  // CUfunction linearAssemble;
  CeedInt     nelem, block_size_x, block_size_y, elemsPerBlock;
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

// CEED_INTERN int CeedSyclGetCublasHandle(Ceed ceed, cublasHandle_t *handle);

CEED_INTERN int CeedVectorCreate_Sycl(CeedSize n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Sycl(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *indices, CeedElemRestriction r);

CEED_INTERN int CeedElemRestrictionCreateBlocked_Sycl(const CeedMemType mem_type, const CeedCopyMode copy_mode, const CeedInt *indices,
                                                      const CeedElemRestriction res);

CEED_INTERN int CeedBasisApplyElems_Sycl(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
                                         const CeedVector u, CeedVector v);

CEED_INTERN int CeedQFunctionApplyElems_Sycl(CeedQFunction qf, const CeedInt Q, const CeedVector *const u, const CeedVector *v);

CEED_INTERN int CeedBasisCreateTensorH1_Sycl(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                             const CeedScalar *qref_1d, const CeedScalar *qweight_1d, CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Sycl(CeedElemTopology, CeedInt, CeedInt, CeedInt, const CeedScalar *, const CeedScalar *, const CeedScalar *,
                                       const CeedScalar *, CeedBasis);

CEED_INTERN int CeedQFunctionCreate_Sycl(CeedQFunction qf);

CEED_INTERN int CeedQFunctionContextCreate_Sycl(CeedQFunctionContext ctx);

CEED_INTERN int CeedOperatorCreate_Sycl(CeedOperator op);

#endif
