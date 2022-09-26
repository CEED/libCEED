// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_hip_h
#define _ceed_hip_h

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <hip/hip_runtime.h>

#include "../hip/ceed-hip-common.h"

typedef struct {
  CeedScalar *h_array;
  CeedScalar *h_array_borrowed;
  CeedScalar *h_array_owned;
  CeedScalar *d_array;
  CeedScalar *d_array_borrowed;
  CeedScalar *d_array_owned;
} CeedVector_Hip;

typedef struct {
  hipModule_t   module;
  hipFunction_t StridedTranspose;
  hipFunction_t StridedNoTranspose;
  hipFunction_t OffsetTranspose;
  hipFunction_t OffsetNoTranspose;
  CeedInt       num_nodes;
  CeedInt      *h_ind;
  CeedInt      *h_ind_allocated;
  CeedInt      *d_ind;
  CeedInt      *d_ind_allocated;
  CeedInt      *d_t_offsets;
  CeedInt      *d_t_indices;
  CeedInt      *d_l_vec_indices;
} CeedElemRestriction_Hip;

typedef struct {
  hipModule_t   module;
  hipFunction_t Interp;
  hipFunction_t Grad;
  hipFunction_t Weight;
  CeedScalar   *d_interp_1d;
  CeedScalar   *d_grad_1d;
  CeedScalar   *d_q_weight_1d;
} CeedBasis_Hip;

typedef struct {
  hipModule_t   module;
  hipFunction_t Interp;
  hipFunction_t Grad;
  hipFunction_t Weight;
  CeedScalar   *d_interp;
  CeedScalar   *d_grad;
  CeedScalar   *d_q_weight;
} CeedBasisNonTensor_Hip;

typedef struct {
  hipModule_t   module;
  char         *qfunction_name;
  char         *qfunction_source;
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
  hipModule_t         module;
  hipFunction_t       linearDiagonal;
  hipFunction_t       linearPointBlock;
  CeedBasis           basisin, basisout;
  CeedElemRestriction diagrstr, pbdiagrstr;
  CeedVector          elemdiag, pbelemdiag;
  CeedInt             numemodein, numemodeout, nnodes;
  CeedEvalMode       *h_emodein, *h_emodeout;
  CeedEvalMode       *d_emodein, *d_emodeout;
  CeedScalar         *d_identity, *d_interpin, *d_interpout, *d_gradin, *d_gradout;
} CeedOperatorDiag_Hip;

typedef struct {
  hipModule_t   module;
  hipFunction_t linearAssemble;
  CeedInt       nelem, block_size_x, block_size_y, elemsPerBlock;
  CeedScalar   *d_B_in, *d_B_out;
} CeedOperatorAssemble_Hip;

typedef struct {
  CeedVector               *evecs;     // E-vectors, inputs followed by outputs
  CeedVector               *qvecsin;   // Input Q-vectors needed to apply operator
  CeedVector               *qvecsout;  // Output Q-vectors needed to apply operator
  CeedInt                   numein;
  CeedInt                   numeout;
  CeedInt                   qfnumactivein, qfnumactiveout;
  CeedVector               *qfactivein;
  CeedOperatorDiag_Hip     *diag;
  CeedOperatorAssemble_Hip *asmb;
} CeedOperator_Hip;

CEED_INTERN int CeedHipGetHipblasHandle(Ceed ceed, hipblasHandle_t *handle);

CEED_INTERN int CeedVectorCreate_Hip(CeedSize n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Hip(CeedMemType mtype, CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction r);

CEED_INTERN int CeedElemRestrictionCreateBlocked_Hip(const CeedMemType mtype, const CeedCopyMode cmode, const CeedInt *indices,
                                                     const CeedElemRestriction res);

CEED_INTERN int CeedBasisApplyElems_Hip(CeedBasis basis, const CeedInt nelem, CeedTransposeMode tmode, CeedEvalMode emode, const CeedVector u,
                                        CeedVector v);

CEED_INTERN int CeedQFunctionApplyElems_Hip(CeedQFunction qf, const CeedInt Q, const CeedVector *const u, const CeedVector *v);

CEED_INTERN int CeedBasisCreateTensorH1_Hip(CeedInt dim, CeedInt P1d, CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
                                            const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Hip(CeedElemTopology, CeedInt, CeedInt, CeedInt, const CeedScalar *, const CeedScalar *, const CeedScalar *,
                                      const CeedScalar *, CeedBasis);

CEED_INTERN int CeedQFunctionCreate_Hip(CeedQFunction qf);

CEED_INTERN int CeedQFunctionContextCreate_Hip(CeedQFunctionContext ctx);

CEED_INTERN int CeedOperatorCreate_Hip(CeedOperator op);

#endif
