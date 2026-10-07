// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#pragma once
#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>

typedef struct {
  CeedScalar *d_interp;
  CeedScalar *d_grad;
  CeedScalar *d_div;
  CeedScalar *d_curl;
  CeedScalar *d_q_weight;
} CeedBasisNonTensor_Cutlass;

CEED_INTERN int CeedBasisCreateH1_Cutlass(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                          const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHdiv_Cutlass(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                            const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHcurl_Cutlass(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                             const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedCutlassGemm_Cuda(bool trans_a, int m, int n, int k, CeedScalar alpha, const CeedScalar *A, int lda, const CeedScalar *B, int ldb,
                                     CeedScalar beta, CeedScalar *C, int ldc);

CEED_INTERN int CeedCutlassWeight_Cuda(const CeedInt num_elem, const CeedInt Q, const CeedScalar *d_q_weight, CeedScalar *d_v);