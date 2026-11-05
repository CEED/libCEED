// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#pragma once
#include <ceed.h>
#include <ceed/backend.h>
#include <cuda.h>

typedef struct {
  CUmodule    module;
  CUfunction  Interp, InterpTranspose, Deriv, DerivTranspose, Weight;
  CeedInt     num_threads[2], num_cols_per_block[2];
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
