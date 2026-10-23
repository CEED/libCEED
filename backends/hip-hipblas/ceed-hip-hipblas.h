// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
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

#include "../hip/ceed-hip-common.h"

typedef struct {
  hipblasHandle_t handle;
  hipModule_t     module;
  hipFunction_t   Weight;

  // Non tensor
  CeedScalar *d_interp;
  CeedScalar *d_grad;
  CeedScalar *d_div;
  CeedScalar *d_curl;
  CeedScalar *d_q_weight;
} CeedBasis_Hip_hipblas;

CEED_INTERN int CeedBasisCreateH1_Hip_hipblas(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                              const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHdiv_Hip_hipblas(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                                const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHcurl_Hip_hipblas(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                                 const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Hip_hipblas_det(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                                  const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHdiv_Hip_hipblas_det(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                                    const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHcurl_Hip_hipblas_det(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts,
                                                     const CeedScalar *interp, const CeedScalar *curl, const CeedScalar *q_ref,
                                                     const CeedScalar *q_weight, CeedBasis basis);
