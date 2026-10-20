// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <cublas_v2.h>
#include <cuda.h>

typedef struct {
  cublasHandle_t handle;
  CUmodule       module;
  CUfunction     Weight;

  // Non tensor
  CeedScalar *d_interp;
  CeedScalar *d_grad;
  CeedScalar *d_div;
  CeedScalar *d_curl;
  CeedScalar *d_q_weight;
} CeedBasis_Cuda_cublas;

CEED_INTERN int CeedBasisCreateH1_Cuda_cublas(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                              const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHdiv_Cuda_cublas(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                                const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHcurl_Cuda_cublas(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                                 const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Cuda_cublas_det(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                                  const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHdiv_Cuda_cublas_det(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                                    const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
CEED_INTERN int CeedBasisCreateHcurl_Cuda_cublas_det(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts,
                                                     const CeedScalar *interp, const CeedScalar *curl, const CeedScalar *q_ref,
                                                     const CeedScalar *q_weight, CeedBasis basis);
