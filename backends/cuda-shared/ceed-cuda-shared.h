// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  CUfunction  Interp;
  CUfunction  InterpTranspose;
  CUfunction  InterpTransposeAdd;
  CUfunction  Grad;
  CUfunction  GradTranspose;
  CUfunction  GradTransposeAdd;
  CUfunction  Weight;
  CUmodule    moduleAtPoints;
  CeedInt     num_points;
  CUfunction  InterpAtPoints;
  CUfunction  InterpTransposeAtPoints;
  CUfunction  GradAtPoints;
  CUfunction  GradTransposeAtPoints;
  CeedScalar *d_interp_1d;
  CeedScalar *d_grad_1d;
  CeedScalar *d_collo_grad_1d;
  CeedScalar *d_q_weight_1d;
  CeedScalar *d_chebyshev_interp_1d;
  CeedScalar *c_B;
  CeedScalar *c_G;
  CeedInt     num_elem_at_points;
  CeedInt    *h_points_per_elem;
  CeedInt    *d_points_per_elem;
} CeedBasis_Cuda_shared;

CEED_INTERN int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                                    const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Cuda_shared(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                              const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
