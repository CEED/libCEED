// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>

typedef struct {
  hipModule_t   module;
  hipFunction_t Interp;
  hipFunction_t InterpTranspose;
  hipFunction_t InterpTransposeAdd;
  hipFunction_t Grad;
  hipFunction_t GradTranspose;
  hipFunction_t GradTransposeAdd;
  hipFunction_t Weight;
  hipModule_t   moduleAtPoints;
  CeedInt       num_points;
  hipFunction_t InterpAtPoints;
  hipFunction_t InterpTransposeAtPoints;
  hipFunction_t InterpTransposeAddAtPoints;
  hipFunction_t GradAtPoints;
  hipFunction_t GradTransposeAtPoints;
  hipFunction_t GradTransposeAddAtPoints;
  CeedInt       block_sizes[3];  // interp, grad, weight thread block sizes
  CeedScalar   *d_interp_1d;
  CeedScalar   *d_grad_1d;
  CeedScalar   *d_collo_grad_1d;
  CeedScalar   *d_q_weight_1d;
  CeedScalar   *d_chebyshev_interp_1d;
  CeedInt       num_elem_at_points;
  CeedInt      *h_points_per_elem;
  CeedInt      *d_points_per_elem;
} CeedBasis_Hip_shared;

CEED_INTERN int CeedBasisCreateTensorH1_Hip_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
                                                   const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Hip_shared(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                                             const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis basis);
