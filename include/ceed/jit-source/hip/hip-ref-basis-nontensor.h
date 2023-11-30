// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for HIP non-tensor product basis
#ifndef CEED_HIP_REF_BASIS_NONTENSOR_H
#define CEED_HIP_REF_BASIS_NONTENSOR_H

#include <ceed.h>

#include "hip-ref-basis-nontensor-templates.h"

//------------------------------------------------------------------------------
// Non-Tensor Basis Kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Interp
//------------------------------------------------------------------------------
extern "C" __global__ void Interp(const CeedInt num_elem, const CeedScalar *__restrict__ d_B, const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    Contract<BASIS_NUM_COMP, BASIS_Q_COMP_INTERP, BASIS_P, BASIS_Q>(elem, BASIS_P, BASIS_Q, BASIS_P * num_elem, BASIS_Q * num_elem,
                                                                    BASIS_NUM_COMP * BASIS_Q * num_elem, d_B, d_U, d_V);
  }
}

extern "C" __global__ void InterpTranspose(const CeedInt num_elem, const CeedScalar *__restrict__ d_B, const CeedScalar *__restrict__ d_U,
                                           CeedScalar *__restrict__ d_V) {
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    ContractTranspose<BASIS_NUM_COMP, BASIS_Q_COMP_INTERP, BASIS_P, BASIS_Q>(elem, BASIS_Q, BASIS_P, BASIS_Q * num_elem, BASIS_P * num_elem,
                                                                             BASIS_NUM_COMP * BASIS_Q * num_elem, d_B, d_U, d_V);
  }
}

//------------------------------------------------------------------------------
// Deriv
//------------------------------------------------------------------------------
extern "C" __global__ void Deriv(const CeedInt num_elem, const CeedScalar *__restrict__ d_B, const CeedScalar *__restrict__ d_U,
                                 CeedScalar *__restrict__ d_V) {
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    Contract<BASIS_NUM_COMP, BASIS_Q_COMP_DERIV, BASIS_P, BASIS_Q>(elem, BASIS_P, BASIS_Q, BASIS_P * num_elem, BASIS_Q * num_elem,
                                                                   BASIS_NUM_COMP * BASIS_Q * num_elem, d_B, d_U, d_V);
  }
}

extern "C" __global__ void DerivTranspose(const CeedInt num_elem, const CeedScalar *__restrict__ d_B, const CeedScalar *__restrict__ d_U,
                                          CeedScalar *__restrict__ d_V) {
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    ContractTranspose<BASIS_NUM_COMP, BASIS_Q_COMP_DERIV, BASIS_P, BASIS_Q>(elem, BASIS_Q, BASIS_P, BASIS_Q * num_elem, BASIS_P * num_elem,
                                                                            BASIS_NUM_COMP * BASIS_Q * num_elem, d_B, d_U, d_V);
  }
}

//------------------------------------------------------------------------------
// Weight
//------------------------------------------------------------------------------
extern "C" __global__ void Weight(const CeedInt num_elem, const CeedScalar *__restrict__ q_weight, CeedScalar *__restrict__ d_V) {
  const CeedInt t_id = threadIdx.x;
  // TODO load q_weight in shared memory if blockDim.z > 1?

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    d_V[elem * BASIS_Q + t_id] = q_weight[t_id];
  }
}

//------------------------------------------------------------------------------

#endif  // CEED_HIP_REF_BASIS_NONTENSOR_H
