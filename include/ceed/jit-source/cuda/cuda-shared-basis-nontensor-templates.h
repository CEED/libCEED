// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA shared memory non-tensor basis templates
#include <ceed/types.h>

//------------------------------------------------------------------------------
// 1D tensor contraction
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void Contract1d(SharedData_Cuda &data, const ScalarIn1 *U, const ScalarIn2 *B, ScalarOut *V) {
  data.slice[data.t_id_x] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.t_id_x * P_1D] * data.slice[i];  // Contract x direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 1D transpose tensor contraction
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractTranspose1d(SharedData_Cuda &data, const ScalarIn1 *U, const ScalarIn2 *B, ScalarOut *V) {
  data.slice[data.t_id_x] = *U;
  __syncthreads();
  if (data.t_id_x < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_x + i * P_1D] * data.slice[i];  // Contract x direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// Interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P, int Q, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void InterpNonTensor(SharedData_Cuda &data, const ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B, ScalarOut *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    Contract1d<NUM_COMP, P, Q>(data, &r_U[comp], c_B, &r_V[comp]);
  }
}

//------------------------------------------------------------------------------
// Interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P, int Q, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void InterpTransposeNonTensor(SharedData_Cuda &data, const ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B,
                                                ScalarOut *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    r_V[comp] = 0.0;
    ContractTranspose1d<NUM_COMP, P, Q>(data, &r_U[comp], c_B, &r_V[comp]);
  }
}

//------------------------------------------------------------------------------
// Derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int DIM, int P, int Q, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void GradNonTensor(SharedData_Cuda &data, const ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_G, ScalarOut *__restrict__ r_V) {
  for (CeedInt dim = 0; dim < DIM; dim++) {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      Contract1d<NUM_COMP, P, Q>(data, &r_U[comp], &c_G[dim * P * Q], &r_V[comp + dim * NUM_COMP]);
    }
  }
}

//------------------------------------------------------------------------------
// Derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int DIM, int P, int Q, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void GradTransposeNonTensor(SharedData_Cuda &data, const ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_G,
                                              ScalarOut *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_V[comp] = 0.0;
  for (CeedInt dim = 0; dim < DIM; dim++) {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      ContractTranspose1d<NUM_COMP, P, Q>(data, &r_U[comp + dim * NUM_COMP], &c_G[dim * P * Q], &r_V[comp]);
    }
  }
}

//------------------------------------------------------------------------------
// Quadrature weights
//------------------------------------------------------------------------------
template <int P, int Q, class ScalarIn, class ScalarOut>
inline __device__ void WeightNonTensor(SharedData_Cuda &data, const ScalarIn *__restrict__ q_weight, ScalarOut *w) {
  *w = (data.t_id_x < Q) ? q_weight[data.t_id_x] : 0.0;
}
