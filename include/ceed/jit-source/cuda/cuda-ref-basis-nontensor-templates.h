// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA non-tensor product basis templates

#include <ceed.h>

//------------------------------------------------------------------------------
// Tensor contraction
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_COMP, int P, int Q>
inline __device__ void Contract(const CeedInt elem, const CeedInt strides_elem_U, const CeedInt strides_elem_V, const CeedInt strides_comp_U,
                                const CeedInt strides_comp_V, const CeedInt strides_q_comp_V, const CeedScalar *__restrict__ d_B,
                                const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  const CeedInt     t_id = threadIdx.x;
  const CeedScalar *U;
  CeedScalar        r_V[Q_COMP];
  // TODO load B in shared memory if blockDim.z > 1?

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    // Run with Q threads
    U = d_U + elem * strides_elem_U + comp * strides_comp_U;
    for (CeedInt d = 0; d < Q_COMP; d++) r_V[d] = 0.0;
    for (CeedInt i = 0; i < P; i++) {
      const CeedScalar val = U[i];

      for (CeedInt d = 0; d < Q_COMP; d++) r_V[d] += d_B[i + t_id * P + d * P * Q] * val;
    }
    for (CeedInt d = 0; d < Q_COMP; d++) {
      d_V[elem * strides_elem_V + comp * strides_comp_V + d * strides_q_comp_V + t_id] = r_V[d];
    }
  }
}

//------------------------------------------------------------------------------
// Tensor contraction transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_COMP, int P, int Q>
inline __device__ void ContractTranspose(const CeedInt elem, const CeedInt strides_elem_U, const CeedInt strides_elem_V, const CeedInt strides_comp_U,
                                         const CeedInt strides_comp_V, const CeedInt strides_q_comp_U, const CeedScalar *__restrict__ d_B,
                                         const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  const CeedInt     t_id = threadIdx.x;
  const CeedScalar *U;
  CeedScalar        r_V;
  // TODO load B in shared memory if blockDim.z > 1?

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    // Run with P threads
    r_V = 0.0;
    for (CeedInt d = 0; d < Q_COMP; d++) {
      U = d_U + elem * strides_elem_U + comp * strides_comp_U + d * strides_q_comp_U;
      for (CeedInt i = 0; i < Q; i++) r_V += d_B[t_id + i * P + d * P * Q] * U[i];
    }
    d_V[elem * strides_elem_V + comp * strides_comp_V + t_id] = r_V;
  }
}
