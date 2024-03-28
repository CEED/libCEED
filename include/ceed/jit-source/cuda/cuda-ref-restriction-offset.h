// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA offset element restriction kernels

#include <ceed.h>

//------------------------------------------------------------------------------
// L-vector -> E-vector, standard (with offsets)
//------------------------------------------------------------------------------
extern "C" __global__ void OffsetNoTranspose(const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < RSTR_NUM_ELEM * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt ind      = indices[node];
    const CeedInt loc_node = node % RSTR_ELEM_SIZE;
    const CeedInt elem     = node / RSTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      v[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] = u[ind + comp * RSTR_COMP_STRIDE];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, standard (with offsets)
//------------------------------------------------------------------------------
#if !USE_DETERMINISTIC
extern "C" __global__ void OffsetTranspose(const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < RSTR_NUM_ELEM * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt ind      = indices[node];
    const CeedInt loc_node = node % RSTR_ELEM_SIZE;
    const CeedInt elem     = node / RSTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      atomicAdd(v + ind + comp * RSTR_COMP_STRIDE, u[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE]);
    }
  }
}
#else
extern "C" __global__ void OffsetTranspose(const CeedInt *__restrict__ l_vec_indices, const CeedInt *__restrict__ t_indices,
                                           const CeedInt *__restrict__ t_offsets, const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  CeedScalar value[RSTR_NUM_COMP];

  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < RSTR_NUM_NODES; i += blockDim.x * gridDim.x) {
    const CeedInt ind     = l_vec_indices[i];
    const CeedInt range_1 = t_offsets[i];
    const CeedInt range_N = t_offsets[i + 1];

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) value[comp] = 0.0;

    for (CeedInt j = range_1; j < range_N; j++) {
      const CeedInt t_ind    = t_indices[j];
      const CeedInt loc_node = t_ind % RSTR_ELEM_SIZE;
      const CeedInt elem     = t_ind / RSTR_ELEM_SIZE;

      for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
        value[comp] += u[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE];
      }
    }

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) v[ind + comp * RSTR_COMP_STRIDE] += value[comp];
  }
}
#endif
