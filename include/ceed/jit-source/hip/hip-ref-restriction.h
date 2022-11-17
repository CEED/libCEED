// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void StridedNoTranspose(const CeedInt num_elem, const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RESTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt loc_node = node % RESTR_ELEM_SIZE;
    const CeedInt elem     = node / RESTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++)
      v[loc_node + comp * RESTR_ELEM_SIZE * RESTR_NUM_ELEM + elem * RESTR_ELEM_SIZE] =
          u[loc_node * RESTR_STRIDE_NODES + comp * RESTR_STRIDE_COMP + elem * RESTR_STRIDE_ELEM];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
extern "C" __global__ void OffsetNoTranspose(const CeedInt num_elem, const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ u,
                                             CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RESTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt ind      = indices[node];
    const CeedInt loc_node = node % RESTR_ELEM_SIZE;
    const CeedInt elem     = node / RESTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++)
      v[loc_node + comp * RESTR_ELEM_SIZE * RESTR_NUM_ELEM + elem * RESTR_ELEM_SIZE] = u[ind + comp * RESTR_COMP_STRIDE];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void StridedTranspose(const CeedInt num_elem, const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < num_elem * RESTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt loc_node = node % RESTR_ELEM_SIZE;
    const CeedInt elem     = node / RESTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++)
      v[loc_node * RESTR_STRIDE_NODES + comp * RESTR_STRIDE_COMP + elem * RESTR_STRIDE_ELEM] +=
          u[loc_node + comp * RESTR_ELEM_SIZE * RESTR_NUM_ELEM + elem * RESTR_ELEM_SIZE];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
extern "C" __global__ void OffsetTranspose(const CeedInt *__restrict__ l_vec_indices, const CeedInt *__restrict__ t_indices,
                                           const CeedInt *__restrict__ t_offsets, const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  CeedScalar value[RESTR_NUM_COMP];

  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x; i < RESTR_NUM_NODES; i += blockDim.x * gridDim.x) {
    const CeedInt ind     = l_vec_indices[i];
    const CeedInt range_1 = t_offsets[i];
    const CeedInt range_N = t_offsets[i + 1];

    for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++) value[comp] = 0.0;

    for (CeedInt j = range_1; j < range_N; ++j) {
      const CeedInt t_ind    = t_indices[j];
      CeedInt       loc_node = t_ind % RESTR_ELEM_SIZE;
      CeedInt       elem     = t_ind / RESTR_ELEM_SIZE;

      for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++)
        value[comp] += u[loc_node + comp * RESTR_ELEM_SIZE * RESTR_NUM_ELEM + elem * RESTR_ELEM_SIZE];
    }

    for (CeedInt comp = 0; comp < RESTR_NUM_COMP; comp++) v[ind + comp * RESTR_COMP_STRIDE] += value[comp];
  }
}

//------------------------------------------------------------------------------
