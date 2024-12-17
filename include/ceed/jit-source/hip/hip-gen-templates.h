// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for HIP backend macro and type definitions for JiT source
#include <ceed/types.h>

//------------------------------------------------------------------------------
// Load matrices for basis actions
//------------------------------------------------------------------------------
template <int P, int Q>
inline __device__ void LoadMatrix(SharedData_Hip &data, const CeedScalar *__restrict__ d_B, CeedScalar *B) {
  for (CeedInt i = data.t_id; i < P * Q; i += blockDim.x * blockDim.y * blockDim.z) B[i] = d_B[i];
}

//------------------------------------------------------------------------------
// AtPoints
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> single point
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int NUM_PTS>
inline __device__ void ReadPoint(SharedData_Hip &data, const CeedInt elem, const CeedInt p, const CeedInt points_in_elem,
                                 const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  const CeedInt ind = indices[p + elem * NUM_PTS];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    r_u[comp] = d_u[ind + comp * COMP_STRIDE];
  }
}

//------------------------------------------------------------------------------
// Single point -> L-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int NUM_PTS>
inline __device__ void WritePoint(SharedData_Hip &data, const CeedInt elem, const CeedInt p, const CeedInt points_in_elem,
                                  const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ r_u, CeedScalar *d_u) {
  if (p < points_in_elem) {
    const CeedInt ind = indices[p + elem * NUM_PTS];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_u[ind + comp * COMP_STRIDE] += r_u[comp];
    }
  }
}

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1d>
inline __device__ void ReadLVecStandard1d(SharedData_Hip &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                          const CeedScalar *__restrict__ d_u, CeedScalar *__restrict__ r_u) {
  if (data.t_id_x < P_1d) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = indices[node + elem * P_1d];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + COMP_STRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void ReadLVecStrided1d(SharedData_Hip &data, const CeedInt elem, const CeedScalar *__restrict__ d_u, CeedScalar *__restrict__ r_u) {
  if (data.t_id_x < P_1d) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1d>
inline __device__ void WriteLVecStandard1d(SharedData_Hip &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                           const CeedScalar *__restrict__ r_v, CeedScalar *__restrict__ d_v) {
  if (data.t_id_x < P_1d) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = indices[node + elem * P_1d];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) atomicAdd(&d_v[ind + COMP_STRIDE * comp], r_v[comp]);
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void WriteLVecStrided1d(SharedData_Hip &data, const CeedInt elem, const CeedScalar *__restrict__ r_v,
                                          CeedScalar *__restrict__ d_v) {
  if (data.t_id_x < P_1d) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) d_v[ind + comp * STRIDES_COMP] += r_v[comp];
  }
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1d>
inline __device__ void ReadLVecStandard2d(SharedData_Hip &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                          const CeedScalar *__restrict__ d_u, CeedScalar *__restrict__ r_u) {
  if (data.t_id_x < P_1d && data.t_id_y < P_1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1d;
    const CeedInt ind  = indices[node + elem * P_1d * P_1d];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + COMP_STRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void ReadLVecStrided2d(SharedData_Hip &data, const CeedInt elem, const CeedScalar *__restrict__ d_u, CeedScalar *__restrict__ r_u) {
  if (data.t_id_x < P_1d && data.t_id_y < P_1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1d;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1d>
inline __device__ void WriteLVecStandard2d(SharedData_Hip &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                           const CeedScalar *__restrict__ r_v, CeedScalar *__restrict__ d_v) {
  if (data.t_id_x < P_1d && data.t_id_y < P_1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1d;
    const CeedInt ind  = indices[node + elem * P_1d * P_1d];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) atomicAdd(&d_v[ind + COMP_STRIDE * comp], r_v[comp]);
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void WriteLVecStrided2d(SharedData_Hip &data, const CeedInt elem, const CeedScalar *__restrict__ r_v,
                                          CeedScalar *__restrict__ d_v) {
  if (data.t_id_x < P_1d && data.t_id_y < P_1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1d;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) d_v[ind + comp * STRIDES_COMP] += r_v[comp];
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1d>
inline __device__ void ReadLVecStandard3d(SharedData_Hip &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                          const CeedScalar *__restrict__ d_u, CeedScalar *__restrict__ r_u) {
  if (data.t_id_x < P_1d && data.t_id_y < P_1d) {
    for (CeedInt z = 0; z < P_1d; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y * P_1d + z * P_1d * P_1d;
      const CeedInt ind  = indices[node + elem * P_1d * P_1d * P_1d];

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[z + comp * P_1d] = d_u[ind + COMP_STRIDE * comp];
    }
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void ReadLVecStrided3d(SharedData_Hip &data, const CeedInt elem, const CeedScalar *__restrict__ d_u, CeedScalar *__restrict__ r_u) {
  if (data.t_id_x < P_1d && data.t_id_y < P_1d) {
    for (CeedInt z = 0; z < P_1d; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y * P_1d + z * P_1d * P_1d;
      const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[z + comp * P_1d] = d_u[ind + comp * STRIDES_COMP];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, offests provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int Q_1d>
inline __device__ void ReadEVecSliceStandard3d(SharedData_Hip &data, const CeedInt nquads, const CeedInt elem, const CeedInt q,
                                               const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ d_u,
                                               CeedScalar *__restrict__ r_u) {
  if (data.t_id_x < Q_1d && data.t_id_y < Q_1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * Q_1d + q * Q_1d * Q_1d;
    const CeedInt ind  = indices[node + elem * Q_1d * Q_1d * Q_1d];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + COMP_STRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void ReadEVecSliceStrided3d(SharedData_Hip &data, const CeedInt elem, const CeedInt q, const CeedScalar *__restrict__ d_u,
                                              CeedScalar *__restrict__ r_u) {
  if (data.t_id_x < Q_1d && data.t_id_y < Q_1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * Q_1d + q * Q_1d * Q_1d;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1d>
inline __device__ void WriteLVecStandard3d(SharedData_Hip &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                           const CeedScalar *__restrict__ r_v, CeedScalar *__restrict__ d_v) {
  if (data.t_id_x < P_1d && data.t_id_y < P_1d) {
    for (CeedInt z = 0; z < P_1d; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y * P_1d + z * P_1d * P_1d;
      const CeedInt ind  = indices[node + elem * P_1d * P_1d * P_1d];

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) atomicAdd(&d_v[ind + COMP_STRIDE * comp], r_v[z + comp * P_1d]);
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void WriteLVecStrided3d(SharedData_Hip &data, const CeedInt elem, const CeedScalar *__restrict__ r_v,
                                          CeedScalar *__restrict__ d_v) {
  if (data.t_id_x < P_1d && data.t_id_y < P_1d) {
    for (CeedInt z = 0; z < P_1d; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y * P_1d + z * P_1d * P_1d;
      const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) d_v[ind + comp * STRIDES_COMP] += r_v[z + comp * P_1d];
    }
  }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives computation
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_1d>
inline __device__ void GradColloSlice3d(SharedData_Hip &data, const CeedInt q, const CeedScalar *__restrict__ r_U, const CeedScalar *c_G,
                                        CeedScalar *__restrict__ r_V) {
  if (data.t_id_x < Q_1d && data.t_id_y < Q_1d) {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      data.slice[data.t_id_x + data.t_id_y * T_1D] = r_U[q + comp * Q_1d];
      __syncthreads();
      // X derivative
      r_V[comp + 0 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1d; i++) {
        r_V[comp + 0 * NUM_COMP] += c_G[i + data.t_id_x * Q_1d] * data.slice[i + data.t_id_y * T_1D];
      }
      // Y derivative
      r_V[comp + 1 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1d; i++) {
        r_V[comp + 1 * NUM_COMP] += c_G[i + data.t_id_y * Q_1d] * data.slice[data.t_id_x + i * T_1D];
      }
      // Z derivative
      r_V[comp + 2 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1d; i++) {
        r_V[comp + 2 * NUM_COMP] += c_G[i + q * Q_1d] * r_U[i + comp * Q_1d];
      }
      __syncthreads();
    }
  }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_1d>
inline __device__ void GradColloSliceTranspose3d(SharedData_Hip &data, const CeedInt q, const CeedScalar *__restrict__ r_U, const CeedScalar *c_G,
                                                 CeedScalar *__restrict__ r_V) {
  if (data.t_id_x < Q_1d && data.t_id_y < Q_1d) {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      // X derivative
      data.slice[data.t_id_x + data.t_id_y * T_1D] = r_U[comp + 0 * NUM_COMP];
      __syncthreads();
      for (CeedInt i = 0; i < Q_1d; i++) {
        r_V[q + comp * Q_1d] += c_G[data.t_id_x + i * Q_1d] * data.slice[i + data.t_id_y * T_1D];
      }
      __syncthreads();
      // Y derivative
      data.slice[data.t_id_x + data.t_id_y * T_1D] = r_U[comp + 1 * NUM_COMP];
      __syncthreads();
      for (CeedInt i = 0; i < Q_1d; i++) {
        r_V[q + comp * Q_1d] += c_G[data.t_id_y + i * Q_1d] * data.slice[data.t_id_x + i * T_1D];
      }
      __syncthreads();
      // Z derivative
      for (CeedInt i = 0; i < Q_1d; i++) {
        r_V[i + comp * Q_1d] += c_G[i + q * Q_1d] * r_U[comp + 2 * NUM_COMP];
      }
    }
  }
}
