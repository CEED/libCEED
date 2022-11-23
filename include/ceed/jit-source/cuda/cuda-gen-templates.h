// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA backend macro and type definitions for JiT source
#ifndef _ceed_cuda_gen_templates_h
#define _ceed_cuda_gen_templates_h

#include <ceed/types.h>

//------------------------------------------------------------------------------
// Load matrices for basis actions
//------------------------------------------------------------------------------
template <int P, int Q>
inline __device__ void loadMatrix(SharedData_Cuda &data, const CeedScalar *__restrict__ d_B, CeedScalar *B) {
  for (CeedInt i = data.t_id; i < P * Q; i += blockDim.x * blockDim.y * blockDim.z) B[i] = d_B[i];
}

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void readDofsOffset1d(SharedData_Cuda &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                        const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P1d) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = indices[node + elem * P1d];
    for (CeedInt comp = 0; comp < NCOMP; ++comp) r_u[comp] = d_u[ind + COMPSTRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void readDofsStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P1d) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NCOMP; ++comp) r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void writeDofsOffset1d(SharedData_Cuda &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                         const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P1d) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = indices[node + elem * P1d];
    for (CeedInt comp = 0; comp < NCOMP; ++comp) atomicAdd(&d_v[ind + COMPSTRIDE * comp], r_v[comp]);
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void writeDofsStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P1d) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NCOMP; ++comp) d_v[ind + comp * STRIDES_COMP] += r_v[comp];
  }
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void readDofsOffset2d(SharedData_Cuda &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                        const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P1d && data.t_id_y < P1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * P1d;
    const CeedInt ind  = indices[node + elem * P1d * P1d];
    for (CeedInt comp = 0; comp < NCOMP; ++comp) r_u[comp] = d_u[ind + COMPSTRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void readDofsStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P1d && data.t_id_y < P1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * P1d;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NCOMP; ++comp) r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void writeDofsOffset2d(SharedData_Cuda &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                         const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P1d && data.t_id_y < P1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * P1d;
    const CeedInt ind  = indices[node + elem * P1d * P1d];
    for (CeedInt comp = 0; comp < NCOMP; ++comp) atomicAdd(&d_v[ind + COMPSTRIDE * comp], r_v[comp]);
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void writeDofsStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P1d && data.t_id_y < P1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * P1d;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NCOMP; ++comp) d_v[ind + comp * STRIDES_COMP] += r_v[comp];
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
// TODO: remove "Dofs" and "Quads" in the following function names?
//   - readDofsOffset3d -> readOffset3d ?
//   - readDofsStrided3d -> readStrided3d ?
//   - readSliceQuadsOffset3d -> readSliceOffset3d ?
//   - readSliceQuadsStrided3d -> readSliceStrided3d ?
//   - writeDofsOffset3d -> writeOffset3d ?
//   - writeDofsStrided3d -> writeStrided3d ?
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void readDofsOffset3d(SharedData_Cuda &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                        const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P1d && data.t_id_y < P1d)
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt node = data.t_id_x + data.t_id_y * P1d + z * P1d * P1d;
      const CeedInt ind  = indices[node + elem * P1d * P1d * P1d];
      for (CeedInt comp = 0; comp < NCOMP; ++comp) r_u[z + comp * P1d] = d_u[ind + COMPSTRIDE * comp];
    }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void readDofsStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P1d && data.t_id_y < P1d)
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt node = data.t_id_x + data.t_id_y * P1d + z * P1d * P1d;
      const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;
      for (CeedInt comp = 0; comp < NCOMP; ++comp) r_u[z + comp * P1d] = d_u[ind + comp * STRIDES_COMP];
    }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, offests provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int Q1d>
inline __device__ void readSliceQuadsOffset3d(SharedData_Cuda &data, const CeedInt nquads, const CeedInt elem, const CeedInt q,
                                              const CeedInt *__restrict__ indices, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < Q1d && data.t_id_y < Q1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * Q1d + q * Q1d * Q1d;
    const CeedInt ind  = indices[node + elem * Q1d * Q1d * Q1d];
    ;
    for (CeedInt comp = 0; comp < NCOMP; ++comp) r_u[comp] = d_u[ind + COMPSTRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int Q1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void readSliceQuadsStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt q, const CeedScalar *__restrict__ d_u,
                                               CeedScalar *r_u) {
  if (data.t_id_x < Q1d && data.t_id_y < Q1d) {
    const CeedInt node = data.t_id_x + data.t_id_y * Q1d + q * Q1d * Q1d;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;
    for (CeedInt comp = 0; comp < NCOMP; ++comp) r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline __device__ void writeDofsOffset3d(SharedData_Cuda &data, const CeedInt nnodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                         const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P1d && data.t_id_y < P1d)
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt node = data.t_id_x + data.t_id_y * P1d + z * P1d * P1d;
      const CeedInt ind  = indices[node + elem * P1d * P1d * P1d];
      for (CeedInt comp = 0; comp < NCOMP; ++comp) atomicAdd(&d_v[ind + COMPSTRIDE * comp], r_v[z + comp * P1d]);
    }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM>
inline __device__ void writeDofsStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P1d && data.t_id_y < P1d)
    for (CeedInt z = 0; z < P1d; ++z) {
      const CeedInt node = data.t_id_x + data.t_id_y * P1d + z * P1d * P1d;
      const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;
      for (CeedInt comp = 0; comp < NCOMP; ++comp) d_v[ind + comp * STRIDES_COMP] += r_v[z + comp * P1d];
    }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives computation
//------------------------------------------------------------------------------
template <int NCOMP, int Q1d>
inline __device__ void gradCollo3d(SharedData_Cuda &data, const CeedInt q, const CeedScalar *__restrict__ r_U, const CeedScalar *c_G,
                                   CeedScalar *__restrict__ r_V) {
  if (data.t_id_x < Q1d && data.t_id_y < Q1d) {
    for (CeedInt comp = 0; comp < NCOMP; ++comp) {
      data.slice[data.t_id_x + data.t_id_y * T_1D] = r_U[q + comp * Q1d];
      __syncthreads();
      // X derivative
      r_V[comp + 0 * NCOMP] = 0.0;
      for (CeedInt i = 0; i < Q1d; ++i)
        r_V[comp + 0 * NCOMP] += c_G[i + data.t_id_x * Q1d] * data.slice[i + data.t_id_y * T_1D];  // Contract x direction (X derivative)
      // Y derivative
      r_V[comp + 1 * NCOMP] = 0.0;
      for (CeedInt i = 0; i < Q1d; ++i)
        r_V[comp + 1 * NCOMP] += c_G[i + data.t_id_y * Q1d] * data.slice[data.t_id_x + i * T_1D];  // Contract y direction (Y derivative)
      // Z derivative
      r_V[comp + 2 * NCOMP] = 0.0;
      for (CeedInt i = 0; i < Q1d; ++i) r_V[comp + 2 * NCOMP] += c_G[i + q * Q1d] * r_U[i + comp * Q1d];  // Contract z direction (Z derivative)
      __syncthreads();
    }
  }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives transpose
//------------------------------------------------------------------------------
template <int NCOMP, int Q1d>
inline __device__ void gradColloTranspose3d(SharedData_Cuda &data, const CeedInt q, const CeedScalar *__restrict__ r_U, const CeedScalar *c_G,
                                            CeedScalar *__restrict__ r_V) {
  if (data.t_id_x < Q1d && data.t_id_y < Q1d) {
    for (CeedInt comp = 0; comp < NCOMP; ++comp) {
      // X derivative
      data.slice[data.t_id_x + data.t_id_y * T_1D] = r_U[comp + 0 * NCOMP];
      __syncthreads();
      for (CeedInt i = 0; i < Q1d; ++i)
        r_V[q + comp * Q1d] += c_G[data.t_id_x + i * Q1d] * data.slice[i + data.t_id_y * T_1D];  // Contract x direction (X derivative)
      __syncthreads();
      // Y derivative
      data.slice[data.t_id_x + data.t_id_y * T_1D] = r_U[comp + 1 * NCOMP];
      __syncthreads();
      for (CeedInt i = 0; i < Q1d; ++i)
        r_V[q + comp * Q1d] += c_G[data.t_id_y + i * Q1d] * data.slice[data.t_id_x + i * T_1D];  // Contract y direction (Y derivative)
      __syncthreads();
      // Z derivative
      for (CeedInt i = 0; i < Q1d; ++i)
        r_V[i + comp * Q1d] += c_G[i + q * Q1d] * r_U[comp + 2 * NCOMP];  // PARTIAL contract z direction (Z derivative)
    }
  }
}

#endif
