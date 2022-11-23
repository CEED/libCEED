// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

//------------------------------------------------------------------------------
// Shared mem kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Sum input into output
//------------------------------------------------------------------------------
inline __device__ void add(CeedScalar *r_V, const CeedScalar *r_U) {
  for (CeedInt i = 0; i < BASIS_P_1D; i++) r_V[i] += r_U[i];
}

//------------------------------------------------------------------------------
// Load matrices for basis actions
//------------------------------------------------------------------------------
inline __device__ void loadMatrix(const CeedScalar *d_B, CeedScalar *B) {
  CeedInt tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  for (CeedInt i = tid; i < BASIS_P_1D * BASIS_Q_1D; i += blockDim.x * blockDim.y * blockDim.z) B[i] = d_B[i];
}

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Read DoFs
//------------------------------------------------------------------------------
inline __device__ void readDofs1d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedInt comp,
                                  const CeedInt num_elem, const CeedScalar *d_U, CeedScalar *slice) {
  for (CeedInt i = 0; i < BASIS_P_1D; i++) slice[i + t_id_z * BASIS_T_1D] = d_U[i + elem * BASIS_P_1D + comp * BASIS_P_1D * num_elem];
  for (CeedInt i = BASIS_P_1D; i < BASIS_Q_1D; i++) slice[i + t_id_z * BASIS_T_1D] = 0.0;
}

//------------------------------------------------------------------------------
// Write DoFs
//------------------------------------------------------------------------------
inline __device__ void writeDofs1d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt comp, const CeedInt num_elem,
                                   const CeedScalar &r_V, CeedScalar *d_V) {
  if (t_id_x < BASIS_P_1D) d_V[t_id_x + elem * BASIS_P_1D + comp * BASIS_P_1D * num_elem] = r_V;
}

//------------------------------------------------------------------------------
// Read quadrature point data
//------------------------------------------------------------------------------
inline __device__ void readQuads1d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedInt comp,
                                   const CeedInt dim, const CeedInt num_elem, const CeedScalar *d_U, CeedScalar *slice) {
  for (CeedInt i = 0; i < BASIS_Q_1D; i++)
    slice[i + t_id_z * BASIS_T_1D] = d_U[i + elem * BASIS_Q_1D + comp * BASIS_Q_1D * num_elem + dim * BASIS_NUM_COMP * num_elem * BASIS_Q_1D];
  for (CeedInt i = BASIS_Q_1D; i < BASIS_P_1D; i++) slice[i + t_id_z * BASIS_T_1D] = 0.0;
}

//------------------------------------------------------------------------------
// Write quadrature point data
//------------------------------------------------------------------------------
inline __device__ void writeQuads1d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt comp, const CeedInt dim,
                                    const CeedInt num_elem, const CeedScalar &r_V, CeedScalar *d_V) {
  if (t_id_x < BASIS_Q_1D) d_V[t_id_x + elem * BASIS_Q_1D + comp * BASIS_Q_1D * num_elem + dim * BASIS_NUM_COMP * num_elem * BASIS_Q_1D] = r_V;
}

//------------------------------------------------------------------------------
// 1D tensor contraction
//------------------------------------------------------------------------------
inline __device__ void ContractX1d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar &U,
                                   const CeedScalar *B, CeedScalar &V) {
  V = 0.0;
  for (CeedInt i = 0; i < BASIS_P_1D; i++) V += B[i + t_id_x * BASIS_P_1D] * slice[i + t_id_z * BASIS_T_1D];  // Contract x direction
}

//------------------------------------------------------------------------------
// 1D transpose tensor contraction
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeX1d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar &U,
                                            const CeedScalar *B, CeedScalar &V) {
  V = 0.0;
  for (CeedInt i = 0; i < BASIS_Q_1D; i++) V += B[t_id_x + i * BASIS_P_1D] * slice[i + t_id_z * BASIS_T_1D];  // Contract x direction
}

//------------------------------------------------------------------------------
// 1D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void interp1d(const CeedInt num_elem, const CeedInt transpose, const CeedScalar *s_B, const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V, CeedScalar *slice) {
  CeedScalar r_V;
  CeedScalar r_t;

  const CeedInt t_id_x = threadIdx.x;
  const CeedInt t_id_y = threadIdx.y;
  const CeedInt t_id_z = threadIdx.z;

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
      if (transpose) {
        readQuads1d(elem, t_id_x, t_id_y, t_id_z, comp, 0, num_elem, d_U, slice);
        ContractTransposeX1d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_V);
        writeDofs1d(elem, t_id_x, t_id_y, comp, num_elem, r_V, d_V);
      } else {
        readDofs1d(elem, t_id_x, t_id_y, t_id_z, comp, num_elem, d_U, slice);
        ContractX1d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_V);
        writeQuads1d(elem, t_id_x, t_id_y, comp, 0, num_elem, r_V, d_V);
      }
    }
  }
}

//------------------------------------------------------------------------------
// 1D derivatives at quadrature points
//------------------------------------------------------------------------------
inline __device__ void grad1d(const CeedInt num_elem, const CeedInt transpose, const CeedScalar *s_B, const CeedScalar *s_G,
                              const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V, CeedScalar *slice) {
  CeedScalar r_U;
  CeedScalar r_V;

  const CeedInt t_id_x = threadIdx.x;
  const CeedInt t_id_y = threadIdx.y;
  const CeedInt t_id_z = threadIdx.z;
  int           dim;

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
      if (transpose) {
        dim = 0;
        readQuads1d(elem, t_id_x, t_id_y, t_id_z, comp, dim, num_elem, d_U, slice);
        ContractTransposeX1d(slice, t_id_x, t_id_y, t_id_z, r_U, s_G, r_V);
        writeDofs1d(elem, t_id_x, t_id_y, comp, num_elem, r_V, d_V);
      } else {
        readDofs1d(elem, t_id_x, t_id_y, t_id_z, comp, num_elem, d_U, slice);
        ContractX1d(slice, t_id_x, t_id_y, t_id_z, r_U, s_G, r_V);
        dim = 0;
        writeQuads1d(elem, t_id_x, t_id_y, comp, dim, num_elem, r_V, d_V);
      }
    }
  }
}

//------------------------------------------------------------------------------
// 1D Quadrature weights
//------------------------------------------------------------------------------
__device__ void weight1d(const CeedInt num_elem, const CeedScalar *q_weight_1d, CeedScalar *w) {
  const CeedInt    tid    = threadIdx.x;
  const CeedScalar weight = q_weight_1d[tid];
  for (CeedInt elem = blockIdx.x * blockDim.y + threadIdx.y; elem < num_elem; elem += gridDim.x * blockDim.y) {
    const CeedInt ind = elem * BASIS_Q_1D + tid;
    w[ind]            = weight;
  }
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Read DoFs
//------------------------------------------------------------------------------
inline __device__ void readDofs2d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt comp, const CeedInt num_elem,
                                  const CeedScalar *d_U, CeedScalar &U) {
  U = (t_id_x < BASIS_P_1D && t_id_y < BASIS_P_1D)
          ? d_U[t_id_x + t_id_y * BASIS_P_1D + elem * BASIS_P_1D * BASIS_P_1D + comp * BASIS_P_1D * BASIS_P_1D * num_elem]
          : 0.0;
}

//------------------------------------------------------------------------------
// Write DoFs
//------------------------------------------------------------------------------
inline __device__ void writeDofs2d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt comp, const CeedInt num_elem,
                                   const CeedScalar &r_V, CeedScalar *d_V) {
  if (t_id_x < BASIS_P_1D && t_id_y < BASIS_P_1D)
    d_V[t_id_x + t_id_y * BASIS_P_1D + elem * BASIS_P_1D * BASIS_P_1D + comp * BASIS_P_1D * BASIS_P_1D * num_elem] = r_V;
}

//------------------------------------------------------------------------------
// Read quadrature point data
//------------------------------------------------------------------------------
inline __device__ void readQuads2d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt comp, const CeedInt dim,
                                   const CeedInt num_elem, const CeedScalar *d_U, CeedScalar &U) {
  U = (t_id_x < BASIS_Q_1D && t_id_y < BASIS_Q_1D)
          ? d_U[t_id_x + t_id_y * BASIS_Q_1D + elem * BASIS_Q_1D * BASIS_Q_1D + comp * BASIS_Q_1D * BASIS_Q_1D * num_elem +
                dim * BASIS_NUM_COMP * num_elem * BASIS_Q_1D * BASIS_Q_1D]
          : 0.0;
}

//------------------------------------------------------------------------------
// Write quadrature point data
//------------------------------------------------------------------------------
inline __device__ void writeQuads2d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt comp, const CeedInt dim,
                                    const CeedInt num_elem, const CeedScalar &r_V, CeedScalar *d_V) {
  if (t_id_x < BASIS_Q_1D && t_id_y < BASIS_Q_1D)
    d_V[t_id_x + t_id_y * BASIS_Q_1D + elem * BASIS_Q_1D * BASIS_Q_1D + comp * BASIS_Q_1D * BASIS_Q_1D * num_elem +
        dim * BASIS_NUM_COMP * num_elem * BASIS_Q_1D * BASIS_Q_1D] = r_V;
}

//------------------------------------------------------------------------------
// 2D tensor contraction x
//------------------------------------------------------------------------------
inline __device__ void ContractX2d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar &U,
                                   const CeedScalar *B, CeedScalar &V) {
  slice[t_id_x + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D] = U;
  __syncthreads();
  V = 0.0;
  if (t_id_x < BASIS_Q_1D)
    for (CeedInt i = 0; i < BASIS_P_1D; i++)
      V += B[i + t_id_x * BASIS_P_1D] * slice[i + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D];  // Contract x direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D tensor contraction y
//------------------------------------------------------------------------------
inline __device__ void ContractY2d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar &U,
                                   const CeedScalar *B, CeedScalar &V) {
  slice[t_id_x + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D] = U;
  __syncthreads();
  V = 0.0;
  if (t_id_y < BASIS_Q_1D)
    for (CeedInt i = 0; i < BASIS_P_1D; i++)
      V += B[i + t_id_y * BASIS_P_1D] * slice[t_id_x + i * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D];  // Contract y direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contraction y
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeY2d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar &U,
                                            const CeedScalar *B, CeedScalar &V) {
  slice[t_id_x + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D] = U;
  __syncthreads();
  V = 0.0;
  if (t_id_y < BASIS_P_1D)
    for (CeedInt i = 0; i < BASIS_Q_1D; i++)
      V += B[t_id_y + i * BASIS_P_1D] * slice[t_id_x + i * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D];  // Contract y direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contraction x
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeX2d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar &U,
                                            const CeedScalar *B, CeedScalar &V) {
  slice[t_id_x + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D] = U;
  __syncthreads();
  V = 0.0;
  if (t_id_x < BASIS_P_1D)
    for (CeedInt i = 0; i < BASIS_Q_1D; i++)
      V += B[t_id_x + i * BASIS_P_1D] * slice[i + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D];  // Contract x direction
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void interp2d(const CeedInt num_elem, const CeedInt transpose, const CeedScalar *s_B, const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V, CeedScalar *slice) {
  CeedScalar r_V;
  CeedScalar r_t;

  const CeedInt t_id_x        = threadIdx.x;
  const CeedInt t_id_y        = threadIdx.y;
  const CeedInt t_id_z        = threadIdx.z;
  const CeedInt blockElem     = t_id_z / BASIS_NUM_COMP;
  const CeedInt elemsPerBlock = blockDim.z / BASIS_NUM_COMP;
  const CeedInt comp          = t_id_z % BASIS_NUM_COMP;

  for (CeedInt elem = blockIdx.x * elemsPerBlock + blockElem; elem < num_elem; elem += gridDim.x * elemsPerBlock) {
    const CeedInt comp = t_id_z % BASIS_NUM_COMP;
    r_V                = 0.0;
    r_t                = 0.0;
    if (transpose) {
      readQuads2d(elem, t_id_x, t_id_y, comp, 0, num_elem, d_U, r_V);
      ContractTransposeY2d(slice, t_id_x, t_id_y, t_id_z, r_V, s_B, r_t);
      ContractTransposeX2d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_V);
      writeDofs2d(elem, t_id_x, t_id_y, comp, num_elem, r_V, d_V);
    } else {
      readDofs2d(elem, t_id_x, t_id_y, comp, num_elem, d_U, r_V);
      ContractX2d(slice, t_id_x, t_id_y, t_id_z, r_V, s_B, r_t);
      ContractY2d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_V);
      writeQuads2d(elem, t_id_x, t_id_y, comp, 0, num_elem, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points
//------------------------------------------------------------------------------
inline __device__ void grad2d(const CeedInt num_elem, const CeedInt transpose, const CeedScalar *s_B, const CeedScalar *s_G,
                              const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V, CeedScalar *slice) {
  CeedScalar r_U;
  CeedScalar r_V;
  CeedScalar r_t;

  const CeedInt t_id_x        = threadIdx.x;
  const CeedInt t_id_y        = threadIdx.y;
  const CeedInt t_id_z        = threadIdx.z;
  const CeedInt blockElem     = t_id_z / BASIS_NUM_COMP;
  const CeedInt elemsPerBlock = blockDim.z / BASIS_NUM_COMP;
  const CeedInt comp          = t_id_z % BASIS_NUM_COMP;
  int           dim;

  for (CeedInt elem = blockIdx.x * elemsPerBlock + blockElem; elem < num_elem; elem += gridDim.x * elemsPerBlock) {
    if (transpose) {
      dim = 0;
      readQuads2d(elem, t_id_x, t_id_y, comp, dim, num_elem, d_U, r_U);
      ContractTransposeY2d(slice, t_id_x, t_id_y, t_id_z, r_U, s_B, r_t);
      ContractTransposeX2d(slice, t_id_x, t_id_y, t_id_z, r_t, s_G, r_V);
      dim = 1;
      readQuads2d(elem, t_id_x, t_id_y, comp, dim, num_elem, d_U, r_U);
      ContractTransposeY2d(slice, t_id_x, t_id_y, t_id_z, r_U, s_G, r_t);
      ContractTransposeX2d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_U);
      r_V += r_U;
      writeDofs2d(elem, t_id_x, t_id_y, comp, num_elem, r_V, d_V);
    } else {
      readDofs2d(elem, t_id_x, t_id_y, comp, num_elem, d_U, r_U);
      ContractX2d(slice, t_id_x, t_id_y, t_id_z, r_U, s_G, r_t);
      ContractY2d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_V);
      dim = 0;
      writeQuads2d(elem, t_id_x, t_id_y, comp, dim, num_elem, r_V, d_V);
      ContractX2d(slice, t_id_x, t_id_y, t_id_z, r_U, s_B, r_t);
      ContractY2d(slice, t_id_x, t_id_y, t_id_z, r_t, s_G, r_V);
      dim = 1;
      writeQuads2d(elem, t_id_x, t_id_y, comp, dim, num_elem, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
__device__ void weight2d(const CeedInt num_elem, const CeedScalar *q_weight_1d, CeedScalar *w) {
  const CeedInt    i      = threadIdx.x;
  const CeedInt    j      = threadIdx.y;
  const CeedScalar weight = q_weight_1d[i] * q_weight_1d[j];
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    const CeedInt ind = elem * BASIS_Q_1D * BASIS_Q_1D + i + j * BASIS_Q_1D;
    w[ind]            = weight;
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Read DoFs
//------------------------------------------------------------------------------
inline __device__ void readDofs3d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt comp, const CeedInt num_elem,
                                  const CeedScalar *d_U, CeedScalar *r_U) {
  for (CeedInt i = 0; i < BASIS_P_1D; i++)
    r_U[i] = (t_id_x < BASIS_P_1D && t_id_y < BASIS_P_1D)
                 ? d_U[t_id_x + t_id_y * BASIS_P_1D + i * BASIS_P_1D * BASIS_P_1D + elem * BASIS_P_1D * BASIS_P_1D * BASIS_P_1D +
                       comp * BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem]
                 : 0.0;
  for (CeedInt i = BASIS_P_1D; i < BASIS_Q_1D; i++) r_U[i] = 0.0;
}

//------------------------------------------------------------------------------
// Write DoFs
//------------------------------------------------------------------------------
inline __device__ void writeDofs3d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt comp, const CeedInt num_elem,
                                   const CeedScalar *r_V, CeedScalar *d_V) {
  if (t_id_x < BASIS_P_1D && t_id_y < BASIS_P_1D) {
    for (CeedInt i = 0; i < BASIS_P_1D; i++)
      d_V[t_id_x + t_id_y * BASIS_P_1D + i * BASIS_P_1D * BASIS_P_1D + elem * BASIS_P_1D * BASIS_P_1D * BASIS_P_1D +
          comp * BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem] = r_V[i];
  }
}

//------------------------------------------------------------------------------
// Read quadrature point data
//------------------------------------------------------------------------------
inline __device__ void readQuads3d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt comp, const CeedInt dim,
                                   const CeedInt num_elem, const CeedScalar *d_U, CeedScalar *r_U) {
  for (CeedInt i = 0; i < BASIS_Q_1D; i++)
    r_U[i] =
        (t_id_x < BASIS_Q_1D && t_id_y < BASIS_Q_1D)
            ? d_U[t_id_x + t_id_y * BASIS_Q_1D + i * BASIS_Q_1D * BASIS_Q_1D + elem * BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D +
                  comp * BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D * num_elem + dim * BASIS_NUM_COMP * num_elem * BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D]
            : 0.0;
  for (CeedInt i = BASIS_Q_1D; i < BASIS_P_1D; i++) r_U[i] = 0.0;
}

//------------------------------------------------------------------------------
// Write quadrature point data
//------------------------------------------------------------------------------
inline __device__ void writeQuads3d(const CeedInt elem, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt comp, const CeedInt dim,
                                    const CeedInt num_elem, const CeedScalar *r_V, CeedScalar *d_V) {
  if (t_id_x < BASIS_Q_1D && t_id_y < BASIS_Q_1D) {
    for (CeedInt i = 0; i < BASIS_Q_1D; i++)
      d_V[t_id_x + t_id_y * BASIS_Q_1D + i * BASIS_Q_1D * BASIS_Q_1D + elem * BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D +
          comp * BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D * num_elem + dim * BASIS_NUM_COMP * num_elem * BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D] = r_V[i];
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract x
//------------------------------------------------------------------------------
inline __device__ void ContractX3d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar *U,
                                   const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < BASIS_P_1D; k++) {
    slice[t_id_x + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (t_id_x < BASIS_Q_1D && t_id_y < BASIS_P_1D)
      for (CeedInt i = 0; i < BASIS_P_1D; i++)
        V[k] += B[i + t_id_x * BASIS_P_1D] * slice[i + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D];  // Contract x direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract y
//------------------------------------------------------------------------------
inline __device__ void ContractY3d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar *U,
                                   const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < BASIS_P_1D; k++) {
    slice[t_id_x + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (t_id_x < BASIS_Q_1D && t_id_y < BASIS_Q_1D)
      for (CeedInt i = 0; i < BASIS_P_1D; i++)
        V[k] += B[i + t_id_y * BASIS_P_1D] * slice[t_id_x + i * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D];  // Contract y direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract z
//------------------------------------------------------------------------------
inline __device__ void ContractZ3d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar *U,
                                   const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < BASIS_Q_1D; k++) {
    V[k] = 0.0;
    if (t_id_x < BASIS_Q_1D && t_id_y < BASIS_Q_1D)
      for (CeedInt i = 0; i < BASIS_P_1D; i++) V[k] += B[i + k * BASIS_P_1D] * U[i];  // Contract z direction
  }
  for (CeedInt k = BASIS_Q_1D; k < BASIS_P_1D; k++) V[k] = 0.0;
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract z
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeZ3d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar *U,
                                            const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < BASIS_P_1D; k++) {
    V[k] = 0.0;
    if (t_id_x < BASIS_Q_1D && t_id_y < BASIS_Q_1D)
      for (CeedInt i = 0; i < BASIS_Q_1D; i++) V[k] += B[k + i * BASIS_P_1D] * U[i];  // Contract z direction
  }
  for (CeedInt k = BASIS_P_1D; k < BASIS_Q_1D; k++) V[k] = 0.0;
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeY3d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar *U,
                                            const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < BASIS_P_1D; k++) {
    slice[t_id_x + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (t_id_x < BASIS_Q_1D && t_id_y < BASIS_P_1D)
      for (CeedInt i = 0; i < BASIS_Q_1D; i++)
        V[k] += B[t_id_y + i * BASIS_P_1D] * slice[t_id_x + i * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D];  // Contract y direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract x
//------------------------------------------------------------------------------
inline __device__ void ContractTransposeX3d(CeedScalar *slice, const CeedInt t_id_x, const CeedInt t_id_y, const CeedInt t_id_z, const CeedScalar *U,
                                            const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < BASIS_P_1D; k++) {
    slice[t_id_x + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (t_id_x < BASIS_P_1D && t_id_y < BASIS_P_1D)
      for (CeedInt i = 0; i < BASIS_Q_1D; i++)
        V[k] += B[t_id_x + i * BASIS_P_1D] * slice[i + t_id_y * BASIS_T_1D + t_id_z * BASIS_T_1D * BASIS_T_1D];  // Contract x direction
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void interp3d(const CeedInt num_elem, const CeedInt transpose, const CeedScalar *s_B, const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V, CeedScalar *slice) {
  CeedScalar r_V[BASIS_T_1D];
  CeedScalar r_t[BASIS_T_1D];

  const CeedInt t_id_x        = threadIdx.x;
  const CeedInt t_id_y        = threadIdx.y;
  const CeedInt t_id_z        = threadIdx.z;
  const CeedInt blockElem     = t_id_z / BASIS_NUM_COMP;
  const CeedInt elemsPerBlock = blockDim.z / BASIS_NUM_COMP;
  const CeedInt comp          = t_id_z % BASIS_NUM_COMP;

  for (CeedInt elem = blockIdx.x * elemsPerBlock + blockElem; elem < num_elem; elem += gridDim.x * elemsPerBlock) {
    for (CeedInt i = 0; i < BASIS_T_1D; i++) {
      r_V[i] = 0.0;
      r_t[i] = 0.0;
    }
    if (transpose) {
      readQuads3d(elem, t_id_x, t_id_y, comp, 0, num_elem, d_U, r_V);
      ContractTransposeZ3d(slice, t_id_x, t_id_y, t_id_z, r_V, s_B, r_t);
      ContractTransposeY3d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_V);
      ContractTransposeX3d(slice, t_id_x, t_id_y, t_id_z, r_V, s_B, r_t);
      writeDofs3d(elem, t_id_x, t_id_y, comp, num_elem, r_t, d_V);
    } else {
      readDofs3d(elem, t_id_x, t_id_y, comp, num_elem, d_U, r_V);
      ContractX3d(slice, t_id_x, t_id_y, t_id_z, r_V, s_B, r_t);
      ContractY3d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_V);
      ContractZ3d(slice, t_id_x, t_id_y, t_id_z, r_V, s_B, r_t);
      writeQuads3d(elem, t_id_x, t_id_y, comp, 0, num_elem, r_t, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
inline __device__ void grad3d(const CeedInt num_elem, const CeedInt transpose, const CeedScalar *s_B, const CeedScalar *s_G,
                              const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V, CeedScalar *slice) {
  // Use BASIS_P_1D for one of these
  CeedScalar r_U[BASIS_T_1D];
  CeedScalar r_V[BASIS_T_1D];
  CeedScalar r_t[BASIS_T_1D];

  const CeedInt t_id_x        = threadIdx.x;
  const CeedInt t_id_y        = threadIdx.y;
  const CeedInt t_id_z        = threadIdx.z;
  const CeedInt blockElem     = t_id_z / BASIS_NUM_COMP;
  const CeedInt elemsPerBlock = blockDim.z / BASIS_NUM_COMP;
  const CeedInt comp          = t_id_z % BASIS_NUM_COMP;
  int           dim;

  for (CeedInt elem = blockIdx.x * elemsPerBlock + blockElem; elem < num_elem; elem += gridDim.x * elemsPerBlock) {
    for (CeedInt i = 0; i < BASIS_T_1D; i++) {
      r_U[i] = 0.0;
      r_V[i] = 0.0;
      r_t[i] = 0.0;
    }
    if (transpose) {
      dim = 0;
      readQuads3d(elem, t_id_x, t_id_y, comp, dim, num_elem, d_U, r_U);
      ContractTransposeZ3d(slice, t_id_x, t_id_y, t_id_z, r_U, s_B, r_t);
      ContractTransposeY3d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_U);
      ContractTransposeX3d(slice, t_id_x, t_id_y, t_id_z, r_U, s_G, r_V);
      dim = 1;
      readQuads3d(elem, t_id_x, t_id_y, comp, dim, num_elem, d_U, r_U);
      ContractTransposeZ3d(slice, t_id_x, t_id_y, t_id_z, r_U, s_B, r_t);
      ContractTransposeY3d(slice, t_id_x, t_id_y, t_id_z, r_t, s_G, r_U);
      ContractTransposeX3d(slice, t_id_x, t_id_y, t_id_z, r_U, s_B, r_t);
      add(r_V, r_t);
      dim = 2;
      readQuads3d(elem, t_id_x, t_id_y, comp, dim, num_elem, d_U, r_U);
      ContractTransposeZ3d(slice, t_id_x, t_id_y, t_id_z, r_U, s_G, r_t);
      ContractTransposeY3d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_U);
      ContractTransposeX3d(slice, t_id_x, t_id_y, t_id_z, r_U, s_B, r_t);
      add(r_V, r_t);
      writeDofs3d(elem, t_id_x, t_id_y, comp, num_elem, r_V, d_V);
    } else {
      readDofs3d(elem, t_id_x, t_id_y, comp, num_elem, d_U, r_U);
      ContractX3d(slice, t_id_x, t_id_y, t_id_z, r_U, s_G, r_V);
      ContractY3d(slice, t_id_x, t_id_y, t_id_z, r_V, s_B, r_t);
      ContractZ3d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_V);
      dim = 0;
      writeQuads3d(elem, t_id_x, t_id_y, comp, dim, num_elem, r_V, d_V);
      ContractX3d(slice, t_id_x, t_id_y, t_id_z, r_U, s_B, r_V);
      ContractY3d(slice, t_id_x, t_id_y, t_id_z, r_V, s_G, r_t);
      ContractZ3d(slice, t_id_x, t_id_y, t_id_z, r_t, s_B, r_V);
      dim = 1;
      writeQuads3d(elem, t_id_x, t_id_y, comp, dim, num_elem, r_V, d_V);
      ContractX3d(slice, t_id_x, t_id_y, t_id_z, r_U, s_B, r_V);
      ContractY3d(slice, t_id_x, t_id_y, t_id_z, r_V, s_B, r_t);
      ContractZ3d(slice, t_id_x, t_id_y, t_id_z, r_t, s_G, r_V);
      dim = 2;
      writeQuads3d(elem, t_id_x, t_id_y, comp, dim, num_elem, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
__device__ void weight3d(const CeedInt num_elem, const CeedScalar *q_weight_1d, CeedScalar *w) {
  const CeedInt    i      = threadIdx.x;
  const CeedInt    j      = threadIdx.y;
  const CeedInt    k      = threadIdx.z;
  const CeedScalar weight = q_weight_1d[i] * q_weight_1d[j] * q_weight_1d[k];
  for (CeedInt e = blockIdx.x; e < num_elem; e += gridDim.x) {
    const CeedInt ind = e * BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D + i + j * BASIS_Q_1D + k * BASIS_Q_1D * BASIS_Q_1D;
    w[ind]            = weight;
  }
}

//------------------------------------------------------------------------------
// Basis kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Interp kernel by dim
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(BASIS_INTERP_BLOCK_SIZE) __global__ void Interp(const CeedInt num_elem, const CeedInt transpose, CeedScalar *d_interp_1d,
                                                                             const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  HIP_DYNAMIC_SHARED(CeedScalar, slice)
  // load interp_1d into shared memory
  __shared__ CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  loadMatrix(d_interp_1d, s_B);
  __syncthreads();

  if (BASIS_DIM == 1) {
    interp1d(num_elem, transpose, s_B, d_U, d_V, slice);
  } else if (BASIS_DIM == 2) {
    interp2d(num_elem, transpose, s_B, d_U, d_V, slice);
  } else if (BASIS_DIM == 3) {
    interp3d(num_elem, transpose, s_B, d_U, d_V, slice);
  }
}

//------------------------------------------------------------------------------
// Grad kernel by dim
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(BASIS_GRAD_BLOCK_SIZE) __global__
    void Grad(const CeedInt num_elem, const CeedInt transpose, CeedScalar *d_interp_1d, CeedScalar *d_grad_1d, const CeedScalar *__restrict__ d_U,
              CeedScalar *__restrict__ d_V) {
  HIP_DYNAMIC_SHARED(CeedScalar, slice)
  // load interp_1d and grad_1d into shared memory
  __shared__ CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  loadMatrix(d_interp_1d, s_B);
  __shared__ CeedScalar s_G[BASIS_P_1D * BASIS_Q_1D];
  loadMatrix(d_grad_1d, s_G);
  __syncthreads();

  if (BASIS_DIM == 1) {
    grad1d(num_elem, transpose, s_B, s_G, d_U, d_V, slice);
  } else if (BASIS_DIM == 2) {
    grad2d(num_elem, transpose, s_B, s_G, d_U, d_V, slice);
  } else if (BASIS_DIM == 3) {
    grad3d(num_elem, transpose, s_B, s_G, d_U, d_V, slice);
  }
}

//------------------------------------------------------------------------------
// Weight kernels by dim
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(BASIS_WEIGHT_BLOCK_SIZE) __global__
    void Weight(const CeedInt num_elem, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *__restrict__ v) {
  if (BASIS_DIM == 1) {
    weight1d(num_elem, q_weight_1d, v);
  } else if (BASIS_DIM == 2) {
    weight2d(num_elem, q_weight_1d, v);
  } else if (BASIS_DIM == 3) {
    weight3d(num_elem, q_weight_1d, v);
  }
}

//------------------------------------------------------------------------------
