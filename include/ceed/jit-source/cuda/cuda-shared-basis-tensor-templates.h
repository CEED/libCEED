// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA shared memory tensor product basis templates
#include <ceed/types.h>

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 1D tensor contraction x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractX1d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  __syncthreads();
  data.slice[data.t_id_x] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.t_id_x * P_1D] * data.slice[i];  // Contract x direction
    }
  }
}

//------------------------------------------------------------------------------
// 1D transpose tensor contraction x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeX1d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  __syncthreads();
  data.slice[data.t_id_x] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_x + i * P_1D] * data.slice[i];  // Contract x direction
    }
  }
}

//------------------------------------------------------------------------------
// 1D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void Interp1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX1d<NUM_COMP, P_1D, Q_1D>(data, &r_U[comp], c_B, &r_V[comp]);
  }
}

//------------------------------------------------------------------------------
// 1D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void InterpTranspose1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                         CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeX1d<NUM_COMP, P_1D, Q_1D>(data, &r_U[comp], c_B, &r_V[comp]);
  }
}

//------------------------------------------------------------------------------
// 1D interpolate to quadrature points, nodes and quadrature points collocated
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void InterpCollocatedNodes1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                               CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    r_V[comp] = r_U[comp];
  }
}

//------------------------------------------------------------------------------
// 1D interpolate transpose, nodes and quadrature points collocated
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void InterpTransposeCollocatedNodes1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                                        CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    r_V[comp] = r_U[comp];
  }
}

//------------------------------------------------------------------------------
// 1D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void Grad1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
                              CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX1d<NUM_COMP, P_1D, Q_1D>(data, &r_U[comp], c_G, &r_V[comp]);
  }
}

//------------------------------------------------------------------------------
// 1D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void GradTranspose1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
                                       CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeX1d<NUM_COMP, P_1D, Q_1D>(data, &r_U[comp], c_G, &r_V[comp]);
  }
}

//------------------------------------------------------------------------------
// 1D quadrature weights
//------------------------------------------------------------------------------
template <int P_1D, int Q_1D>
inline __device__ void Weight1d(SharedData_Cuda &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
  *w = (data.t_id_x < Q_1D) ? q_weight_1d[data.t_id_x] : 0.0;
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 2D tensor contraction x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractX2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  __syncthreads();
  data.slice[data.t_id_x + data.t_id_y * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.t_id_x * P_1D] * data.slice[i + data.t_id_y * T_1D];  // Contract x direction
    }
  }
}

//------------------------------------------------------------------------------
// 2D tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractY2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  __syncthreads();
  data.slice[data.t_id_x + data.t_id_y * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.t_id_y * P_1D] * data.slice[data.t_id_x + i * T_1D];  // Contract y direction
    }
  }
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractTransposeY2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  __syncthreads();
  data.slice[data.t_id_x + data.t_id_y * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_y + i * P_1D] * data.slice[data.t_id_x + i * T_1D];  // Contract y direction
    }
  }
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractTransposeX2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  __syncthreads();
  data.slice[data.t_id_x + data.t_id_y * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_x + i * P_1D] * data.slice[i + data.t_id_y * T_1D];  // Contract x direction
    }
  }
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract and add x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractTransposeAddX2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  __syncthreads();
  data.slice[data.t_id_x + data.t_id_y * T_1D] = *U;
  __syncthreads();
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_x + i * P_1D] * data.slice[i + data.t_id_y * T_1D];  // Contract x direction
    }
  }
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void InterpTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                      CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp], c_B, r_t);
    ContractY2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t, c_B, &r_V[comp]);
  }
}

//------------------------------------------------------------------------------
// 2D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void InterpTransposeTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                               CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeY2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp], c_B, r_t);
    ContractTransposeX2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t, c_B, &r_V[comp]);
  }
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points, nodes and quadrature points collocated
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void InterpTensorCollocatedNodes2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                                     CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    r_V[comp] = r_U[comp];
  }
}

//------------------------------------------------------------------------------
// 2D interpolate transpose, nodes and quadrature points collocated
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void InterpTransposeTensorCollocatedNodes2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                                              CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    r_V[comp] = r_U[comp];
  }
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void GradTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
                                    CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp], c_G, r_t);
    ContractY2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t, c_B, &r_V[comp + 0 * NUM_COMP]);
    ContractX2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp], c_B, r_t);
    ContractY2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t, c_G, &r_V[comp + 1 * NUM_COMP]);
  }
}

//------------------------------------------------------------------------------
// 2D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void GradTransposeTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
                                             CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeY2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp + 0 * NUM_COMP], c_B, r_t);
    ContractTransposeX2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t, c_G, &r_V[comp]);
    ContractTransposeY2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp + 1 * NUM_COMP], c_G, r_t);
    ContractTransposeAddX2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t, c_B, &r_V[comp]);
  }
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points, nodes and quadrature points collocated
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void GradTensorCollocatedNodes2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                                   const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp], c_G, &r_V[comp + 0 * NUM_COMP]);
    ContractY2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp], c_G, &r_V[comp + 1 * NUM_COMP]);
  }
}

//------------------------------------------------------------------------------
// 2D derivatives transpose, nodes and quadrature points collocated
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void GradTransposeTensorCollocatedNodes2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                                            const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeY2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp + 1 * NUM_COMP], c_G, &r_V[comp]);
    ContractTransposeAddX2d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp + 0 * NUM_COMP], c_G, &r_V[comp]);
  }
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
template <int P_1D, int Q_1D>
inline __device__ void WeightTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
  *w = (data.t_id_x < Q_1D && data.t_id_y < Q_1D) ? q_weight_1d[data.t_id_x] * q_weight_1d[data.t_id_y] : 0.0;
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 3D tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractX3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[P_1D];
  for (CeedInt i = 0; i < P_1D; i++) {
    r_B[i] = B[i + data.t_id_x * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    __syncthreads();
    data.slice[data.t_id_x + data.t_id_y * T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += r_B[i] * data.slice[i + data.t_id_y * T_1D];  // Contract x direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractY3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[P_1D];
  for (CeedInt i = 0; i < P_1D; i++) {
    r_B[i] = B[i + data.t_id_y * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    __syncthreads();
    data.slice[data.t_id_x + data.t_id_y * T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += r_B[i] * data.slice[data.t_id_x + i * T_1D];  // Contract y direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract z
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractZ3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < Q_1D; k++) {
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += B[i + k * P_1D] * U[i];  // Contract z direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract z
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractTransposeZ3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < P_1D; k++) {
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += B[k + i * P_1D] * U[i];  // Contract z direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractTransposeY3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.t_id_y + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    __syncthreads();
    data.slice[data.t_id_x + data.t_id_y * T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.slice[data.t_id_x + i * T_1D];  // Contract y direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractTransposeAddY3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.t_id_y + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    __syncthreads();
    data.slice[data.t_id_x + data.t_id_y * T_1D] = U[k];
    __syncthreads();
    if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.slice[data.t_id_x + i * T_1D];  // Contract y direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractTransposeX3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.t_id_x + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    __syncthreads();
    data.slice[data.t_id_x + data.t_id_y * T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.slice[i + data.t_id_y * T_1D];  // Contract x direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract add x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ContractTransposeAddX3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.t_id_x + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    __syncthreads();
    data.slice[data.t_id_x + data.t_id_y * T_1D] = U[k];
    __syncthreads();
    if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.slice[i + data.t_id_y * T_1D];  // Contract x direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D Cheetah execution pipelines
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, int STRIDE_1D>
inline __device__ void InterpTensorPipeline3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *matX,
                                              const CeedScalar *matY, const CeedScalar *matZ, CeedScalar *__restrict__ r_V) {
  const CeedInt S        = STRIDE_1D;
  const CeedInt tx       = data.t_id_x;
  const CeedInt ty       = data.t_id_y;
  CeedScalar   *s_buf[2] = {data.slice, data.slice + S * S * S};

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    // Stage 1: Load U into shared memory
    if (tx < P_1D && ty < P_1D) {
#pragma unroll
      for (CeedInt z = 0; z < P_1D; z++) {
        s_buf[0][tx + ty * S + z * S * S] = r_U[z + comp * P_1D];
      }
    }
    __syncthreads();

    // Stage 2: X-contraction with matX
    if (tx < P_1D && ty < P_1D) {
      CeedScalar r_t1[Q_1D], r_u[P_1D];
#pragma unroll
      for (CeedInt x = 0; x < P_1D; x++) {
        r_u[x] = s_buf[0][x + tx * S + ty * S * S];
      }
#pragma unroll
      for (CeedInt qx = 0; qx < Q_1D; qx++) {
        r_t1[qx] = 0.0;
#pragma unroll
        for (CeedInt x = 0; x < P_1D; x++) {
          r_t1[qx] += matX[x + qx * P_1D] * r_u[x];
        }
        s_buf[1][qx + tx * S + ty * S * S] = r_t1[qx];
      }
    }
    __syncthreads();

    // Stage 3: Y-contraction with matY
    if (tx < Q_1D && ty < P_1D) {
      CeedScalar r_t2[Q_1D], r_t1_row[P_1D];
#pragma unroll
      for (CeedInt y = 0; y < P_1D; y++) {
        r_t1_row[y] = s_buf[1][tx + y * S + ty * S * S];
      }
#pragma unroll
      for (CeedInt qy = 0; qy < Q_1D; qy++) {
        r_t2[qy] = 0.0;
#pragma unroll
        for (CeedInt y = 0; y < P_1D; y++) {
          r_t2[qy] += matY[y + qy * P_1D] * r_t1_row[y];
        }
        s_buf[0][tx + qy * S + ty * S * S] = r_t2[qy];
      }
    }
    __syncthreads();

    // Stage 4: Z-contraction with matZ
    if (tx < Q_1D && ty < Q_1D) {
      CeedScalar r_t2_col[P_1D];
#pragma unroll
      for (CeedInt z = 0; z < P_1D; z++) {
        r_t2_col[z] = s_buf[0][tx + ty * S + z * S * S];
      }
#pragma unroll
      for (CeedInt qz = 0; qz < Q_1D; qz++) {
        CeedScalar val = 0.0;
#pragma unroll
        for (CeedInt z = 0; z < P_1D; z++) {
          val += matZ[z + qz * P_1D] * r_t2_col[z];
        }
        r_V[qz + comp * Q_1D] = val;
      }
    }
    __syncthreads();
  }
}

template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, int STRIDE_1D>
inline __device__ void InterpTransposeTensorPipeline3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *matX,
                                                       const CeedScalar *matY, const CeedScalar *matZ, CeedScalar *__restrict__ r_V,
                                                       bool add_to_output) {
  const CeedInt S        = STRIDE_1D;
  const CeedInt tx       = data.t_id_x;
  const CeedInt ty       = data.t_id_y;
  CeedScalar   *s_buf[2] = {data.slice, data.slice + S * S * S};

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    // Stage 1: Z-contraction transpose with matZ
    CeedScalar r_t1[P_1D];
    if (tx < Q_1D && ty < Q_1D) {
#pragma unroll
      for (CeedInt z = 0; z < P_1D; z++) {
        r_t1[z] = 0.0;
#pragma unroll
        for (CeedInt qz = 0; qz < Q_1D; qz++) {
          r_t1[z] += matZ[z + qz * P_1D] * r_U[qz + comp * Q_1D];
        }
        s_buf[0][tx + ty * S + z * S * S] = r_t1[z];
      }
    }
    __syncthreads();

    // Stage 2: Y-contraction transpose with matY
    CeedScalar r_t2[P_1D];
    if (tx < Q_1D && ty < P_1D) {
      CeedScalar r_t1_col[Q_1D];
#pragma unroll
      for (CeedInt qy = 0; qy < Q_1D; qy++) {
        r_t1_col[qy] = s_buf[0][tx + qy * S + ty * S * S];
      }
#pragma unroll
      for (CeedInt y = 0; y < P_1D; y++) {
        r_t2[y] = 0.0;
#pragma unroll
        for (CeedInt qy = 0; qy < Q_1D; qy++) {
          r_t2[y] += matY[y + qy * P_1D] * r_t1_col[qy];
        }
        s_buf[1][tx + y * S + ty * S * S] = r_t2[y];
      }
    }
    __syncthreads();

    // Stage 3: X-contraction transpose with matX
    CeedScalar r_v[P_1D];
    if (tx < P_1D && ty < P_1D) {
      CeedScalar r_t2_row[Q_1D];
#pragma unroll
      for (CeedInt qx = 0; qx < Q_1D; qx++) {
        r_t2_row[qx] = s_buf[1][qx + tx * S + ty * S * S];
      }
#pragma unroll
      for (CeedInt x = 0; x < P_1D; x++) {
        r_v[x] = 0.0;
#pragma unroll
        for (CeedInt qx = 0; qx < Q_1D; qx++) {
          r_v[x] += matX[x + qx * P_1D] * r_t2_row[qx];
        }
        s_buf[0][x + tx * S + ty * S * S] = r_v[x];
      }
    }
    __syncthreads();

    // Stage 4: Read column along z into r_V
    if (tx < P_1D && ty < P_1D) {
#pragma unroll
      for (CeedInt z = 0; z < P_1D; z++) {
        if (add_to_output) {
          r_V[z + comp * P_1D] += s_buf[0][tx + ty * S + z * S * S];
        } else {
          r_V[z + comp * P_1D] = s_buf[0][tx + ty * S + z * S * S];
        }
      }
    }
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, int STRIDE_1D = 0>
inline __device__ void InterpTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                      CeedScalar *__restrict__ r_V) {
  if (STRIDE_1D == 0) {
    CeedScalar r_t1[T_1D];
    CeedScalar r_t2[T_1D];
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      ContractX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * P_1D], c_B, r_t1);
      ContractY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t1, c_B, r_t2);
      ContractZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t2, c_B, &r_V[comp * Q_1D]);
    }
    return;
  }
  InterpTensorPipeline3d<NUM_COMP, P_1D, Q_1D, T_1D, STRIDE_1D>(data, r_U, c_B, c_B, c_B, r_V);
}

//------------------------------------------------------------------------------
// 3D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, int STRIDE_1D = 0>
inline __device__ void InterpTransposeTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                               CeedScalar *__restrict__ r_V) {
  if (STRIDE_1D == 0) {
    CeedScalar r_t1[T_1D];
    CeedScalar r_t2[T_1D];
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * Q_1D], c_B, r_t1);
      ContractTransposeY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t1, c_B, r_t2);
      ContractTransposeX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t2, c_B, &r_V[comp * P_1D]);
    }
    return;
  }
  InterpTransposeTensorPipeline3d<NUM_COMP, P_1D, Q_1D, T_1D, STRIDE_1D>(data, r_U, c_B, c_B, c_B, r_V, false);
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points, nodes and quadrature points collocated
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void InterpTensorCollocatedNodes3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                                     CeedScalar *__restrict__ r_V) {
  for (CeedInt i = 0; i < Q_1D; i++) {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_V[i + comp * Q_1D] = r_U[i + comp * P_1D];
    }
  }
}

//------------------------------------------------------------------------------
// 3D interpolate transpose, nodes and quadrature points collocated
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void InterpTransposeTensorCollocatedNodes3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                                              CeedScalar *__restrict__ r_V) {
  for (CeedInt i = 0; i < Q_1D; i++) {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_V[i + comp * P_1D] = r_U[i + comp * Q_1D];
    }
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, int STRIDE_1D = 0>
inline __device__ void GradTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
                                    CeedScalar *__restrict__ r_V) {
  if (STRIDE_1D == 0) {
    CeedScalar r_t1[T_1D];
    CeedScalar r_t2[T_1D];
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      ContractX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * P_1D], c_G, r_t1);
      ContractY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t1, c_B, r_t2);
      ContractZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t2, c_B, &r_V[comp * Q_1D + 0 * NUM_COMP * Q_1D]);
      ContractX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * P_1D], c_B, r_t1);
      ContractY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t1, c_G, r_t2);
      ContractZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t2, c_B, &r_V[comp * Q_1D + 1 * NUM_COMP * Q_1D]);
      ContractX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * P_1D], c_B, r_t1);
      ContractY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t1, c_B, r_t2);
      ContractZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t2, c_G, &r_V[comp * Q_1D + 2 * NUM_COMP * Q_1D]);
    }
    return;
  }
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    InterpTensorPipeline3d<1, P_1D, Q_1D, T_1D, STRIDE_1D>(data, &r_U[comp * P_1D], c_G, c_B, c_B, &r_V[comp * Q_1D + 0 * NUM_COMP * Q_1D]);
    InterpTensorPipeline3d<1, P_1D, Q_1D, T_1D, STRIDE_1D>(data, &r_U[comp * P_1D], c_B, c_G, c_B, &r_V[comp * Q_1D + 1 * NUM_COMP * Q_1D]);
    InterpTensorPipeline3d<1, P_1D, Q_1D, T_1D, STRIDE_1D>(data, &r_U[comp * P_1D], c_B, c_B, c_G, &r_V[comp * Q_1D + 2 * NUM_COMP * Q_1D]);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, int STRIDE_1D = 0>
inline __device__ void GradTransposeTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
                                             CeedScalar *__restrict__ r_V) {
  if (STRIDE_1D == 0) {
    CeedScalar r_t1[T_1D];
    CeedScalar r_t2[T_1D];
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * Q_1D + 0 * NUM_COMP * Q_1D], c_B, r_t1);
      ContractTransposeY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t1, c_B, r_t2);
      ContractTransposeX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t2, c_G, &r_V[comp * P_1D]);
      ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * Q_1D + 1 * NUM_COMP * Q_1D], c_B, r_t1);
      ContractTransposeY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t1, c_G, r_t2);
      ContractTransposeAddX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t2, c_B, &r_V[comp * P_1D]);
      ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * Q_1D + 2 * NUM_COMP * Q_1D], c_G, r_t1);
      ContractTransposeY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t1, c_B, r_t2);
      ContractTransposeAddX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t2, c_B, &r_V[comp * P_1D]);
    }
    return;
  }
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    InterpTransposeTensorPipeline3d<1, P_1D, Q_1D, T_1D, STRIDE_1D>(data, &r_U[comp * Q_1D + 0 * NUM_COMP * Q_1D], c_G, c_B, c_B, &r_V[comp * P_1D],
                                                                    false);
    InterpTransposeTensorPipeline3d<1, P_1D, Q_1D, T_1D, STRIDE_1D>(data, &r_U[comp * Q_1D + 1 * NUM_COMP * Q_1D], c_B, c_G, c_B, &r_V[comp * P_1D],
                                                                    true);
    InterpTransposeTensorPipeline3d<1, P_1D, Q_1D, T_1D, STRIDE_1D>(data, &r_U[comp * Q_1D + 2 * NUM_COMP * Q_1D], c_B, c_B, c_G, &r_V[comp * P_1D],
                                                                    true);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, int STRIDE_1D = 0>
inline __device__ void GradTensorCollocated3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
                                              CeedScalar *__restrict__ r_V) {
  if (STRIDE_1D == 0) {
    CeedScalar r_t1[T_1D];
    CeedScalar r_t2[T_1D];
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      ContractX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * P_1D], c_B, r_t1);
      ContractY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t1, c_B, r_t2);
      ContractZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t2, c_B, r_t1);
      ContractX3d<NUM_COMP, Q_1D, Q_1D, T_1D>(data, r_t1, c_G, &r_V[comp * Q_1D + 0 * NUM_COMP * Q_1D]);
      ContractY3d<NUM_COMP, Q_1D, Q_1D, T_1D>(data, r_t1, c_G, &r_V[comp * Q_1D + 1 * NUM_COMP * Q_1D]);
      ContractZ3d<NUM_COMP, Q_1D, Q_1D, T_1D>(data, r_t1, c_G, &r_V[comp * Q_1D + 2 * NUM_COMP * Q_1D]);
    }
    return;
  }

  const CeedInt S        = STRIDE_1D;
  const CeedInt tx       = data.t_id_x;
  const CeedInt ty       = data.t_id_y;
  CeedScalar   *s_buf[2] = {data.slice, data.slice + S * S * S};

  // Preload comp 0
  if (tx < P_1D && ty < P_1D) {
#pragma unroll
    for (CeedInt z = 0; z < P_1D; z++) {
      s_buf[0][tx + ty * S + z * S * S] = r_U[z + 0 * P_1D];
    }
  }
  __syncthreads();

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    // Forward interpolation of comp to quadrature points
    // X-contraction
    CeedScalar r_t1[Q_1D];
    if (tx < P_1D && ty < P_1D) {
      CeedScalar r_u[P_1D];
#pragma unroll
      for (CeedInt x = 0; x < P_1D; x++) {
        r_u[x] = s_buf[0][x + tx * S + ty * S * S];
      }
      for (CeedInt qx = 0; qx < Q_1D; qx++) {
        r_t1[qx] = 0.0;
#pragma unroll
        for (CeedInt x = 0; x < P_1D; x++) {
          r_t1[qx] += c_B[x + qx * P_1D] * r_u[x];
        }
      }
    }
    if (tx < P_1D && ty < P_1D) {
#pragma unroll
      for (CeedInt qx = 0; qx < Q_1D; qx++) {
        s_buf[1][qx + tx * S + ty * S * S] = r_t1[qx];
      }
    }
    // Sync ends read on 0, write on 1
    __syncthreads();

    // Y-contraction
    CeedScalar r_t2[Q_1D];
    if (tx < Q_1D && ty < P_1D) {
      CeedScalar r_t1_row[P_1D];
#pragma unroll
      for (CeedInt y = 0; y < P_1D; y++) {
        r_t1_row[y] = s_buf[1][tx + y * S + ty * S * S];
      }
#pragma unroll
      for (CeedInt qy = 0; qy < Q_1D; qy++) {
        r_t2[qy] = 0.0;
#pragma unroll
        for (CeedInt y = 0; y < P_1D; y++) {
          r_t2[qy] += c_B[y + qy * P_1D] * r_t1_row[y];
        }
      }
    }
    if (tx < Q_1D && ty < P_1D) {
#pragma unroll
      for (CeedInt qy = 0; qy < Q_1D; qy++) {
        s_buf[0][tx + qy * S + ty * S * S] = r_t2[qy];
      }
    }
    // Sync ends read on 1, write on 0
    __syncthreads();

    // Z-contraction: store interpolated values U_q into shared memory and keep in registers
    CeedScalar r_uq[Q_1D];
    if (tx < Q_1D && ty < Q_1D) {
      CeedScalar r_t2_col[P_1D];
#pragma unroll
      for (CeedInt z = 0; z < P_1D; z++) {
        r_t2_col[z] = s_buf[0][tx + ty * S + z * S * S];
      }
#pragma unroll
      for (CeedInt qz = 0; qz < Q_1D; qz++) {
        r_uq[qz] = 0.0;
#pragma unroll
        for (CeedInt z = 0; z < P_1D; z++) {
          r_uq[qz] += c_B[z + qz * P_1D] * r_t2_col[z];
        }
      }
    }
    if (tx < Q_1D && ty < Q_1D) {
#pragma unroll
      for (CeedInt qz = 0; qz < Q_1D; qz++) {
        s_buf[1][tx + ty * S + qz * S * S] = r_uq[qz];
      }
    }
    // Sync ends read on 0, write on 1
    __syncthreads();

    // 2. Collocated derivatives from U_q:
    if (tx < Q_1D && ty < Q_1D) {
      for (CeedInt qz = 0; qz < Q_1D; qz++) {
        CeedScalar vz = 0.0;
#pragma unroll
        for (CeedInt z = 0; z < Q_1D; z++) {
          vz += c_G[z + qz * Q_1D] * r_uq[z];
        }
        r_V[qz + comp * Q_1D + 2 * NUM_COMP * Q_1D] = vz;

        CeedScalar vx = 0.0;
        CeedScalar vy = 0.0;
#pragma unroll
        for (CeedInt i = 0; i < Q_1D; i++) {
          vx += c_G[i + tx * Q_1D] * s_buf[1][i + ty * S + qz * S * S];
          vy += c_G[i + ty * Q_1D] * s_buf[1][tx + i * S + qz * S * S];
        }
        r_V[qz + comp * Q_1D + 0 * NUM_COMP * Q_1D] = vx;
        r_V[qz + comp * Q_1D + 1 * NUM_COMP * Q_1D] = vy;
      }
    }
    if (comp + 1 < NUM_COMP && tx < P_1D && ty < P_1D) {
#pragma unroll
      for (CeedInt z = 0; z < P_1D; z++) {
        s_buf[0][tx + ty * S + z * S * S] = r_U[z + (comp + 1) * P_1D];
      }
    }
    // Sync ends read on 1, write on 0
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, int STRIDE_1D = 0>
inline __device__ void GradTransposeTensorCollocated3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                                       const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  if (STRIDE_1D == 0) {
    CeedScalar r_t1[T_1D];
    CeedScalar r_t2[T_1D];
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      ContractTransposeZ3d<NUM_COMP, Q_1D, Q_1D, T_1D>(data, &r_U[comp * Q_1D + 2 * NUM_COMP * Q_1D], c_G, r_t2);
      ContractTransposeAddY3d<NUM_COMP, Q_1D, Q_1D, T_1D>(data, &r_U[comp * Q_1D + 1 * NUM_COMP * Q_1D], c_G, r_t2);
      ContractTransposeAddX3d<NUM_COMP, Q_1D, Q_1D, T_1D>(data, &r_U[comp * Q_1D + 0 * NUM_COMP * Q_1D], c_G, r_t2);
      ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t2, c_B, r_t1);
      ContractTransposeY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t1, c_B, r_t2);
      ContractTransposeX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, r_t2, c_B, &r_V[comp * P_1D]);
    }
    return;
  }

  const CeedInt S        = STRIDE_1D;
  const CeedInt tx       = data.t_id_x;
  const CeedInt ty       = data.t_id_y;
  CeedScalar   *s_buf[2] = {data.slice, data.slice + S * S * S};

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    // Term 2 (Z derivative transpose) computed directly from registers
    CeedScalar r_w[Q_1D];
    if (tx < Q_1D && ty < Q_1D) {
#pragma unroll
      for (CeedInt qz = 0; qz < Q_1D; qz++) {
        r_w[qz] = 0.0;
#pragma unroll
        for (CeedInt k = 0; k < Q_1D; k++) {
          r_w[qz] += c_G[qz + k * Q_1D] * r_U[k + comp * Q_1D + 2 * NUM_COMP * Q_1D];
        }
      }
// Store U1 to shared memory
#pragma unroll
      for (CeedInt qz = 0; qz < Q_1D; qz++) {
        s_buf[0][tx + ty * S + qz * S * S] = r_U[qz + comp * Q_1D + 1 * NUM_COMP * Q_1D];
      }
// Store U0 to shared memory
#pragma unroll
      for (CeedInt qz = 0; qz < Q_1D; qz++) {
        s_buf[1][tx + ty * S + qz * S * S] = r_U[qz + comp * Q_1D + 0 * NUM_COMP * Q_1D];
      }
    }
    // Sync ends write on 0, 1
    __syncthreads();

    // Term 1 (Y derivative transpose)
    if (tx < Q_1D && ty < Q_1D) {
#pragma unroll
      for (CeedInt qz = 0; qz < Q_1D; qz++) {
#pragma unroll
        for (CeedInt i = 0; i < Q_1D; i++) {
          r_w[qz] += c_G[ty + i * Q_1D] * s_buf[0][tx + i * S + qz * S * S];
        }
      }
    }

    // Term 0 (X derivative transpose)
    if (tx < Q_1D && ty < Q_1D) {
#pragma unroll
      for (CeedInt qz = 0; qz < Q_1D; qz++) {
#pragma unroll
        for (CeedInt i = 0; i < Q_1D; i++) {
          r_w[qz] += c_G[tx + i * Q_1D] * s_buf[1][i + ty * S + qz * S * S];
        }
      }
    }
    // Sync ends read on 0, 1
    __syncthreads();

    // Interpolate transpose of r_w back to nodal points
    InterpTransposeTensorPipeline3d<1, P_1D, Q_1D, T_1D, STRIDE_1D>(data, r_w, c_B, c_B, c_B, &r_V[comp * P_1D], false);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points, nodes and quadrature points collocated
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, int STRIDE_1D = 0>
inline __device__ void GradTensorCollocatedNodes3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                                   const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  if (STRIDE_1D == 0) {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      ContractX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * P_1D], c_G, &r_V[comp * Q_1D + 0 * NUM_COMP * Q_1D]);
      ContractY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * P_1D], c_G, &r_V[comp * Q_1D + 1 * NUM_COMP * Q_1D]);
      ContractZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * P_1D], c_G, &r_V[comp * Q_1D + 2 * NUM_COMP * Q_1D]);
    }
    return;
  }

  const CeedInt S        = STRIDE_1D;
  const CeedInt tx       = data.t_id_x;
  const CeedInt ty       = data.t_id_y;
  CeedScalar   *s_buf[2] = {data.slice, data.slice + S * S * S};

  // Preload comp 0
  for (CeedInt z = 0; z < P_1D; z++) {
    s_buf[0][tx + ty * S + z * S * S] = r_U[z + 0 * P_1D];
  }
  // Sync ends write on 0
  __syncthreads();
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    if (tx < P_1D && ty < P_1D) {
      for (CeedInt qz = 0; qz < P_1D; qz++) {
        CeedScalar vz = 0.0;
        for (CeedInt i = 0; i < P_1D; i++) {
          vz += c_G[i + qz * P_1D] * r_U[i + comp * P_1D];
        }
        r_V[qz + comp * P_1D + 2 * NUM_COMP * P_1D] = vz;
      }
    }

    if (tx < P_1D && ty < P_1D) {
      for (CeedInt qz = 0; qz < P_1D; qz++) {
        CeedScalar vx = 0.0;
        CeedScalar vy = 0.0;
        for (CeedInt i = 0; i < P_1D; i++) {
          vx += c_G[i + tx * P_1D] * s_buf[0][i + ty * S + qz * S * S];
          vy += c_G[i + ty * P_1D] * s_buf[0][tx + i * S + qz * S * S];
        }
        r_V[qz + comp * P_1D + 0 * NUM_COMP * P_1D] = vx;
        r_V[qz + comp * P_1D + 1 * NUM_COMP * P_1D] = vy;
      }
    }
    if (comp + 1 < NUM_COMP && tx < P_1D && ty < P_1D) {
      for (CeedInt z = 0; z < P_1D; z++) {
        s_buf[(comp + 1) % 2][tx + ty * S + z * S * S] = r_U[z + (comp + 1) * P_1D];
      }
    }
    // Sync ends read on comp % 2, write on comp + 1 % 2
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D derivatives transpose, nodes and quadrature points collocated
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, int STRIDE_1D = 0>
inline __device__ void GradTransposeTensorCollocatedNodes3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
                                                            const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  if (STRIDE_1D == 0) {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * Q_1D + 2 * NUM_COMP * Q_1D], c_G, &r_V[comp * P_1D]);
      ContractTransposeAddY3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * Q_1D + 1 * NUM_COMP * Q_1D], c_G, &r_V[comp * P_1D]);
      ContractTransposeAddX3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, &r_U[comp * Q_1D + 0 * NUM_COMP * Q_1D], c_G, &r_V[comp * P_1D]);
    }
    return;
  }

  const CeedInt S  = STRIDE_1D;
  const CeedInt tx = data.t_id_x;
  const CeedInt ty = data.t_id_y;

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    CeedScalar r_v[P_1D];
    if (tx < P_1D && ty < P_1D) {
      for (CeedInt z = 0; z < P_1D; z++) {
        r_v[z] = 0.0;
        for (CeedInt k = 0; k < P_1D; k++) {
          r_v[z] += c_G[z + k * P_1D] * r_U[k + comp * P_1D + 2 * NUM_COMP * P_1D];
        }
      }
      for (CeedInt z = 0; z < P_1D; z++) {
        data.slice[tx + ty * S + z * S * S] = r_U[z + comp * P_1D + 1 * NUM_COMP * P_1D];
      }
    }
    __syncthreads();

    if (tx < P_1D && ty < P_1D) {
      for (CeedInt z = 0; z < P_1D; z++) {
        for (CeedInt i = 0; i < P_1D; i++) {
          r_v[z] += c_G[ty + i * P_1D] * data.slice[tx + i * S + z * S * S];
        }
      }
      for (CeedInt z = 0; z < P_1D; z++) {
        data.slice[tx + ty * S + z * S * S] = r_U[z + comp * P_1D + 0 * NUM_COMP * P_1D];
      }
    }
    __syncthreads();

    if (tx < P_1D && ty < P_1D) {
      for (CeedInt z = 0; z < P_1D; z++) {
        for (CeedInt i = 0; i < P_1D; i++) {
          r_v[z] += c_G[tx + i * P_1D] * data.slice[i + ty * S + z * S * S];
        }
        r_V[z + comp * P_1D] = r_v[z];
      }
    }
    if (comp + 1 < NUM_COMP) {
      __syncthreads();
    }
  }
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
template <int P_1D, int Q_1D>
inline __device__ void WeightTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
  const bool       quad = (data.t_id_x < Q_1D && data.t_id_y < Q_1D);
  const CeedScalar pw   = quad ? q_weight_1d[data.t_id_x] * q_weight_1d[data.t_id_y] : 0.0;
  for (CeedInt q = 0; q < Q_1D; q++) {
    w[q] = quad ? pw * q_weight_1d[q] : 0.0;
  }
}
