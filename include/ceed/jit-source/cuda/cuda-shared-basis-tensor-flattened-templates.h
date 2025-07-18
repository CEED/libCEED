// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA shared memory tensor product basis templates
#include <ceed/types.h>

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 2D tensor contraction x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractX2dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const ScalarIn1 *U, const ScalarIn2 *B,
                                            ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (t_id_x < Q_1D && t_id_y < P_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + t_id_x * P_1D] * data.slice[i + t_id_y * T_1D];  // Contract x direction
    }
  }
}

//------------------------------------------------------------------------------
// 2D tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractY2dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const ScalarIn1 *U, const ScalarIn2 *B,
                                            ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (t_id_x < Q_1D && t_id_y < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + t_id_y * P_1D] * data.slice[t_id_x + i * T_1D];  // Contract y direction
    }
  }
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractTransposeY2dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const ScalarIn1 *U,
                                                     const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (t_id_x < Q_1D && t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[t_id_y + i * P_1D] * data.slice[t_id_x + i * T_1D];  // Contract y direction
    }
  }
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractTransposeX2dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const ScalarIn1 *U,
                                                     const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (t_id_x < P_1D && t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[t_id_x + i * P_1D] * data.slice[i + t_id_y * T_1D];  // Contract x direction
    }
  }
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract and add x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractTransposeAddX2dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const ScalarIn1 *U,
                                                        const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D] = *U;
  __syncthreads();
  if (t_id_x < P_1D && t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[t_id_x + i * P_1D] * data.slice[i + t_id_y * T_1D];  // Contract x direction
    }
  }
}

//------------------------------------------------------------------------------
// 2D pack/unpack quadrature values
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_1D, int T_1D, class ScalarOut>
inline __device__ void QPack2d(SharedData_Cuda &data, const int t_id_x, const int t_id_y, ScalarOut *U) {
  const CeedInt new_t_id_x = data.t_id_x % Q_1D, new_t_id_y = data.t_id_x / Q_1D;

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    __syncthreads();
    if (t_id_x < Q_1D && t_id_y < Q_1D) data.slice[t_id_x + t_id_y * T_1D] = U[comp];
    __syncthreads();
    U[comp] = data.t_id_x < (Q_1D * Q_1D) ? data.slice[new_t_id_x + new_t_id_y * T_1D] : 0.0;
  }
}

template <int NUM_COMP, int Q_1D, int T_1D, class ScalarOut>
inline __device__ void QUnpack2d(SharedData_Cuda &data, const int t_id_x, const int t_id_y, ScalarOut *U) {
  const CeedInt old_t_id_x = data.t_id_x % Q_1D, old_t_id_y = data.t_id_x / Q_1D;

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    __syncthreads();
    if (data.t_id_x < (Q_1D * Q_1D)) data.slice[old_t_id_x + old_t_id_y * T_1D] = U[comp];
    __syncthreads();
    U[comp] = (t_id_x < Q_1D && t_id_y < Q_1D) ? data.slice[t_id_x + t_id_y * T_1D] : 0.0;
  }
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void InterpTensor2dFlattened(SharedData_Cuda &data, ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B,
                                               ScalarOut *__restrict__ r_V) {
  const int  t_id_x = data.t_id_x % T_1D, t_id_y = data.t_id_x / T_1D;
  CeedScalar r_t[1];

  if (P_1D != T_1D) QUnpack2d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, r_U);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, &r_U[comp], c_B, r_t);
    ContractY2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, r_t, c_B, &r_V[comp]);
  }
  __syncthreads();
  if (P_1D != T_1D) QPack2d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, r_U);
  if (Q_1D != T_1D) QPack2d<NUM_COMP, Q_1D, T_1D>(data, t_id_x, t_id_y, r_V);
}

//------------------------------------------------------------------------------
// 2D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void InterpTransposeTensor2dFlattened(SharedData_Cuda &data, ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B,
                                                        ScalarOut *__restrict__ r_V) {
  const int  t_id_x = data.t_id_x % T_1D, t_id_y = data.t_id_x / T_1D;
  CeedScalar r_t[1];

  if (Q_1D != T_1D) QUnpack2d<NUM_COMP, Q_1D, T_1D>(data, t_id_x, t_id_y, r_U);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeY2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, &r_U[comp], c_B, r_t);
    ContractTransposeX2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, r_t, c_B, &r_V[comp]);
  }
  __syncthreads();
  if (P_1D != T_1D) QPack2d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, r_V);
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarIn3, class ScalarOut>
inline __device__ void GradTensor2dFlattened(SharedData_Cuda &data, ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B, const ScalarIn3 *c_G,
                                             ScalarOut *__restrict__ r_V) {
  const int  t_id_x = data.t_id_x % T_1D, t_id_y = data.t_id_x / T_1D;
  CeedScalar r_t[1];

  if (P_1D != T_1D) QUnpack2d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, r_U);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, &r_U[comp], c_G, r_t);
    ContractY2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, r_t, c_B, &r_V[comp + 0 * NUM_COMP]);
    ContractX2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, &r_U[comp], c_B, r_t);
    ContractY2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, r_t, c_G, &r_V[comp + 1 * NUM_COMP]);
  }
  __syncthreads();
  if (P_1D != T_1D) QPack2d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, r_U);
  if (Q_1D != T_1D) QPack2d<NUM_COMP * 2, Q_1D, T_1D>(data, t_id_x, t_id_y, r_V);
}

//------------------------------------------------------------------------------
// 2D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarIn3, class ScalarOut>
inline __device__ void GradTransposeTensor2dFlattened(SharedData_Cuda &data, ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B, const ScalarIn3 *c_G,
                                                      ScalarOut *__restrict__ r_V) {
  const int  t_id_x = data.t_id_x % T_1D, t_id_y = data.t_id_x / T_1D;
  CeedScalar r_t[1];

  if (Q_1D != T_1D) QUnpack2d<NUM_COMP * 2, Q_1D, T_1D>(data, t_id_x, t_id_y, r_U);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeY2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, &r_U[comp + 0 * NUM_COMP], c_B, r_t);
    ContractTransposeX2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, r_t, c_G, &r_V[comp]);
    ContractTransposeY2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, &r_U[comp + 1 * NUM_COMP], c_G, r_t);
    ContractTransposeAddX2dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, r_t, c_B, &r_V[comp]);
  }
  __syncthreads();
  if (P_1D != T_1D) QPack2d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, r_V);
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
template <int P_1D, int Q_1D, class ScalarIn, class ScalarOut>
inline __device__ void WeightTensor2dFlattened(SharedData_Cuda &data, const ScalarIn *__restrict__ q_weight_1d, ScalarOut *w) {
  const int t_id_x = data.t_id_x % Q_1D, t_id_y = data.t_id_x / Q_1D;

  *w = (t_id_x < Q_1D && t_id_y < Q_1D) ? q_weight_1d[t_id_x] * q_weight_1d[t_id_y] : 0.0;
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 3D tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractX3dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const int t_id_z, const ScalarIn1 *U,
                                            const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D + t_id_z * T_1D * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (t_id_x < Q_1D && t_id_y < P_1D && t_id_z < P_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + t_id_x * P_1D] * data.slice[i + t_id_y * T_1D + t_id_z * T_1D * T_1D];  // Contract x direction
    }
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractY3dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const int t_id_z, const ScalarIn1 *U,
                                            const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D + t_id_z * T_1D * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (t_id_x < Q_1D && t_id_y < Q_1D && t_id_z < P_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + t_id_y * P_1D] * data.slice[t_id_x + i * T_1D + t_id_z * T_1D * T_1D];  // Contract y direction
    }
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract z
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractZ3dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const int t_id_z, const ScalarIn1 *U,
                                            const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D + t_id_z * T_1D * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (t_id_x < Q_1D && t_id_y < Q_1D && t_id_z < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + t_id_z * P_1D] * data.slice[t_id_x + t_id_y * T_1D + i * T_1D * T_1D];  // Contract z direction
    }
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract z
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractTransposeZ3dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const int t_id_z, const ScalarIn1 *U,
                                                     const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D + t_id_z * T_1D * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (t_id_x < Q_1D && t_id_y < Q_1D && t_id_z < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[t_id_z + i * P_1D] * data.slice[t_id_x + t_id_y * T_1D + i * T_1D * T_1D];  // Contract z direction
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract z
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractTransposeAddZ3dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const int t_id_z,
                                                        const ScalarIn1 *U, const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D + t_id_z * T_1D * T_1D] = *U;
  __syncthreads();
  if (t_id_x < Q_1D && t_id_y < Q_1D && t_id_z < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[t_id_z + i * P_1D] * data.slice[t_id_x + t_id_y * T_1D + i * T_1D * T_1D];  // Contract z direction
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractTransposeY3dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const int t_id_z, const ScalarIn1 *U,
                                                     const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D + t_id_z * T_1D * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (t_id_x < Q_1D && t_id_y < P_1D && t_id_z < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[t_id_y + i * P_1D] * data.slice[t_id_x + i * T_1D + t_id_z * T_1D * T_1D];  // Contract y direction
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractTransposeAddY3dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const int t_id_z,
                                                        const ScalarIn1 *U, const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D + t_id_z * T_1D * T_1D] = *U;
  __syncthreads();
  if (t_id_x < Q_1D && t_id_y < P_1D && t_id_z < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[t_id_y + i * P_1D] * data.slice[t_id_x + i * T_1D + t_id_z * T_1D * T_1D];  // Contract y direction
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractTransposeX3dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const int t_id_z, const ScalarIn1 *U,
                                                     const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D + t_id_z * T_1D * T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (t_id_x < P_1D && t_id_y < P_1D && t_id_z < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[t_id_x + i * P_1D] * data.slice[i + t_id_y * T_1D + t_id_z * T_1D * T_1D];  // Contract x direction
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract add x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void ContractTransposeAddX3dFlattened(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const int t_id_z,
                                                        const ScalarIn1 *U, const ScalarIn2 *B, ScalarOut *V) {
  __syncthreads();
  data.slice[t_id_x + t_id_y * T_1D + t_id_z * T_1D * T_1D] = *U;
  __syncthreads();
  if (t_id_x < P_1D && t_id_y < P_1D && t_id_z < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[t_id_x + i * P_1D] * data.slice[i + t_id_y * T_1D + t_id_z * T_1D * T_1D];  // Contract x direction
    }
  }
}

//------------------------------------------------------------------------------
// 3D pack/unpack quadrature values
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_1D, int T_1D, class ScalarOut>
inline __device__ void QPack3d(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const int t_id_z, ScalarOut *U) {
  const CeedInt new_t_id_x = data.t_id_x % Q_1D, new_t_id_y = (data.t_id_x / Q_1D) % Q_1D, new_t_id_z = data.t_id_x / (Q_1D * Q_1D);

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    __syncthreads();
    if (t_id_x < Q_1D && t_id_y < Q_1D) data.slice[t_id_x + t_id_y * T_1D + t_id_z * T_1D * T_1D] = U[comp];
    __syncthreads();
    U[comp] = data.t_id_x < (Q_1D * Q_1D * Q_1D) ? data.slice[new_t_id_x + new_t_id_y * T_1D + new_t_id_z * T_1D * T_1D] : 0.0;
  }
}

template <int NUM_COMP, int Q_1D, int T_1D, class ScalarOut>
inline __device__ void QUnpack3d(SharedData_Cuda &data, const int t_id_x, const int t_id_y, const int t_id_z, ScalarOut *U) {
  const CeedInt old_t_id_x = data.t_id_x % Q_1D, old_t_id_y = (data.t_id_x / Q_1D) % Q_1D, old_t_id_z = data.t_id_x / (Q_1D * Q_1D);

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    __syncthreads();
    if (data.t_id_x < Q_1D * Q_1D * Q_1D) data.slice[old_t_id_x + old_t_id_y * T_1D + old_t_id_z * T_1D * T_1D] = U[comp];
    __syncthreads();
    U[comp] = (t_id_x < Q_1D && t_id_y < Q_1D) ? data.slice[t_id_x + t_id_y * T_1D + t_id_z * T_1D * T_1D] : 0.0;
  }
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void InterpTensor3dFlattened(SharedData_Cuda &data, ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B,
                                               ScalarOut *__restrict__ r_V) {
  const CeedInt t_id_x = data.t_id_x % T_1D, t_id_y = (data.t_id_x / T_1D) % T_1D, t_id_z = data.t_id_x / (T_1D * T_1D);
  CeedScalar    r_t1[1], r_t2[1];

  if (P_1D != T_1D) QUnpack3d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_U);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, &r_U[comp], c_B, r_t1);
    ContractY3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t1, c_B, r_t2);
    ContractZ3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t2, c_B, &r_V[comp]);
  }
  __syncthreads();
  if (P_1D != T_1D) QPack3d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_U);
  if (Q_1D != T_1D) QPack3d<NUM_COMP, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_V);
}

//------------------------------------------------------------------------------
// 3D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void InterpTransposeTensor3dFlattened(SharedData_Cuda &data, ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B,
                                                        ScalarOut *__restrict__ r_V) {
  const CeedInt t_id_x = data.t_id_x % T_1D, t_id_y = (data.t_id_x / T_1D) % T_1D, t_id_z = data.t_id_x / (T_1D * T_1D);
  CeedScalar    r_t1[1], r_t2[1];

  if (Q_1D != T_1D) QUnpack3d<NUM_COMP, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_U);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeZ3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, &r_U[comp], c_B, r_t1);
    ContractTransposeY3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t1, c_B, r_t2);
    ContractTransposeX3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t2, c_B, &r_V[comp]);
  }
  __syncthreads();
  if (P_1D != T_1D) QPack3d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_V);
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarIn3, class ScalarOut>
inline __device__ void GradTensor3dFlattened(SharedData_Cuda &data, ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B, const ScalarIn3 *c_G,
                                             ScalarOut *__restrict__ r_V) {
  const CeedInt t_id_x = data.t_id_x % T_1D, t_id_y = (data.t_id_x / T_1D) % T_1D, t_id_z = data.t_id_x / (T_1D * T_1D);
  CeedScalar    r_t1[1], r_t2[1];

  if (P_1D != T_1D) QUnpack3d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_U);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, &r_U[comp], c_G, r_t1);
    ContractY3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t1, c_B, r_t2);
    ContractZ3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t2, c_B, &r_V[comp + 0 * NUM_COMP]);
    ContractX3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, &r_U[comp], c_B, r_t1);
    ContractY3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t1, c_G, r_t2);
    ContractZ3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t2, c_B, &r_V[comp + 1 * NUM_COMP]);
    ContractX3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, &r_U[comp], c_B, r_t1);
    ContractY3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t1, c_B, r_t2);
    ContractZ3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t2, c_G, &r_V[comp + 2 * NUM_COMP]);
  }
  __syncthreads();
  if (P_1D != T_1D) QPack3d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_U);
  if (Q_1D != T_1D) QPack3d<NUM_COMP * 3, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_V);
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarIn3, class ScalarOut>
inline __device__ void GradTransposeTensor3dFlattened(SharedData_Cuda &data, ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B, const ScalarIn3 *c_G,
                                                      ScalarOut *__restrict__ r_V) {
  const CeedInt t_id_x = data.t_id_x % T_1D, t_id_y = (data.t_id_x / T_1D) % T_1D, t_id_z = data.t_id_x / (T_1D * T_1D);
  CeedScalar    r_t1[1], r_t2[1];

  if (Q_1D != T_1D) QUnpack3d<NUM_COMP * 3, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_U);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeZ3dFlattened<NUM_COMP, t_id_x, t_id_y, t_id_z, P_1D, Q_1D, T_1D>(data, &r_U[comp + 0 * NUM_COMP], c_B, r_t1);
    ContractTransposeY3dFlattened<NUM_COMP, t_id_x, t_id_y, t_id_z, P_1D, Q_1D, T_1D>(data, r_t1, c_B, r_t2);
    ContractTransposeX3dFlattened<NUM_COMP, t_id_x, t_id_y, t_id_z, P_1D, Q_1D, T_1D>(data, r_t2, c_G, &r_V[comp]);
    ContractTransposeZ3dFlattened<NUM_COMP, t_id_x, t_id_y, t_id_z, P_1D, Q_1D, T_1D>(data, &r_U[comp + 1 * NUM_COMP], c_B, r_t1);
    ContractTransposeY3dFlattened<NUM_COMP, t_id_x, t_id_y, t_id_z, P_1D, Q_1D, T_1D>(data, r_t1, c_G, r_t2);
    ContractTransposeAddX3dFlattened<NUM_COMP, t_id_x, t_id_y, t_id_z, P_1D, Q_1D, T_1D>(data, r_t2, c_B, &r_V[comp]);
    ContractTransposeZ3dFlattened<NUM_COMP, t_id_x, t_id_y, t_id_z, P_1D, Q_1D, T_1D>(data, &r_U[comp + 2 * NUM_COMP], c_G, r_t1);
    ContractTransposeY3dFlattened<NUM_COMP, t_id_x, t_id_y, t_id_z, P_1D, Q_1D, T_1D>(data, r_t1, c_B, r_t2);
    ContractTransposeAddX3dFlattened<NUM_COMP, t_id_x, t_id_y, t_id_z, P_1D, Q_1D, T_1D>(data, r_t2, c_B, &r_V[comp]);
  }
  __syncthreads();
  if (P_1D != T_1D) QPack3d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_V);
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarIn3, class ScalarOut>
inline __device__ void GradTensorCollocated3dFlattened(SharedData_Cuda &data, ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B, const ScalarIn3 *c_G,
                                                       ScalarOut *__restrict__ r_V) {
  const CeedInt t_id_x = data.t_id_x % T_1D, t_id_y = (data.t_id_x / T_1D) % T_1D, t_id_z = data.t_id_x / (T_1D * T_1D);
  CeedScalar    r_t1[1], r_t2[1];

  if (P_1D != T_1D) QUnpack3d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_U);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, &r_U[comp], c_B, r_t1);
    ContractY3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t1, c_B, r_t2);
    ContractZ3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t2, c_B, r_t1);
    ContractX3dFlattened<NUM_COMP, Q_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t1, c_G, &r_V[comp + 0 * NUM_COMP]);
    ContractY3dFlattened<NUM_COMP, Q_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t1, c_G, &r_V[comp + 1 * NUM_COMP]);
    ContractZ3dFlattened<NUM_COMP, Q_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t1, c_G, &r_V[comp + 2 * NUM_COMP]);
  }
  __syncthreads();
  if (P_1D != T_1D) QPack3d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_U);
  if (Q_1D != T_1D) QPack3d<NUM_COMP * 3, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_V);
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarIn3, class ScalarOut>
inline __device__ void GradTransposeTensorCollocated3dFlattened(SharedData_Cuda &data, ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_B,
                                                                const ScalarIn3 *c_G, ScalarOut *__restrict__ r_V) {
  const CeedInt t_id_x = data.t_id_x % T_1D, t_id_y = (data.t_id_x / T_1D) % T_1D, t_id_z = data.t_id_x / (T_1D * T_1D);
  CeedScalar    r_t1[1], r_t2[1];

  if (Q_1D != T_1D) QUnpack3d<NUM_COMP * 3, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_U);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeZ3dFlattened<NUM_COMP, Q_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, &r_U[comp + 2 * NUM_COMP], c_G, r_t2);
    ContractTransposeAddY3dFlattened<NUM_COMP, Q_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, &r_U[comp + 1 * NUM_COMP], c_G, r_t2);
    ContractTransposeAddX3dFlattened<NUM_COMP, Q_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, &r_U[comp + 0 * NUM_COMP], c_G, r_t2);
    ContractTransposeZ3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t2, c_B, r_t1);
    ContractTransposeY3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t1, c_B, r_t2);
    ContractTransposeX3dFlattened<NUM_COMP, P_1D, Q_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_t2, c_B, &r_V[comp]);
  }
  __syncthreads();
  if (P_1D != T_1D) QPack3d<NUM_COMP, P_1D, T_1D>(data, t_id_x, t_id_y, t_id_z, r_V);
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
template <int P_1D, int Q_1D, class ScalarIn, class ScalarOut>
inline __device__ void WeightTensor3dFlattened(SharedData_Cuda &data, const ScalarIn *__restrict__ q_weight_1d, ScalarOut *w) {
  const CeedInt t_id_x = data.t_id_x % Q_1D, t_id_y = (data.t_id_x / Q_1D) % Q_1D, t_id_z = data.t_id_x / (Q_1D * Q_1D);

  *w = (t_id_x < Q_1D && t_id_y < Q_1D && t_id_z < Q_1D) ? q_weight_1d[t_id_x] * q_weight_1d[t_id_y] * q_weight_1d[t_id_z] : 0.0;
}
