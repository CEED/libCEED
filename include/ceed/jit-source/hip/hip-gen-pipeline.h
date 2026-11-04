// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed/types.h>

// Cheetah tensor staging adapted to HIP LDS; no wave-size or implicit synchronization assumptions.
// Algorithm origin: https://doi.org/10.1145/3797905.3807857

// The original Cheetah scalar contraction: one input read feeds every register
// accumulator. Strides express the original CuTe mappings without a dependency.
template <int IN, int OUT, int MATRIX_IN_STRIDE, int MATRIX_OUT_STRIDE, int INPUT_STRIDE, int OUTPUT_STRIDE>
inline __device__ void ContractRegisterVector(const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar values[OUT];
#pragma unroll
  for (CeedInt j = 0; j < OUT; j++) values[j] = 0;
#pragma unroll
  for (CeedInt k = 0; k < IN; k++) {
    const CeedScalar value = U[k * INPUT_STRIDE];
#pragma unroll
    for (CeedInt j = 0; j < OUT; j++) values[j] += value * B[k * MATRIX_IN_STRIDE + j * MATRIX_OUT_STRIDE];
  }
#pragma unroll
  for (CeedInt j = 0; j < OUT; j++) V[j * OUTPUT_STRIDE] = values[j];
}

// Preserve Cheetah's x -> y -> z contraction and next-component preload.
// Physical scratch is [bank,z,y,x] with x contiguous. Each stage writes directly
// to the next stage's consumer layout. No per-plane synchronization is used.
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void ForwardTensorRegister3d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B_x, const CeedScalar *B_y,
                                               const CeedScalar *B_z, CeedScalar *V) {
  const int x = data.t_id_x, y = data.t_id_y;
  __syncthreads();  // Retire the preceding operation before initializing scratch.
  if (x < P_1D && y < P_1D) {
#pragma unroll
    for (CeedInt z = 0; z < P_1D; z++) data.slice[z * T_1D * T_1D + y * T_1D + x] = U[z];
  }
#pragma unroll
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    CeedScalar *a = data.slice + (comp % 2) * T_1D * T_1D * T_1D;
    CeedScalar *b = data.slice + ((comp + 1) % 2) * T_1D * T_1D * T_1D;
    __syncthreads();
    if (x < P_1D && y < P_1D) {
      ContractRegisterVector<P_1D, Q_1D, 1, P_1D, 1, 1>(a + y * T_1D * T_1D + x * T_1D, B_x, b + y * T_1D * T_1D + x * T_1D);
    }
    __syncthreads();
    if (x < Q_1D && y < P_1D) {
      ContractRegisterVector<P_1D, Q_1D, 1, P_1D, T_1D, T_1D>(b + y * T_1D * T_1D + x, B_y, a + y * T_1D * T_1D + x);
    }
    __syncthreads();
    // The second bank's y-stage readers have retired; overlap next input stores
    // with the current component's register-owned z contraction.
    if (comp + 1 < NUM_COMP && x < P_1D && y < P_1D) {
#pragma unroll
      for (CeedInt z = 0; z < P_1D; z++) b[z * T_1D * T_1D + y * T_1D + x] = U[z + (comp + 1) * P_1D];
    }
    if (x < Q_1D && y < Q_1D) {
      ContractRegisterVector<P_1D, Q_1D, 1, P_1D, T_1D * T_1D, 1>(a + y * T_1D + x, B_z, V + comp * Q_1D);
    }
  }
}

// Preserve Cheetah's transpose z -> y -> x order and register extraction.
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void TransposeTensorRegister3d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B_x, const CeedScalar *B_y,
                                                 const CeedScalar *B_z, CeedScalar *V) {
  CeedScalar *a = data.slice;
  CeedScalar *b = a + T_1D * T_1D * T_1D;
  const int   x = data.t_id_x, y = data.t_id_y;
#pragma unroll
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    __syncthreads();
    if (x < Q_1D && y < Q_1D) {
      ContractRegisterVector<Q_1D, P_1D, P_1D, 1, 1, T_1D * T_1D>(U + comp * Q_1D, B_z, b + y * T_1D + x);
    }
    __syncthreads();
    if (x < Q_1D && y < P_1D) {
      ContractRegisterVector<Q_1D, P_1D, P_1D, 1, T_1D, T_1D>(b + y * T_1D * T_1D + x, B_y, a + y * T_1D * T_1D + x);
    }
    __syncthreads();
    if (x < P_1D && y < P_1D) {
      ContractRegisterVector<Q_1D, P_1D, P_1D, 1, 1, 1>(a + y * T_1D * T_1D + x * T_1D, B_x, b + y * T_1D * T_1D + x * T_1D);
    }
    __syncthreads();
    if (x < P_1D && y < P_1D) {
#pragma unroll
      for (CeedInt z = 0; z < P_1D; z++) V[z + comp * P_1D] = b[z * T_1D * T_1D + y * T_1D + x];
    }
  }
}

// INTERP and each GRAD direction share the same recovered contraction owner.
template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void InterpTensorRegister3d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  ForwardTensorRegister3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, U, B, B, B, V);
}

template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void InterpTransposeTensorRegister3d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  TransposeTensorRegister3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, U, B, B, B, V);
}

template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void GradTensorRegister3d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B, const CeedScalar *G, CeedScalar *V) {
  ForwardTensorRegister3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, U, G, B, B, V);
  ForwardTensorRegister3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, U, B, G, B, V + NUM_COMP * Q_1D);
  ForwardTensorRegister3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, U, B, B, G, V + 2 * NUM_COMP * Q_1D);
}

template <int NUM_COMP, int P_1D, int Q_1D, int T_1D>
inline __device__ void GradTransposeTensorRegister3d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B, const CeedScalar *G,
                                                     CeedScalar *V) {
  CeedScalar contribution[NUM_COMP * P_1D];
  TransposeTensorRegister3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, U, G, B, B, V);
  TransposeTensorRegister3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, U + NUM_COMP * Q_1D, B, G, B, contribution);
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt j = 0; j < NUM_COMP * P_1D; j++) V[j] += contribution[j];
  }
  TransposeTensorRegister3d<NUM_COMP, P_1D, Q_1D, T_1D>(data, U + 2 * NUM_COMP * Q_1D, B, B, G, contribution);
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt j = 0; j < NUM_COMP * P_1D; j++) V[j] += contribution[j];
  }
}

template <int NUM_COMP, int Q_1D, int T_1D>
inline __device__ void GradColloSlice3dRegister(SharedData_Hip &data, const CeedInt q, const CeedScalar *__restrict__ r_U, const CeedScalar *s_G,
                                                const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    __syncthreads();
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
      data.slice[data.t_id_x + data.t_id_y * T_1D] = r_U[q + comp * Q_1D];
    }
    __syncthreads();
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
      // X derivative
      r_V[comp + 0 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[comp + 0 * NUM_COMP] += s_G[i + data.t_id_x * Q_1D] * data.slice[i + data.t_id_y * T_1D];
      }
      // Y derivative
      r_V[comp + 1 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[comp + 1 * NUM_COMP] += s_G[i + data.t_id_y * Q_1D] * data.slice[data.t_id_x + i * T_1D];
      }
      // Z derivative
      r_V[comp + 2 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[comp + 2 * NUM_COMP] += c_G[i + q * Q_1D] * r_U[i + comp * Q_1D];
      }
    }
  }
}

template <int NUM_COMP, int Q_1D, int T_1D>
inline __device__ void GradColloSliceTranspose3dRegister(SharedData_Hip &data, const CeedInt q, const CeedScalar *__restrict__ r_U,
                                                         const CeedScalar *s_G, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  // Publish both transverse derivatives together. Separate planes retain their
  // values until both contractions finish, so no intermediate overwrite needs
  // another block-wide synchronization. The caller owns at least two planes.
  CeedScalar *x_values = data.slice;
  CeedScalar *y_values = data.slice + T_1D * T_1D;

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    __syncthreads();
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
      x_values[data.t_id_x + data.t_id_y * T_1D] = r_U[comp + 0 * NUM_COMP];
      y_values[data.t_id_x + data.t_id_y * T_1D] = r_U[comp + 1 * NUM_COMP];
    }
    __syncthreads();
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
      // X derivative
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[q + comp * Q_1D] += s_G[data.t_id_x + i * Q_1D] * x_values[i + data.t_id_y * T_1D];
      }
      // Y derivative
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[q + comp * Q_1D] += s_G[data.t_id_y + i * Q_1D] * y_values[data.t_id_x + i * T_1D];
      }
      // Z derivative
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[i + comp * Q_1D] += c_G[i + q * Q_1D] * r_U[comp + 2 * NUM_COMP];
      }
    }
  }
}

// Original Cheetah's two-plane staging: publish the next register slice into the
// free bank while the current bank is consumed. Axis and transpose are mathematical
// choices, not order classes; bounds describe the actual intermediate shape.
template <int IN, int OUT, int T, int SLICES, int INPUT_X, int INPUT_Y, int OUTPUT_X, int OUTPUT_Y, bool AXIS_X, bool TRANSPOSE>
inline __device__ void ContractPrefetchedPlanes(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  const int  x = data.t_id_x, y = data.t_id_y;
  const int  row              = AXIS_X ? x : y;
  CeedScalar coefficients[IN] = {0};

  if (row < OUT) {
#pragma unroll
    for (CeedInt k = 0; k < IN; k++) coefficients[k] = B[TRANSPOSE ? row + k * OUT : k + row * IN];
  }
  __syncthreads();
  if (x < INPUT_X && y < INPUT_Y) data.slice[x + y * T] = U[0];
  __syncthreads();
#pragma unroll
  for (CeedInt z = 0; z < SLICES; z++) {
    const CeedScalar *current = data.slice + (z % 2) * T * T;
    CeedScalar       *next    = data.slice + ((z + 1) % 2) * T * T;

    if (z + 1 < SLICES && x < INPUT_X && y < INPUT_Y) next[x + y * T] = U[z + 1];
    V[z] = 0;
    if (x < OUTPUT_X && y < OUTPUT_Y) {
#pragma unroll
      for (CeedInt k = 0; k < IN; k++) V[z] += coefficients[k] * current[AXIS_X ? k + y * T : x + k * T];
    }
    __syncthreads();
  }
}

template <int NUM_COMP, int P, int Q, int T>
inline __device__ void InterpTensorPrefetched3d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar x_values[P], y_values[P];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractPrefetchedPlanes<P, Q, T, P, P, P, Q, P, true, false>(data, U + comp * P, B, x_values);
    ContractPrefetchedPlanes<P, Q, T, P, Q, P, Q, Q, false, false>(data, x_values, B, y_values);
    if (data.t_id_x < Q && data.t_id_y < Q) ContractRegisterVector<P, Q, 1, P, 1, 1>(y_values, B, V + comp * Q);
  }
}

template <int NUM_COMP, int P, int Q, int T>
inline __device__ void InterpTransposeTensorPrefetched3d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar z_values[P], y_values[P];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    if (data.t_id_x < Q && data.t_id_y < Q) ContractRegisterVector<Q, P, P, 1, 1, 1>(U + comp * Q, B, z_values);
    ContractPrefetchedPlanes<Q, P, T, P, Q, Q, Q, P, false, true>(data, z_values, B, y_values);
    ContractPrefetchedPlanes<Q, P, T, P, Q, P, P, P, true, true>(data, y_values, B, V + comp * P);
  }
}

// In 2D, consecutive staged values are components rather than z slices.
// Reuse the same two-bank publication and lane-dependent shared coefficients.
template <int NUM_COMP, int P, int Q, int T>
inline __device__ void InterpTensorPrefetched2d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar intermediate[NUM_COMP];
  ContractPrefetchedPlanes<P, Q, T, NUM_COMP, P, P, Q, P, true, false>(data, U, B, intermediate);
  ContractPrefetchedPlanes<P, Q, T, NUM_COMP, Q, P, Q, Q, false, false>(data, intermediate, B, V);
}

template <int NUM_COMP, int P, int Q, int T>
inline __device__ void InterpTransposeTensorPrefetched2d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar intermediate[NUM_COMP];
  ContractPrefetchedPlanes<Q, P, T, NUM_COMP, Q, Q, Q, P, false, true>(data, U, B, intermediate);
  ContractPrefetchedPlanes<Q, P, T, NUM_COMP, Q, P, P, P, true, true>(data, intermediate, B, V);
}

template <int NUM_COMP, int P, int Q, int T>
inline __device__ void GradTensorPrefetched2d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B, const CeedScalar *G, CeedScalar *V) {
  CeedScalar intermediate[NUM_COMP];
  ContractPrefetchedPlanes<P, Q, T, NUM_COMP, P, P, Q, P, true, false>(data, U, G, intermediate);
  ContractPrefetchedPlanes<P, Q, T, NUM_COMP, Q, P, Q, Q, false, false>(data, intermediate, B, V);
  ContractPrefetchedPlanes<P, Q, T, NUM_COMP, P, P, Q, P, true, false>(data, U, B, intermediate);
  ContractPrefetchedPlanes<P, Q, T, NUM_COMP, Q, P, Q, Q, false, false>(data, intermediate, G, V + NUM_COMP);
}

template <int NUM_COMP, int P, int Q, int T>
inline __device__ void GradTransposeTensorPrefetched2d(SharedData_Hip &data, const CeedScalar *U, const CeedScalar *B, const CeedScalar *G,
                                                       CeedScalar *V) {
  CeedScalar intermediate[NUM_COMP], contribution[NUM_COMP];
  ContractPrefetchedPlanes<Q, P, T, NUM_COMP, Q, Q, Q, P, false, true>(data, U, B, intermediate);
  ContractPrefetchedPlanes<Q, P, T, NUM_COMP, Q, P, P, P, true, true>(data, intermediate, G, V);
  ContractPrefetchedPlanes<Q, P, T, NUM_COMP, Q, Q, Q, P, false, true>(data, U + NUM_COMP, G, intermediate);
  ContractPrefetchedPlanes<Q, P, T, NUM_COMP, Q, P, P, P, true, true>(data, intermediate, B, contribution);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) V[comp] += contribution[comp];
}
