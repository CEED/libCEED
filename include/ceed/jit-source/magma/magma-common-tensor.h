// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA backend common tensor basis definitions
#ifndef CEED_MAGMA_COMMON_TENSOR_H
#define CEED_MAGMA_COMMON_TENSOR_H

#include "magma-common-defs.h"

////////////////////////////////////////////////////////////////////////////////
// read U or V of a 1D element into shared memory sU[][] or sV[][] --  for all components
// the devptr is assumed to point directly to the element
// must sync after call
template <typename T, int LENGTH, int NUM_COMP>
static __device__ __inline__ void read_1d(const T *devptr, const int compstride, T *sBuffer[NUM_COMP], const int tx) {
  if (tx < LENGTH) {
    for (int comp = 0; comp < NUM_COMP; comp++) {
      sBuffer[comp][tx] = devptr[comp * compstride + tx];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// write V of a 1D element into global memory from sV[][] --  for all components
// the devptr is assumed to point directly to the element
template <typename T, int LENGTH, int NUM_COMP>
static __device__ __inline__ void write_1d(T *sBuffer[NUM_COMP], T *devptr, const int compstride, const int tx) {
  if (tx < LENGTH) {
    for (int comp = 0; comp < NUM_COMP; comp++) {
      devptr[comp * compstride + tx] = sBuffer[comp][tx];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// read U of a 2D element into registers rU[][][] --  for all components of a single dim
// dU is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rU[DIM_U][NUM_COMP][rU_SIZE]
// i_DIM specifies which dimension is being read into in rU
// rU_SIZE can be different from P (e.g. max(P, Q))
// sTmp is a shared memory workspace of size P^2
template <typename T, int P, int DIM_U, int NUM_COMP, int rU_SIZE, int i_DIM>
static __device__ __inline__ void read_U_2d(const T *dU, const int compstride, T rU[DIM_U][NUM_COMP][rU_SIZE], T *sTmp, const int tx) {
  // read U as a batch P of (1 x P) vectors
  // vec 0  : [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
  // vec 1  : [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
  // ...
  // vec P-1: [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
  // threads collaboratively read vec0 and then vec1 and so on
  // but for the kernel, we want
  // thread 0 to hold all of vec0 in registers, and
  // thread 1 to hold all of vec1 in registers, and and so on
  // so we need to transpose
  for (int comp = 0; comp < NUM_COMP; comp++) {
    // read from global memory into shared memory
    if (tx < P) {
      for (int i = 0; i < P; i++) {
        sTmp[i * P + tx] = dU[comp * compstride + i * P + tx];
      }
    }
    __syncthreads();

    if (tx < P) {
      for (int i = 0; i < P; i++) {
        rU[i_DIM][comp][i] = sTmp[tx * P + i];
      }
    }
    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////////////
// read V of a 2D element into registers rV[][][] --  for all components of a single dim
// dV is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rV[DIM_V][NUM_COMP][rV_SIZE]
// i_DIM specifies which dimension is being read into in rV
// rV_SIZE can be different from P (e.g. max(P, Q))
template <typename T, int Q, int DIM_V, int NUM_COMP, int rV_SIZE, int i_DIM>
static __device__ __inline__ void read_V_2d(const T *dV, const int compstride, T rV[DIM_V][NUM_COMP][rV_SIZE], const int tx) {
  if (tx < Q) {
    for (int comp = 0; comp < NUM_COMP; comp++) {
      for (int j = 0; j < Q; j++) {
        rV[i_DIM][comp][j] = dV[comp * compstride + j * Q + tx];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// write V of a 2D element from registers rV[][][] to global memory --  for all components of a single dim
// dV is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rV[DIM_V][NUM_COMP][rV_SIZE]
// i_DIM specifies which dimension is being written to in dV
// rV_SIZE can be different from P (e.g. max(P, Q))
template <typename T, int Q, int DIM_V, int NUM_COMP, int rV_SIZE, int i_DIM>
static __device__ __inline__ void write_V_2d(T *dV, const int compstride, T rV[DIM_V][NUM_COMP][rV_SIZE], const int tx) {
  if (tx < Q) {
    for (int comp = 0; comp < NUM_COMP; comp++) {
      for (int j = 0; j < Q; j++) {
        dV[comp * compstride + j * Q + tx] = rV[i_DIM][comp][j];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// read U of a 3D element into registers rU[][][] --  for all components of a single dim
// dU is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rU[DIM_U][NUM_COMP][rU_SIZE]
// i_DIM specifies which dimension is being read into in rU
// rU_SIZE can be different from P (e.g. max(P, Q))
// sTmp is a shared memory workspace of size P^3
template <typename T, int P, int DIM_U, int NUM_COMP, int rU_SIZE, int i_DIM>
static __device__ __inline__ void read_U_3d(const T *dU, const int compstride, T rU[DIM_U][NUM_COMP][rU_SIZE], T *sTmp, const int tx) {
  // read U as a batch P^2 of (1 x P_) vectors
  // vec 0    : [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
  // vec 1    : [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
  // ...
  // vec P^2-1: [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
  // threads collaboratively read vec0 and then vec1 and so on
  // but for the kernel, we want
  // thread 0 to hold all of vec0 in registers, and
  // thread 1 to hold all of vec1 in registers, and and so on
  // so we need to transpose
  for (int comp = 0; comp < NUM_COMP; comp++) {
    // read from global memory into shared memory
    if (tx < P * P) {
      for (int i = 0; i < P; i++) {
        sTmp[i * P * P + tx] = dU[comp * compstride + i * P * P + tx];
      }
    }
    __syncthreads();

    if (tx < P * P) {
      for (int i = 0; i < P; i++) {
        rU[i_DIM][comp][i] = sTmp[tx * P + i];
      }
    }
    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////////////
// read V of a 3D element into registers rV[][][] --  for all components of a single dim
// dV is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rV[DIM_V][NUM_COMP][rV_SIZE]
// i_DIM specifies which dimension is being read into in rV
// rV_SIZE can be different from P (e.g. max(P, Q))
template <typename T, int Q, int DIM_V, int NUM_COMP, int rV_SIZE, int i_DIM>
static __device__ __inline__ void read_V_3d(const T *dV, const int compstride, T rV[DIM_V][NUM_COMP][rV_SIZE], const int tx) {
  if (tx < Q * Q) {
    for (int comp = 0; comp < NUM_COMP; comp++) {
      for (int j = 0; j < Q; j++) {
        rV[i_DIM][comp][j] = dV[comp * compstride + j * (Q * Q) + tx];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// write V of a 3D element from registers rV[][][] to global memory --  for all components of a single dim
// dV is assumed to point directly to the element (i.e. already offset by elem-stride)
// register is assumed to be rV[DIM_V][NUM_COMP][rV_SIZE]
// i_DIM specifies which dimension is being written to in dV
// rV_SIZE can be different from P (e.g. max(P, Q))
template <typename T, int Q, int DIM_V, int NUM_COMP, int rV_SIZE, int i_DIM>
static __device__ __inline__ void write_V_3d(T *dV, const int compstride, T rV[DIM_V][NUM_COMP][rV_SIZE], const int tx) {
  if (tx < (Q * Q)) {
    for (int comp = 0; comp < NUM_COMP; comp++) {
      for (int j = 0; j < Q; j++) {
        dV[comp * compstride + j * (Q * Q) + tx] = rV[i_DIM][comp][j];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// reads T (no-trans) into shared memory
// T is B x J
// must sync after call
template <int B, int J>
static __device__ __inline__ void read_T_notrans_gm2sm(const int tx, const CeedScalar *dT, CeedScalar *sT) {
  if (tx < B) {
    for (int i = 0; i < J; i++) {
      sT[i * B + tx] = dT[i * B + tx];
    }
  }
  // must sync after call
}

////////////////////////////////////////////////////////////////////////////////
// reads T (trans) into shared memory
// T is J x B
// must sync after call
template <int B, int J>
static __device__ __inline__ void read_T_trans_gm2sm(const int tx, const CeedScalar *dT, CeedScalar *sT) {
  if (tx < J) {
    for (int i = 0; i < B; i++) {
      sT[tx * B + i] = dT[i * J + tx];
    }
  }
  // must sync after call
}

#endif  // CEED_MAGMA_COMMON_TENSOR_H
