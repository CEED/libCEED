// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA tensor product basis with AtPoints evaluation
#include <ceed/types.h>

#include "cuda-shared-basis-read-write-templates.h"
#include "cuda-shared-basis-tensor-at-points-templates.h"
#include "cuda-shared-basis-tensor-templates.h"

//------------------------------------------------------------------------------
// Tensor Basis Kernels AtPoints
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Interp
//------------------------------------------------------------------------------
extern "C" __global__ void InterpAtPoints(const CeedInt num_elem, const CeedScalar *__restrict__ c_B, const CeedInt *__restrict__ points_per_elem,
                                          const CeedScalar *__restrict__ d_X, const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
  CeedScalar r_C[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP];

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    // Map to coefficients
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, d_U, r_U);
      Interp1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_C);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      InterpTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_C);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                       BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      InterpTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_C);
    }

    // Map to points
    InterpAtPoints<BASIS_DIM, BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_Q_1D>(data, num_elem * BASIS_NUM_PTS, r_C, &d_X[elem * BASIS_NUM_PTS], r_V,
                                                                         &d_V[elem * BASIS_NUM_PTS]);
  }
}

extern "C" __global__ void InterpTransposeAtPoints(const CeedInt num_elem, const CeedScalar *__restrict__ c_B,
                                                   const CeedInt *__restrict__ points_per_elem, const CeedScalar *__restrict__ d_X,
                                                   const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP];
  CeedScalar r_C[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    // Map from points
    InterpTransposeAtPoints<BASIS_DIM, BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_Q_1D>(data, num_elem * BASIS_NUM_PTS, points_per_elem[elem],
                                                                                  &d_U[elem * BASIS_NUM_PTS], r_U, &d_X[elem * BASIS_NUM_PTS], r_C);

    // Map from coefficients
    if (BASIS_DIM == 1) {
      InterpTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_C, c_B, r_V);
      SumElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      InterpTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_C, c_B, r_V);
      SumElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      InterpTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_C, c_B, r_V);
      SumElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                      BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// Grad
//------------------------------------------------------------------------------
extern "C" __global__ void GradAtPoints(const CeedInt num_elem, const CeedScalar *__restrict__ c_B, const CeedInt *__restrict__ points_per_elem,
                                        const CeedScalar *__restrict__ d_X, const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
  CeedScalar r_C[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * BASIS_DIM];

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    // Map to coefficients
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, d_U, r_U);
      Interp1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_C);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      InterpTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_C);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                       BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      InterpTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_C);
    }

    // Map to points
    GradAtPoints<BASIS_DIM, BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_Q_1D>(data, num_elem * BASIS_NUM_PTS, r_C, &d_X[elem * BASIS_NUM_PTS], r_V,
                                                                       &d_V[elem * BASIS_NUM_PTS]);
  }
}

extern "C" __global__ void GradTransposeAtPoints(const CeedInt num_elem, const CeedScalar *__restrict__ c_B,
                                                 const CeedInt *__restrict__ points_per_elem, const CeedScalar *__restrict__ d_X,
                                                 const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * BASIS_DIM];
  CeedScalar r_C[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    // Map from points
    GradTransposeAtPoints<BASIS_DIM, BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_Q_1D>(data, num_elem * BASIS_NUM_PTS, points_per_elem[elem],
                                                                                &d_U[elem * BASIS_NUM_PTS], r_U, &d_X[elem * BASIS_NUM_PTS], r_C);

    // Map from coefficients
    if (BASIS_DIM == 1) {
      InterpTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_C, c_B, r_V);
      SumElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      InterpTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_C, c_B, r_V);
      SumElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      InterpTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_C, c_B, r_V);
      SumElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                      BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    }
  }
}
