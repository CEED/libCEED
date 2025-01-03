// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA shared memory non-tensor basis
#include <ceed/types.h>

#include "cuda-shared-basis-nontensor-templates.h"
#include "cuda-shared-basis-read-write-templates.h"

//------------------------------------------------------------------------------
// Interp kernel by dim
//------------------------------------------------------------------------------
extern "C" __global__ void Interp(const CeedInt num_elem, const CeedScalar *c_B, const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D;

  CeedScalar r_U[BASIS_NUM_COMP];
  CeedScalar r_V[BASIS_NUM_COMP];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P>(data, elem, 1, BASIS_P * num_elem, BASIS_P, d_U, r_U);
    InterpNonTensor<BASIS_NUM_COMP, BASIS_P, BASIS_Q>(data, r_U, c_B, r_V);
    WriteElementStrided1d<BASIS_NUM_COMP, BASIS_Q>(data, elem, 1, BASIS_Q * num_elem, BASIS_Q, r_V, d_V);
  }
}

extern "C" __global__ void InterpTranspose(const CeedInt num_elem, const CeedScalar *c_B, const CeedScalar *__restrict__ d_U,
                                           CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D;

  CeedScalar r_U[BASIS_NUM_COMP];
  CeedScalar r_V[BASIS_NUM_COMP];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    ReadElementStrided1d<BASIS_NUM_COMP, BASIS_Q>(data, elem, 1, BASIS_Q * num_elem, BASIS_Q, d_U, r_U);
    InterpTransposeNonTensor<BASIS_NUM_COMP, BASIS_P, BASIS_Q>(data, r_U, c_B, r_V);
    WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P>(data, elem, 1, BASIS_P * num_elem, BASIS_P, r_V, d_V);
  }
}

extern "C" __global__ void InterpTransposeAdd(const CeedInt num_elem, const CeedScalar *c_B, const CeedScalar *__restrict__ d_U,
                                              CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D;

  CeedScalar r_U[BASIS_NUM_COMP];
  CeedScalar r_V[BASIS_NUM_COMP];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    ReadElementStrided1d<BASIS_NUM_COMP, BASIS_Q>(data, elem, 1, BASIS_Q * num_elem, BASIS_Q, d_U, r_U);
    InterpTransposeNonTensor<BASIS_NUM_COMP, BASIS_P, BASIS_Q>(data, r_U, c_B, r_V);
    SumElementStrided1d<BASIS_NUM_COMP, BASIS_P>(data, elem, 1, BASIS_P * num_elem, BASIS_P, r_V, d_V);
  }
}

//------------------------------------------------------------------------------
// Grad kernel by dim
//------------------------------------------------------------------------------
extern "C" __global__ void Grad(const CeedInt num_elem, const CeedScalar *c_G, const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D;

  CeedScalar r_U[BASIS_NUM_COMP];
  CeedScalar r_V[BASIS_NUM_COMP * BASIS_DIM];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P>(data, elem, 1, BASIS_P * num_elem, BASIS_P, d_U, r_U);
    GradNonTensor<BASIS_NUM_COMP, BASIS_DIM, BASIS_P, BASIS_Q>(data, r_U, c_G, r_V);
    WriteElementStrided1d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q>(data, elem, 1, BASIS_Q * num_elem, BASIS_Q, r_V, d_V);
  }
}

extern "C" __global__ void GradTranspose(const CeedInt num_elem, const CeedScalar *c_G, const CeedScalar *__restrict__ d_U,
                                         CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D;

  CeedScalar r_U[BASIS_NUM_COMP * BASIS_DIM];
  CeedScalar r_V[BASIS_NUM_COMP];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    ReadElementStrided1d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q>(data, elem, 1, BASIS_Q * num_elem, BASIS_Q, d_U, r_U);
    GradTransposeNonTensor<BASIS_NUM_COMP, BASIS_DIM, BASIS_P, BASIS_Q>(data, r_U, c_G, r_V);
    WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P>(data, elem, 1, BASIS_P * num_elem, BASIS_P, r_V, d_V);
  }
}

extern "C" __global__ void GradTransposeAdd(const CeedInt num_elem, const CeedScalar *c_G, const CeedScalar *__restrict__ d_U,
                                            CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D;

  CeedScalar r_U[BASIS_NUM_COMP * BASIS_DIM];
  CeedScalar r_V[BASIS_NUM_COMP];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    ReadElementStrided1d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q>(data, elem, 1, BASIS_Q * num_elem, BASIS_Q, d_U, r_U);
    GradTransposeNonTensor<BASIS_NUM_COMP, BASIS_DIM, BASIS_P, BASIS_Q>(data, r_U, c_G, r_V);
    SumElementStrided1d<BASIS_NUM_COMP, BASIS_P>(data, elem, 1, BASIS_P * num_elem, BASIS_P, r_V, d_V);
  }
}

//------------------------------------------------------------------------------
// Weight kernels by dim
//------------------------------------------------------------------------------
extern "C" __global__ void Weight(const CeedInt num_elem, const CeedScalar *__restrict__ q_weight, CeedScalar *__restrict__ d_W) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D;

  CeedScalar r_W[1];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    WeightNonTensor<BASIS_Q>(data, q_weight, r_W);
    WriteElementStrided1d<1, BASIS_Q>(data, elem, 1, BASIS_Q * num_elem, BASIS_Q, r_W, d_W);
  }
}
