// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA shared memory tensor product basis
#ifndef _ceed_cuda_shared_basis_tensor_h
#define _ceed_cuda_shared_basis_tensor_h

#include <ceed.h>

#include "cuda-shared-basis-read-write-templates.h"
#include "cuda-shared-basis-tensor-templates.h"

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
  data.slice  = slice + data.t_id_z * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, d_U, r_U);
      Interp1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_V);
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * num_elem, BASIS_Q_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      InterpTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_V);
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * num_elem, BASIS_Q_1D * BASIS_Q_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                       BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      InterpTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_V);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D * num_elem,
                                                        BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D, r_V, d_V);
    }
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
  data.slice  = slice + data.t_id_z * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * num_elem, BASIS_Q_1D, d_U, r_U);
      InterpTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_V);
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * num_elem, BASIS_Q_1D * BASIS_Q_1D, d_U, r_U);
      InterpTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_V);
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D * num_elem,
                                                       BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D, d_U, r_U);
      InterpTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, r_V);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                        BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// Grad kernel by dim
//------------------------------------------------------------------------------
extern "C" __global__ void Grad(const CeedInt num_elem, const CeedScalar *c_B, const CeedScalar *c_G, const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * BASIS_DIM * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, d_U, r_U);
      Grad1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, c_G, r_V);
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * num_elem, BASIS_Q_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      GradTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, c_G, r_V);
      WriteElementStrided2d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * num_elem, BASIS_Q_1D * BASIS_Q_1D, r_V,
                                                                    d_V);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                       BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      if (BASIS_HAS_COLLOCATED_GRAD) GradTensorCollocated3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, c_G, r_V);
      else GradTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, c_G, r_V);
      WriteElementStrided3d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D * num_elem,
                                                                    BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D, r_V, d_V);
    }
  }
}

extern "C" __global__ void GradTranspose(const CeedInt num_elem, const CeedScalar *c_B, const CeedScalar *c_G, const CeedScalar *__restrict__ d_U,
                                         CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * BASIS_DIM * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * num_elem, BASIS_Q_1D, d_U, r_U);
      GradTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, c_G, r_V);
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * num_elem, BASIS_Q_1D * BASIS_Q_1D, d_U,
                                                                   r_U);
      GradTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, c_G, r_V);
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP * BASIS_DIM, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D * num_elem,
                                                                   BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D, d_U, r_U);
      if (BASIS_HAS_COLLOCATED_GRAD) GradTransposeTensorCollocated3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, c_G, r_V);
      else GradTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, c_B, c_G, r_V);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                        BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// Weight kernels by dim
//------------------------------------------------------------------------------
extern "C" __global__ void Weight(const CeedInt num_elem, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *__restrict__ d_W) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

  CeedScalar r_W[BASIS_DIM > 2 ? BASIS_Q_1D : 1];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    if (BASIS_DIM == 1) {
      Weight1d<BASIS_Q_1D>(data, q_weight_1d, r_W);
      WriteElementStrided1d<1, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * num_elem, BASIS_Q_1D, r_W, d_W);
    } else if (BASIS_DIM == 2) {
      WeightTensor2d<BASIS_Q_1D>(data, q_weight_1d, r_W);
      WriteElementStrided2d<1, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * num_elem, BASIS_Q_1D * BASIS_Q_1D, r_W, d_W);
    } else if (BASIS_DIM == 3) {
      WeightTensor3d<BASIS_Q_1D>(data, q_weight_1d, r_W);
      WriteElementStrided3d<1, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D * num_elem, BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D, r_W,
                                           d_W);
    }
  }
}

//------------------------------------------------------------------------------

#endif
