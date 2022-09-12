// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include "cuda-shared-basis-tensor-templates.h"

//------------------------------------------------------------------------------
// Interp kernel by dim
//------------------------------------------------------------------------------
extern "C" __global__ void Interp(const CeedInt num_elem
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  BackendData data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
  data.slice = slice + data.t_id_z * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * BASIS_P_1D];
  CeedScalar r_V[BASIS_NUM_COMP * BASIS_Q_1D];

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, d_U, r_U);
      Interp1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, d_V);
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*BASIS_P_1D*num_elem, BASIS_P_1D*BASIS_P_1D, d_U, r_U);
      InterpTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, d_V);
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*BASIS_Q_1D*num_elem, BASIS_Q_1D*BASIS_Q_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*BASIS_P_1D*BASIS_P_1D*num_elem, BASIS_P_1D*BASIS_P_1D*BASIS_P_1D, d_U, r_U);
      InterpTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, d_V);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*BASIS_Q_1D*BASIS_Q_1D*num_elem, BASIS_Q_1D*BASIS_Q_1D*BASIS_Q_1D, r_V, d_V);
    }
  }
}

extern "C" __global__ void InterpTranspose(const CeedInt num_elem
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  BackendData data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
  data.slice = slice + data.t_id_z * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * BASIS_Q_1D];
  CeedScalar r_V[BASIS_NUM_COMP * BASIS_P_1D];

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, d_U, r_U);
      InterpTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, d_V);
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*BASIS_Q_1D*num_elem, BASIS_Q_1D*BASIS_Q_1D, d_U, r_U);
      InterpTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, d_V);
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*BASIS_P_1D*num_elem, BASIS_P_1D*BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*BASIS_Q_1D*BASIS_Q_1D*num_elem, BASIS_Q_1D*BASIS_Q_1D*BASIS_Q_1D, d_U, r_U);
      InterpTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, d_V);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*BASIS_P_1D*BASIS_P_1D*num_elem, BASIS_P_1D*BASIS_P_1D*BASIS_P_1D, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// Grad kernel by dim
//------------------------------------------------------------------------------
extern "C" __global__ void Grad(const CeedInt num_elem, const CeedInt transpose,
                                const CeedScalar *c_B, const CeedScalar *c_G,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  BackendData data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
  data.slice = slice + data.t_id_z * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * BASIS_P_1D];
  CeedScalar r_V[BASIS_NUM_COMP * BASIS_Q_1D];

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, d_U, r_U);
      Grad1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, c_G, d_V);
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*BASIS_P_1D*num_elem, BASIS_P_1D*BASIS_P_1D, comp, d_U, r_U);
      GradTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, c_G, d_V);
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*BASIS_Q_1D*num_elem, BASIS_Q_1D*BASIS_Q_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*BASIS_P_1D*BASIS_P_1D*num_elem, BASIS_P_1D*BASIS_P_1D*BASIS_P_1D, d_U, r_U);
      GradTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, c_G, d_V);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*BASIS_Q_1D*BASIS_Q_1D*num_elem, BASIS_Q_1D*BASIS_Q_1D*BASIS_Q_1D, r_V, d_V);
    }
  }
}

extern "C" __global__ void GradTranspose(const CeedInt num_elem
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  BackendData data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
  data.slice = slice + data.t_id_z * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * BASIS_Q_1D];
  CeedScalar r_V[BASIS_NUM_COMP * BASIS_P_1D];

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, d_U, r_U);
      GradTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, c_G, d_V);
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*BASIS_Q_1D*num_elem, BASIS_Q_1D*BASIS_Q_1D, comp, d_U, r_U);
      GradTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, c_G, d_V);
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*BASIS_P_1D*num_elem, BASIS_P_1D*BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*BASIS_Q_1D*BASIS_Q_1D*num_elem, BASIS_Q_1D*BASIS_Q_1D*BASIS_Q_1D, d_U, r_U);
      GradTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, d_U, c_B, c_G, d_V);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*BASIS_P_1D*BASIS_P_1D*num_elem, BASIS_P_1D*BASIS_P_1D*BASIS_P_1D, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// Weight kernels by dim
//------------------------------------------------------------------------------
extern "C" __global__ void Weight(const CeedInt num_elem,
                                  const CeedScalar *__restrict__ q_weight_1d,
                                  CeedScalar *__restrict__ d_W) {
  extern __shared__ CeedScalar slice[];

  BackendData data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
  data.slice = slice + data.t_id_z * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  CeedScalar r_W[BASIS_Q_1D];

  if (BASIS_DIM == 1) {
    Weight1d<BASIS_Q_1D>(data, q_weight_1d, r_W);
    WriteElementStrided1d<1, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, r_W, d_W);
  } else if (BASIS_DIM == 2) {
    Weight2d<BASIS_Q_1D>(data, q_weight_1d, r_W);
    WriteElementStrided2d<1, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*BASIS_Q_1D*num_elem, BASIS_Q_1D*BASIS_Q_1D, r_W, d_W);
  } else if (BASIS_DIM == 3) {
    Weight3d<BASIS_Q_1D>(data, q_weight_1d, r_W);
    WriteElementStrided3d<1, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D*BASIS_Q_1D*BASIS_Q_1D*num_elem, BASIS_Q_1D*BASIS_Q_1D*BASIS_Q_1D, r_W, d_W);
  }
}

//------------------------------------------------------------------------------
