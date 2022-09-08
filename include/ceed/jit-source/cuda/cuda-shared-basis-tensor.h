// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include "cuda-shared-basis-tensor-templates.h"

//------------------------------------------------------------------------------
// 1D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void Interp1d(BackendData &data, const CeedInt num_elem, const CeedInt transpose,
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  CeedScalar r_U[BASIS_BUF_LEN];
  CeedScalar r_V[BASIS_BUF_LEN];

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    if (transpose) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, d_U, &r_U);
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        ContractTransposeX1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp], c_B, &r_V[comp]);
      }
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, &r_V, d_V);
    } else {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, d_U, &r_U);
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        ContractX1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp], c_B, &r_V[comp]);
      }
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, &r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 1D derivatives at quadrature points
//------------------------------------------------------------------------------
inline __device__ void Grad1d(BackendData &data, const CeedInt num_elem, const CeedInt transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V) {
  CeedScalar r_U[BASIS_BUF_LEN];
  CeedScalar r_V[BASIS_BUF_LEN];

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    if (transpose) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, d_U, &r_U);
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        ContractTransposeX1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp], c_G, &r_V[comp]);
      }
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, &r_V, d_V);
    } else {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, d_U, &r_U);
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        ContractX1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp], c_G, &r_V[comp]);
      }
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, &r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 1D Quadrature weights
//------------------------------------------------------------------------------
__device__ void weight1d(BackendData &data, const CeedInt num_elem, const CeedScalar *q_weight_1d,
                         CeedScalar *__restrict__ d_W) {
  const CeedScalar weight = q_weight_1d[data.t_id_x];

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    const CeedInt ind = data.t_id_x + elem*BASIS_Q_1D;
    d_W[ind] = weight;
  }
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void Interp2d(BackendData &data, const CeedInt num_elem, const CeedInt transpose,
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  CeedScalar r_U[BASIS_BUF_LEN];
  CeedScalar r_t[BASIS_BUF_LEN];
  CeedScalar r_V[BASIS_BUF_LEN];

  const CeedInt block_elem = data.t_id_z % BASIS_NUM_COMP;
  const CeedInt elems_per_block = blockDim.z / BASIS_NUM_COMP;
  const CeedInt comp = data.t_id_z % BASIS_NUM_COMP;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    if (transpose) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, comp, d_U, &r_U);
      ContractTransposeY2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp], c_B, &r_t);
      ContractTransposeX2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t, c_B, &r_V[comp]);
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, comp, &r_V, d_V);
    } else {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, comp, d_U, &r_U);
      ContractX2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp], c_B, &r_t);
      ContractY2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t, c_B, &r_V[comp]);
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, comp, &r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points
//------------------------------------------------------------------------------
inline __device__ void Grad2d(BackendData &data, const CeedInt num_elem, const CeedInt transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V) {
  CeedScalar r_U[BASIS_BUF_LEN];
  CeedScalar r_t[BASIS_BUF_LEN];
  CeedScalar r_V[BASIS_BUF_LEN];

  const CeedInt block_elem = data.t_id_z % BASIS_NUM_COMP;
  const CeedInt elems_per_block = blockDim.z / BASIS_NUM_COMP;
  const CeedInt comp = data.t_id_z % BASIS_NUM_COMP;
  CeedInt dim = 0;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    if (transpose) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, d_U, &r_U);
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        dim = 0;
        ContractTransposeY2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp + dim*BASIS_NUM_COMP], c_B, &r_t);
        ContractTransposeX2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t, c_G, &r_V[comp]);
        dim = 1;
        ContractTransposeY2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp dim*BASIS_NUM_COMP], c_G, &r_t);
        ContractTransposeAddX2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t, c_B, &r_V[comp]);
      }
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, &r_V, d_V);
    } else {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, d_U, &r_U);
      for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
        dim = 0;
        ContractX2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp], c_G, &r_t);
        ContractY2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t, c_B, &r_V[comp + dim * BASIS_NUM_COMP]);
        dim = 1;
        ContractX2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp], c_B, &r_t);
        ContractY2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t, c_G, &r_V[comp + dim * BASIS_NUM_COMP]);
      }
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, &r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
__device__ void Weight2d(BackendData &data, const CeedInt num_elem, const CeedScalar *q_weight_1d,
                         CeedScalar *__restrict__ d_W) {
  const CeedScalar weight = q_weight_1d[data.t_id_x]*q_weight_1d[data.t_id_y];
  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    const CeedInt ind = data.t_id_x + data.t_id_y*BASIS_Q_1D + elem*BASIS_Q_1D*BASIS_Q_1D;
    d_W[ind] = weight;
  }
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points
//------------------------------------------------------------------------------
inline __device__ void Interp3d(BackendData &data, const CeedInt num_elem, const CeedInt transpose,
                                const CeedScalar *c_B,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  CeedScalar r_U[BASIS_BUF_LEN];
  CeedScalar r_t_1[BASIS_BUF_LEN];
  CeedScalar r_t_2[BASIS_BUF_LEN];
  CeedScalar r_V[BASIS_BUF_LEN];

  const CeedInt block_elem = data.t_id_z / BASIS_NUM_COMP;
  const CeedInt elems_per_block = blockDim.z  BASIS_NUM_COMP;
  const CeedInt comp = data.t_id_z % BASIS_NUM_COMP;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    if (transpose) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, d_U, &r_U);
      ContractTransposeZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp], c_B, &r_t_1);
      ContractTransposeY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_t_2);
      ContractTransposeX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_2, c_B, &r_V[comp]);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, &r_V, d_V);
    } else {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, d_U, &r_U);
      ContractX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp], c_B, &r_t_1);
      ContractY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_t_2);
      ContractZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_2, c_B, &r_V[comp]);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, &r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
// TODO: rename to avoid conflict?
inline __device__ void Grad3d(BackendData &data, const CeedInt num_elem, const CeedInt transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V) {
  CeedScalar r_U[BASIS_BUF_LEN];
  CeedScalar r_t_1[BASIS_BUF_LEN];
  CeedScalar r_t_2[BASIS_BUF_LEN];
  CeedScalar r_V[BASIS_BUF_LEN];

  const CeedInt block_elem = data.t_id_z / BASIS_NUM_COMP;
  const CeedInt elems_per_block = blockDim.z / BASIS_NUM_COMP;
  const CeedInt comp = data.t_id_z % BASIS_NUM_COMP;
  CeedInt dim = 0
  // It looks like z-threads are BASIS_NUM_COMP * batch_size
  // This is different from cuda-gen where z-threads are only batch_size
  // Should we generalize cuda-gen to support this kind of parallization strategy?
  CeedInt dim;

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    if (transpose) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, d_U, &r_U);
      dim = 0;
      ContractTransposeZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D], c_B, &r_t_1);
      ContractTransposeY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_t_2);
      ContractTransposeX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_2, c_G, &r_V[comp*BASIS_P_1D]);
      dim = 1;
      ContractTransposeZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D], c_B, &r_t_1);
      ContractTransposeY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_t_2);
      ContractTransposeAddX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_2, c_G, &r_V[comp*BASIS_P_1D]);
      dim = 2;
      ContractTransposeZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D], c_B, &r_t_1);
      ContractTransposeY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_t_2);
      ContractTransposeAddX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_2, c_G, &r_V[comp*BASIS_P_1D]);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, &r_V, d_V);
    } else {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, d_U, &r_U);
      dim = 0;
      ContractX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp*BASIS_P_1D], c_G, &r_t_1);
      ContractY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_t_2);
      ContractZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_2, c_B, &r_V[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D]);
      dim = 1;
      ContractX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp*BASIS_P_1D], c_B, &r_t_1);
      ContractY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_G, &r_t_2);
      ContractZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_2, c_B, &r_V[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D]);
      dim = 2;
      ContractX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp*BASIS_P_1D], c_B, &r_t_1);
      ContractY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_t_2);
      ContractZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_2, c_G, &r_V[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D]);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, &r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points, collocated
//------------------------------------------------------------------------------
inline __device__ void GradCollocated3d(BackendData &data, const CeedInt num_elem, const CeedInt transpose,
                              const CeedScalar *c_B, const CeedScalar *c_G,
                              const CeedScalar *__restrict__ d_U,
                              CeedScalar *__restrict__ d_V) {
  CeedScalar r_U[BASIS_BUF_LEN];
  CeedScalar r_t_1[BASIS_BUF_LEN];
  CeedScalar r_t_2[BASIS_BUF_LEN];
  CeedScalar r_V[BASIS_BUF_LEN];

  const CeedInt block_elem = t_in_z / BASIS_NUM_COMP;
  const CeedInt elems_per_block = blockDim.z / BASIS_NUM_COMP;
  const CeedInt comp = data.t_id_z % BASIS_NUM_COMP;
  CeedInt dim = 0
  // It looks like z-threads are BASIS_NUM_COMP * batch_size
  // This is different from cuda-gen where z-threads are only batch_size
  // Should we generalize cuda-gen to support this kind of parallization strategy?

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    if (transpose) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, d_U, &r_U);
      dim = 2;
      ContractTransposeZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D], c_B, &r_t_2);
      dim = 1;
      ContractTransposeAddY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D], c_B, &r_t_2);
      dim = 0;
      ContractTransposeAddX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D], c_B, &r_t_2);
      ContractTransposeZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_2, c_B, &r_t_1);
      ContractTransposeY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_t_2);
      ContractTransposeX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_2, c_B, &r_V[comp]);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, &r_V, d_V);
    } else {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D*num_elem, BASIS_P_1D, d_U, &r_U);
      ContractX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_U[comp], c_B, &r_t_1);
      ContractY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_t_2);
      ContractZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_2, c_B, &r_t_1);
      dim = 0;
      ContractX3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_V[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D]);
      dim = 1;
      ContractY3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_V[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D]);
      dim = 2;
      ContractZ3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, &r_t_1, c_B, &r_V[comp*BASIS_Q_1D + dim*BASIS_NUM_COMP*BASIS_Q_1D]);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_Q_1D*num_elem, BASIS_Q_1D, &r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
__device__ void Weight3d(BackendData &data, const CeedInt num_elem, const CeedScalar *q_weight_1d,
                         CeedScalar *__restrict__ d_W) {
  const CeedScalar weight = q_weight_1d[data.t_id_x]*q_weight_1d[data.t_id_y]*q_weight_1d[data.t_id_z];
  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x*blockDim.z) {
    const CeedInt ind = data.t_id_x + data.t_id_y*BASIS_Q_1D + data.t_id_z*BASIS_Q_1D*BASIS_Q_1D + elem*BASIS_Q_1D*BASIS_Q_1D*BASIS_Q_1D;
    d_W[ind] = weight;
  }
}

//------------------------------------------------------------------------------
// Basis kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Interp kernel by dim
//------------------------------------------------------------------------------
extern "C" __global__ void Interp(const CeedInt num_elem, const CeedInt transpose,
                                  const CeedScalar *c_B,
                                  const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  BackendData data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id  = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;
  data.slice = slice + data.tidz * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  if (BASIS_DIM == 1) {
    Interp1d(data, num_elem, transpose, c_B, d_U, d_V);
  } else if (BASIS_DIM == 2) {
    Interp2d(data, num_elem, transpose, c_B, d_U, d_V);
  } else if (BASIS_DIM == 3) {
    Interp3d(data, num_elem, transpose, c_B, d_U, d_V);
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
  data.slice = slice + data.tidz * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  if (BASIS_DIM == 1) {
    Grad1d(data, num_elem, transpose, c_B, c_G, d_U, d_V);
  } else if (BASIS_DIM == 2) {
    Grad2d(data, num_elem, transpose, c_B, c_G, d_U, d_V);
  } else if (BASIS_DIM == 3) {
    if (BASIS_IS_COLLOCATED) {
      GradCollocated3d(data, num_elem, transpose, c_B, c_G, d_U, d_V);
    } else {
      Grad3d(data, num_elem, transpose, c_B, c_G, d_U, d_V);
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
  data.slice = slice + data.tidz * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  if (BASIS_DIM == 1) {
    Weight1d(data, num_elem, q_weight_1d, d_W);
  } else if (BASIS_DIM == 2) {
    Weight2d(data, num_elem, q_weight_1d, d_W);
  } else if (BASIS_DIM == 3) {
    Weight3d(data, num_elem, q_weight_1d, d_W);
  }
}

//------------------------------------------------------------------------------
