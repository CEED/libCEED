// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
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
extern "C" __global__ void InterpAtPoints(const CeedInt num_elem, const CeedScalarBase *__restrict__ c_B, const CeedInt *__restrict__ points_per_elem,
                                          const CeedScalarBase *__restrict__ d_X, const CeedScalarBase *__restrict__ d_U,
                                          CeedScalarBase *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  CeedScalar r_X[BASIS_DIM];
  CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
  CeedScalar r_C[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP];

  // load interp_1d into shared memory
  __shared__ CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  LoadMatrix<BASIS_P_1D, BASIS_Q_1D>(data, c_B, s_B);
  __syncthreads();

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    // Map to coefficients
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, d_U, r_U);
      Interp1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_U, s_B, r_C);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      InterpTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_U, s_B, r_C);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                       BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      InterpTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_U, s_B, r_C);
    }

    // Map to points
    const CeedInt point_loop_bound = (blockDim.x * blockDim.y) * ceil(1.0 * BASIS_NUM_PTS / (blockDim.x * blockDim.y));

    for (CeedInt i = threadIdx.x + threadIdx.y * blockDim.x; i < point_loop_bound; i += blockDim.x * blockDim.y) {
      const CeedInt p = i % BASIS_NUM_PTS;

      ReadPoint<BASIS_DIM, BASIS_NUM_PTS>(data, elem, p, BASIS_NUM_PTS, 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, d_X, r_X);
      if (BASIS_DIM == 1) {
        InterpAtPoints1d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_C, r_X, r_V);
      } else if (BASIS_DIM == 2) {
        InterpAtPoints2d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_C, r_X, r_V);
      } else if (BASIS_DIM == 3) {
        InterpAtPoints3d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_C, r_X, r_V);
      }
      WritePoint<BASIS_NUM_COMP, BASIS_NUM_PTS>(data, elem, p, BASIS_NUM_PTS, 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, r_V, d_V);
    }
  }
}

extern "C" __global__ void InterpTransposeAtPoints(const CeedInt num_elem, const CeedScalarBase *__restrict__ c_B,
                                                   const CeedInt *__restrict__ points_per_elem, const CeedScalarBase *__restrict__ d_X,
                                                   const CeedScalarBase *__restrict__ d_U, CeedScalarBase *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  CeedScalar r_X[BASIS_DIM];
  CeedScalar r_U[BASIS_NUM_COMP];
  CeedScalar r_C[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];

  // load interp_1d into shared memory
  __shared__ CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  LoadMatrix<BASIS_P_1D, BASIS_Q_1D>(data, c_B, s_B);
  __syncthreads();

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    // Clear register
    for (CeedInt i = 0; i < BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1); i++) r_C[i] = 0.0;

    // Clear output vector
    for (CeedInt i = 0; i < BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1); i++) r_V[i] = 0.0;
    if (BASIS_DIM == 1) {
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                        BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    }

    // Map from points
    const CeedInt point_loop_bound = (blockDim.x * blockDim.y) * ceil(1.0 * BASIS_NUM_PTS / (blockDim.x * blockDim.y));

    for (CeedInt i = threadIdx.x + threadIdx.y * blockDim.x; i < point_loop_bound; i += blockDim.x * blockDim.y) {
      const CeedInt p = i % BASIS_NUM_PTS;

      ReadPoint<BASIS_DIM, BASIS_NUM_PTS>(data, elem, p, BASIS_NUM_PTS, 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, d_X, r_X);
      ReadPoint<BASIS_NUM_COMP, BASIS_NUM_PTS>(data, elem, i, points_per_elem[elem], 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, d_U, r_U);
      if (BASIS_DIM == 1) {
        InterpTransposeAtPoints1d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      } else if (BASIS_DIM == 2) {
        InterpTransposeAtPoints2d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      } else if (BASIS_DIM == 3) {
        InterpTransposeAtPoints3d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      }
    }

    // Map from coefficients
    if (BASIS_DIM == 1) {
      InterpTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      InterpTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      InterpTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                      BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    }
  }
}

extern "C" __global__ void InterpTransposeAddAtPoints(const CeedInt num_elem, const CeedScalarBase *__restrict__ c_B,
                                                      const CeedInt *__restrict__ points_per_elem, const CeedScalarBase *__restrict__ d_X,
                                                      const CeedScalarBase *__restrict__ d_U, CeedScalarBase *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  CeedScalar r_X[BASIS_DIM];
  CeedScalar r_U[BASIS_NUM_COMP];
  CeedScalar r_C[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];

  // load interp_1d into shared memory
  __shared__ CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  LoadMatrix<BASIS_P_1D, BASIS_Q_1D>(data, c_B, s_B);
  __syncthreads();

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    // Clear register
    for (CeedInt i = 0; i < BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1); i++) r_C[i] = 0.0;

    // Map from points
    const CeedInt point_loop_bound = (blockDim.x * blockDim.y) * ceil(1.0 * BASIS_NUM_PTS / (blockDim.x * blockDim.y));

    for (CeedInt i = threadIdx.x + threadIdx.y * blockDim.x; i < point_loop_bound; i += blockDim.x * blockDim.y) {
      const CeedInt p = i % BASIS_NUM_PTS;

      ReadPoint<BASIS_DIM, BASIS_NUM_PTS>(data, elem, p, BASIS_NUM_PTS, 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, d_X, r_X);
      ReadPoint<BASIS_NUM_COMP, BASIS_NUM_PTS>(data, elem, i, points_per_elem[elem], 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, d_U, r_U);
      if (BASIS_DIM == 1) {
        InterpTransposeAtPoints1d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      } else if (BASIS_DIM == 2) {
        InterpTransposeAtPoints2d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      } else if (BASIS_DIM == 3) {
        InterpTransposeAtPoints3d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      }
    }

    // Map from coefficients
    if (BASIS_DIM == 1) {
      InterpTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      InterpTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      InterpTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                      BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    }
  }
}

//------------------------------------------------------------------------------
// Grad
//------------------------------------------------------------------------------
extern "C" __global__ void GradAtPoints(const CeedInt num_elem, const CeedScalarBase *__restrict__ c_B, const CeedInt *__restrict__ points_per_elem,
                                        const CeedScalarBase *__restrict__ d_X, const CeedScalarBase *__restrict__ d_U,
                                        CeedScalarBase *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  CeedScalar r_X[BASIS_DIM];
  CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
  CeedScalar r_C[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * BASIS_DIM];

  // load interp_1d into shared memory
  __shared__ CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  LoadMatrix<BASIS_P_1D, BASIS_Q_1D>(data, c_B, s_B);
  __syncthreads();

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    // Map to coefficients
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, d_U, r_U);
      Interp1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_U, s_B, r_C);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      InterpTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_U, s_B, r_C);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                       BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      InterpTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_U, s_B, r_C);
    }

    // Map to points
    const CeedInt point_loop_bound = (blockDim.x * blockDim.y) * ceil(1.0 * BASIS_NUM_PTS / (blockDim.x * blockDim.y));

    for (CeedInt i = threadIdx.x + threadIdx.y * blockDim.x; i < point_loop_bound; i += blockDim.x * blockDim.y) {
      const CeedInt p = i % BASIS_NUM_PTS;

      ReadPoint<BASIS_DIM, BASIS_NUM_PTS>(data, elem, p, BASIS_NUM_PTS, 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, d_X, r_X);
      if (BASIS_DIM == 1) {
        GradAtPoints1d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_C, r_X, r_V);
      } else if (BASIS_DIM == 2) {
        GradAtPoints2d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_C, r_X, r_V);
      } else if (BASIS_DIM == 3) {
        GradAtPoints3d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_C, r_X, r_V);
      }
      WritePoint<BASIS_NUM_COMP * BASIS_DIM, BASIS_NUM_PTS>(data, elem, p, BASIS_NUM_PTS, 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, r_V, d_V);
    }
  }
}

extern "C" __global__ void GradTransposeAtPoints(const CeedInt num_elem, const CeedScalarBase *__restrict__ c_B,
                                                 const CeedInt *__restrict__ points_per_elem, const CeedScalarBase *__restrict__ d_X,
                                                 const CeedScalarBase *__restrict__ d_U, CeedScalarBase *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  CeedScalar r_X[BASIS_DIM];
  CeedScalar r_U[BASIS_NUM_COMP * BASIS_DIM];
  CeedScalar r_C[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];

  // load interp_1d into shared memory
  __shared__ CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  LoadMatrix<BASIS_P_1D, BASIS_Q_1D>(data, c_B, s_B);
  __syncthreads();

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    // Clear register
    for (CeedInt i = 0; i < BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1); i++) r_C[i] = 0.0;

    // Clear output vector
    for (CeedInt i = 0; i < BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1); i++) r_V[i] = 0.0;
    if (BASIS_DIM == 1) {
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                        BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    }

    // Map from points
    const CeedInt point_loop_bound = (blockDim.x * blockDim.y) * ceil(1.0 * BASIS_NUM_PTS / (blockDim.x * blockDim.y));

    for (CeedInt i = threadIdx.x + threadIdx.y * blockDim.x; i < point_loop_bound; i += blockDim.x * blockDim.y) {
      const CeedInt p = i % BASIS_NUM_PTS;

      ReadPoint<BASIS_DIM, BASIS_NUM_PTS>(data, elem, p, BASIS_NUM_PTS, 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, d_X, r_X);
      ReadPoint<BASIS_NUM_COMP * BASIS_DIM, BASIS_NUM_PTS>(data, elem, i, points_per_elem[elem], 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, d_U,
                                                           r_U);
      if (BASIS_DIM == 1) {
        GradTransposeAtPoints1d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      } else if (BASIS_DIM == 2) {
        GradTransposeAtPoints2d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      } else if (BASIS_DIM == 3) {
        GradTransposeAtPoints3d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      }
    }

    // Map from coefficients
    if (BASIS_DIM == 1) {
      InterpTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      InterpTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      InterpTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                      BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    }
  }
}

extern "C" __global__ void GradTransposeAddAtPoints(const CeedInt num_elem, const CeedScalarBase *__restrict__ c_B,
                                                    const CeedInt *__restrict__ points_per_elem, const CeedScalarBase *__restrict__ d_X,
                                                    const CeedScalarBase *__restrict__ d_U, CeedScalarBase *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];

  SharedData_Cuda data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * BASIS_T_1D * (BASIS_DIM > 1 ? BASIS_T_1D : 1);

  CeedScalar r_X[BASIS_DIM];
  CeedScalar r_U[BASIS_NUM_COMP * BASIS_DIM];
  CeedScalar r_C[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];

  // load interp_1d into shared memory
  __shared__ CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  LoadMatrix<BASIS_P_1D, BASIS_Q_1D>(data, c_B, s_B);
  __syncthreads();

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    // Clear register
    for (CeedInt i = 0; i < BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1); i++) r_C[i] = 0.0;

    // Map from points
    const CeedInt point_loop_bound = (blockDim.x * blockDim.y) * ceil(1.0 * BASIS_NUM_PTS / (blockDim.x * blockDim.y));

    for (CeedInt i = threadIdx.x + threadIdx.y * blockDim.x; i < point_loop_bound; i += blockDim.x * blockDim.y) {
      const CeedInt p = i % BASIS_NUM_PTS;

      ReadPoint<BASIS_DIM, BASIS_NUM_PTS>(data, elem, p, BASIS_NUM_PTS, 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, d_X, r_X);
      ReadPoint<BASIS_NUM_COMP * BASIS_DIM, BASIS_NUM_PTS>(data, elem, i, points_per_elem[elem], 1, num_elem * BASIS_NUM_PTS, BASIS_NUM_PTS, d_U,
                                                           r_U);
      if (BASIS_DIM == 1) {
        GradTransposeAtPoints1d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      } else if (BASIS_DIM == 2) {
        GradTransposeAtPoints2d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      } else if (BASIS_DIM == 3) {
        GradTransposeAtPoints3d<BASIS_NUM_COMP, BASIS_NUM_PTS, BASIS_P_1D, BASIS_Q_1D>(data, i, r_U, r_X, r_C);
      }
    }

    // Map from coefficients
    if (BASIS_DIM == 1) {
      InterpTranspose1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      InterpTransposeTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      InterpTransposeTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D, BASIS_T_1D>(data, r_C, s_B, r_V);
      SumElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                      BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, r_V, d_V);
    }
  }
}
