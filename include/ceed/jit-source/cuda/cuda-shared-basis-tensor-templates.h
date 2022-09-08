// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

typedef struct {
  CeedInt t_id_x;
  CeedInt t_id_y;
  CeedInt t_id_z;
  CeedInt t_id;
  CeedScalar* slice;
} BackendData;

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void ReadElementStrided1d(BackendData &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = d_u[ind + comp * strides_com];
    }
  }
}

//------------------------------------------------------------------------------
// Single element -> E-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void WriteElementStrided1d(BackendData &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] += r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// 1D tensor contraction x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractX1d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.t_id_x * P_1D] * data.slice[i]; // Contract x direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 1D transpose tensor contraction x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeX1d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_x + i * P_1D] * data.slice[i]; // Contract x direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void ReadElementStrided2d(BackendData &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y*P_1D;
    const CeedInt ind = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = d_u[ind + comp * strides_comp];
    }
  }
}

//------------------------------------------------------------------------------
// Single element -> E-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void WriteElementStrided2d(BackendData &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y*P_1D;
    const CeedInt ind = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] += r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// 2D tensor contraction x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractX2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x+data.t_id_y*T1d] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.t_id_x*P_1D] * data.slice[i + data.t_id_y*T1d]; // Contract x direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractY2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x+data.t_id_y*T1d] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.t_id_y*P_1D] * data.slice[data.t_id_x + i*T1d]; // Contract y direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeY2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x+data.t_id_y*T1d] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_y + i*P_1D] * data.slice[data.t_id_x + i*T1d]; // Contract y direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeX2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x+data.t_id_y*T1d] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_x + i*P_1D] * data.slice[i + data.t_id_y*T1d]; // Contract x direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract and add x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeAddX2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x+data.t_id_y*T1d] = *U;
  __syncthreads();
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_x + i*P_1D] * data.slice[i + data.t_id_y*T1d]; // Contract x direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

// TODO: remove "Dofs" and "Quads" in the following function names?
//   - ReadDofsOffset3d -> readOffset3d ?
//   - ReadDofsStrided3d -> readStrided3d ?
//   - ReadSliceQuadsOffset3d -> ReadSliceOffset3d ?
//   - ReadSliceQuadsStrided3d -> ReadSliceStrided3d ?
//   - WriteDofsOffset3d -> writeOffset3d ?
//   - WriteDofsStrided3d -> writeStrided3d ?

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void ReadElementStrided3d(BackendData &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt z = 0; z < P_1D; ++z) {
      const CeedInt node = data.t_id_x + data.t_id_y*P_1D + z*P_1D*P_1D;
      const CeedInt ind = node * strides_node + elem * strides_elem;
      for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
        r_u[z + comp * P_1D] = d_u[ind + comp * strides_comp];
      }
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> single element plane
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_1D>
inline __device__ void ReadElementSliceStrided3d(BackendData &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp, const CeedInt strides_elem, const CeedInt q, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y*Q_1D + q*Q_1D*Q_1D;
    const CeedInt ind = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = d_u[ind + comp * strides_comp];
    }
  }
}

//------------------------------------------------------------------------------
// Single element -> E-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void WriteElementStrided3d(BackendData &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt z = 0; z < P_1D; ++z) {
      const CeedInt node = data.t_id_x + data.t_id_y*P_1D + z*P_1D*P_1D;
      const CeedInt ind = node * strides_node + elem * strides_elem;
      for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
        d_v[ind + comp * strides_comp] += r_v[z + comp * P_1D];
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractX3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[P_1D];
  for (CeedInt i = 0; i < P_1D; i++) {
    r_B[i] = B[i + data.t_id_x*P_1D];
  }

  for (CeedInt k = 0; k < P_1D; ++k) {
    data.slice[data.t_id_x+data.t_id_y*T1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += r_B[i] * data.slice[i + data.t_id_y*T1d]; // Contract x direction
      }
    }
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractY3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[P_1D];
  for (CeedInt i = 0; i < P_1D; i++) {
    r_B[i] = B[i + data.t_id_y*P_1D];
  }

  for (CeedInt k = 0; k < P_1D; ++k) {
    data.slice[data.t_id_x+data.t_id_y*T1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += r_B[i] * data.slice[data.t_id_x + i*T1d]; // Contract y direction
      }
    }
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract z
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractZ3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < Q_1D; ++k) {
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += B[i + k*P_1D] * U[i]; // Contract z direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract z
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeZ3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  for (CeedInt k = 0; k < P_1D; ++k) {
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += B[k + i*P_1D] * U[i]; // Contract z direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeY3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.t_id_y + i*P_1D];
  }

  for (CeedInt k = 0; k < P_1D; ++k) {
    data.slice[data.t_id_x+data.t_id_y*T1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.slice[data.t_id_x + i*T1d]; // Contract y direction
      }
    }
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract add y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeAddY3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.t_id_y + i*P_1D];
  }

  for (CeedInt k = 0; k < P_1D; ++k) {
    data.slice[data.t_id_x+data.t_id_y*T1d] = U[k];
    __syncthreads();
    if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.slice[data.t_id_x + i*T1d]; // Contract y direction
      }
    }
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeX3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.t_id_x + i*P_1D];
  }

  for (CeedInt k = 0; k < P_1D; ++k) {
    data.slice[data.t_id_x+data.t_id_y*T1d] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.slice[i + data.t_id_y*T1d]; // Contract x direction
      }
    }
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract add x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeAddX3d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  CeedScalar r_B[Q_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[data.t_id_x + i*P_1D];
  }

  for (CeedInt k = 0; k < P_1D; ++k) {
    data.slice[data.t_id_x+data.t_id_y*T1d] = U[k];
    __syncthreads();
    if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.slice[i + data.t_id_y*T1d]; // Contract x direction
      }
    }
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
