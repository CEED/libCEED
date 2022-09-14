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
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void ReadElementStrided1d(BackendData &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
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
inline __device__ void WriteElementStrided1d(BackendData &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] = r_v[comp];
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
// 1D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void Interp1d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX1d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, c_B, r_V + comp);
  }
}

//------------------------------------------------------------------------------
// 1D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void InterpTranspose1d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeX1d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, c_B, r_V + comp);
  }
}

//------------------------------------------------------------------------------
// 1D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void Grad1d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX1d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, c_G, r_V + comp);
  }
}

//------------------------------------------------------------------------------
// 1D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void GradTranspose1d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeX1d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, c_G, r_V + comp);
  }
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
      d_v[ind + comp * strides_comp] = r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// 2D tensor contraction x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractX2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x+data.t_id_y*T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.t_id_x*P_1D] * data.slice[i + data.t_id_y*T_1D]; // Contract x direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractY2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x+data.t_id_y*T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + data.t_id_y*P_1D] * data.slice[data.t_id_x + i*T_1D]; // Contract y direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract y
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeY2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x+data.t_id_y*T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_y + i*P_1D] * data.slice[data.t_id_x + i*T_1D]; // Contract y direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeX2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x+data.t_id_y*T_1D] = *U;
  __syncthreads();
  *V = 0.0;
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_x + i*P_1D] * data.slice[i + data.t_id_y*T_1D]; // Contract x direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract and add x
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void ContractTransposeAddX2d(BackendData &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
  data.slice[data.t_id_x+data.t_id_y*T_1D] = *U;
  __syncthreads();
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[data.t_id_x + i*P_1D] * data.slice[i + data.t_id_y*T_1D]; // Contract x direction
    }
  }
  __syncthreads();
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void InterpTensor2d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, c_B, r_t);
    ContractY2d<NUM_COMP, P_1D, Q_1D>(data, r_t, c_B, r_V + comp);
  }
}

//------------------------------------------------------------------------------
// 2D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void InterpTransposeTensor2d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeY2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, c_B, r_t);
    ContractTransposeX2d<NUM_COMP, P_1D, Q_1D>(data, r_t, c_B, r_V + comp);
  }
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void GradTensor2d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, c_G, r_t);
    ContractY2d<NUM_COMP, P_1D, Q_1D>(data, r_t, c_B, r_V + comp + 0*NUM_COMP);
    ContractX2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, c_B, r_t);
    ContractY2d<NUM_COMP, P_1D, Q_1D>(data, r_t, c_G, r_V + comp + 1*NUM_COMP);
  }
}

//------------------------------------------------------------------------------
// 2D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void GradTransposeTensor2d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeY2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp + 0*NUM_COMP, c_B, r_t);
    ContractTransposeX2d<NUM_COMP, P_1D, Q_1D>(data, r_t, c_G, r_V + comp);
    ContractTransposeY2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp + 1*NUM_COMP, c_G, r_t);
    ContractTransposeAddX2d<NUM_COMP, P_1D, Q_1D>(data, r_t, c_B, r_V + comp);
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void ReadElementStrided3d(BackendData &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt z = 0; z < P_1D; z++) {
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
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y*P_1D + z*P_1D*P_1D;
      const CeedInt ind = node * strides_node + elem * strides_elem;
      for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
        d_v[ind + comp * strides_comp] = r_v[z + comp * P_1D];
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

  for (CeedInt k = 0; k < P_1D; k++) {
    data.slice[data.t_id_x+data.t_id_y*T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += r_B[i] * data.slice[i + data.t_id_y*T_1D]; // Contract x direction
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

  for (CeedInt k = 0; k < P_1D; k++) {
    data.slice[data.t_id_x+data.t_id_y*T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += r_B[i] * data.slice[data.t_id_x + i*T_1D]; // Contract y direction
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
  for (CeedInt k = 0; k < Q_1D; k++) {
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
  for (CeedInt k = 0; k < P_1D; k++) {
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

  for (CeedInt k = 0; k < P_1D; k++) {
    data.slice[data.t_id_x+data.t_id_y*T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < Q_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.slice[data.t_id_x + i*T_1D]; // Contract y direction
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

  for (CeedInt k = 0; k < P_1D; k++) {
    data.slice[data.t_id_x+data.t_id_y*T_1D] = U[k];
    __syncthreads();
    V[k] = 0.0;
    if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.slice[i + data.t_id_y*T_1D]; // Contract x direction
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

  for (CeedInt k = 0; k < P_1D; k++) {
    data.slice[data.t_id_x+data.t_id_y*T_1D] = U[k];
    __syncthreads();
    if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * data.slice[i + data.t_id_y*T_1D]; // Contract x direction
      }
    }
    __syncthreads();
  }
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void InterpTensor3d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp*P_1D, c_B, r_t1);
    ContractY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, c_B, r_t2);
    ContractZ3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, c_B, r_V + comp*Q_1D);
  }
}

//------------------------------------------------------------------------------
// 3D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void InterpTransposeTensor3d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp*Q_1D, c_B, r_t1);
    ContractTransposeY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, c_B, r_t2);
    ContractTransposeX3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, c_B, r_V + comp*P_1D);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void GradTensor3d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp*P_1D, c_G, r_t1);
    ContractY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, c_B, r_t2);
    ContractZ3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, c_B, r_V + comp*Q_1D + 0*NUM_COMP*Q_1D);
    ContractX3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp*P_1D, c_B, r_t1);
    ContractY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, c_G, r_t2);
    ContractZ3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, c_B, r_V + comp*Q_1D + 1*NUM_COMP*Q_1D);
    ContractX3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp*P_1D, c_B, r_t1);
    ContractY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, c_B, r_t2);
    ContractZ3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, c_G, r_V + comp*Q_1D + 2*NUM_COMP*Q_1D);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void GradTransposeTensor3d(BackendData &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp*Q_1D + 0*NUM_COMP*Q_1D, c_B, r_t1);
    ContractTransposeY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, c_B, r_t2);
    ContractTransposeX3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, c_G, r_V + comp*P_1D);
    ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp*Q_1D + 1*NUM_COMP*Q_1D, c_B, r_t1);
    ContractTransposeY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, c_G, r_t2);
    ContractTransposeAddX3d<NUM_COMP,P_1D, Q_1D>(data, r_t2, c_B, r_V + comp*P_1D);
    ContractTransposeZ3d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp*Q_1D + 2*NUM_COMP*Q_1D, c_G, r_t1);
    ContractTransposeY3d<NUM_COMP, P_1D, Q_1D>(data, r_t1, c_B, r_t2);
    ContractTransposeAddX3d<NUM_COMP, P_1D, Q_1D>(data, r_t2, c_B, r_V + comp*P_1D);
  }
}

//------------------------------------------------------------------------------
// 1D quadrature weights
//------------------------------------------------------------------------------
template <int Q_1D>
inline __device__ void Weight1d(BackendData &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
  *w = (data.t_id_x < Q_1D) ? q_weight_1d[data.t_id_x] : 0.0;
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
template <int Q_1D>
inline __device__ void Weight2d(BackendData &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
  *w = (data.t_id_x < Q_1D && data.t_id_y < Q_1D) ?
        q_weight_1d[data.t_id_x]*q_weight_1d[data.t_id_y] : 0.0;
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
template <int Q_1D>
inline __device__ void Weight3d(BackendData &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
  const bool quad = (data.t_id_x < Q_1D && data.t_id_y < Q_1D);
  const CeedScalar pw = quad ? q_weight_1d[data.t_id_x]*q_weight_1d[data.t_id_y] : 0.0;
  for (CeedInt q = 0; q < Q_1D; q++) {
    w[q] = quad ? pw*q_weight_1d[q] : 0.0;
  }
}

//------------------------------------------------------------------------------
