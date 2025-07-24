// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA backend macro and type definitions for JiT source
#include <ceed/types.h>

//------------------------------------------------------------------------------
// Load matrices for basis actions
//------------------------------------------------------------------------------
template <int P, int Q, class ScalarIn, class ScalarOut>
inline __device__ void LoadMatrix(SharedData_Cuda &data, const ScalarIn *__restrict__ d_B, ScalarOut *B) {
  for (CeedInt i = data.t_id; i < P * Q; i += blockDim.x * blockDim.y * blockDim.z) B[i] = d_B[i];
}

//------------------------------------------------------------------------------
// AtPoints
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> single point
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int NUM_PTS, class ScalarIn, class ScalarOut>
inline __device__ void ReadPoint(SharedData_Cuda &data, const CeedInt elem, const CeedInt p, const CeedInt points_in_elem,
                                 const CeedInt *__restrict__ indices, const ScalarIn *__restrict__ d_u, ScalarOut *r_u) {
  const CeedInt ind = indices[p + elem * NUM_PTS];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    r_u[comp] = d_u[ind + comp * COMP_STRIDE];
  }
}

//------------------------------------------------------------------------------
// Single point -> L-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int NUM_PTS, class ScalarIn, class ScalarOut>
inline __device__ void WritePoint(SharedData_Cuda &data, const CeedInt elem, const CeedInt p, const CeedInt points_in_elem,
                                  const CeedInt *__restrict__ indices, const ScalarIn *__restrict__ r_u, ScalarOut *d_u) {
  if (p < points_in_elem) {
    const CeedInt ind = indices[p + elem * NUM_PTS];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_u[ind + comp * COMP_STRIDE] += r_u[comp];
    }
  }
}

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Set E-vector value
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void SetEVecStandard1d_Single(SharedData_Cuda &data, const CeedInt n, const ScalarIn value, ScalarOut *__restrict__ r_v) {
  const CeedInt target_comp = n / P_1D;
  const CeedInt target_node = n % P_1D;

  if (data.t_id_x == target_node) {
    r_v[target_comp] = value;
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void ReadLVecStandard1d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                          const ScalarIn *__restrict__ d_u, ScalarOut *__restrict__ r_u) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = indices[node + elem * P_1D];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + COMP_STRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM, class ScalarIn, class ScalarOut>
inline __device__ void ReadLVecStrided1d(SharedData_Cuda &data, const CeedInt elem, const ScalarIn *__restrict__ d_u, ScalarOut *__restrict__ r_u) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard1d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                           const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = indices[node + elem * P_1D];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) atomicAdd(&d_v[ind + COMP_STRIDE * comp], r_v[comp]);
  }
}

template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard1d_Single(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt n,
                                                  const CeedInt *__restrict__ indices, const ScalarIn *__restrict__ r_v,
                                                  ScalarOut *__restrict__ d_v) {
  const CeedInt target_comp = n / P_1D;
  const CeedInt target_node = n % P_1D;

  if (data.t_id_x == target_node) {
    const CeedInt ind = indices[target_node + elem * P_1D];

    atomicAdd(&d_v[ind + COMP_STRIDE * target_comp], r_v[target_comp]);
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, full assembly
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard1d_Assembly(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt in,
                                                    const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  const CeedInt in_comp    = in / P_1D;
  const CeedInt in_node    = in % P_1D;
  const CeedInt e_vec_size = P_1D * NUM_COMP;

  if (data.t_id_x < P_1D) {
    const CeedInt out_node = data.t_id_x;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[elem * e_vec_size * e_vec_size + (in_comp * NUM_COMP + comp) * P_1D * P_1D + out_node * P_1D + in_node] += r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, Qfunction assembly
//------------------------------------------------------------------------------
template <int NUM_COMP_OUT, int NUM_COMP_FIELD, int Q_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard1d_QFAssembly(SharedData_Cuda &data, const CeedInt num_elem, const CeedInt elem, const CeedInt input_offset,
                                                      const CeedInt output_offset, const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  if (data.t_id_x < Q_1D) {
    const CeedInt ind = data.t_id_x + elem * Q_1D;

    for (CeedInt comp = 0; comp < NUM_COMP_FIELD; comp++) {
      d_v[ind + (input_offset * NUM_COMP_OUT + output_offset + comp) * (Q_1D * num_elem)] = r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStrided1d(SharedData_Cuda &data, const CeedInt elem, const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) d_v[ind + comp * STRIDES_COMP] += r_v[comp];
  }
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Set E-vector value
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void SetEVecStandard2d_Single(SharedData_Cuda &data, const CeedInt n, const ScalarIn value, ScalarOut *__restrict__ r_v) {
  const CeedInt target_comp   = n / (P_1D * P_1D);
  const CeedInt target_node_x = n % P_1D;
  const CeedInt target_node_y = (n % (P_1D * P_1D)) / P_1D;

  if (data.t_id_x == target_node_x && data.t_id_y == target_node_y) {
    r_v[target_comp] = value;
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void ReadLVecStandard2d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                          const ScalarIn *__restrict__ d_u, ScalarOut *__restrict__ r_u) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1D;
    const CeedInt ind  = indices[node + elem * P_1D * P_1D];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + COMP_STRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM, class ScalarIn, class ScalarOut>
inline __device__ void ReadLVecStrided2d(SharedData_Cuda &data, const CeedInt elem, const ScalarIn *__restrict__ d_u, ScalarOut *__restrict__ r_u) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1D;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard2d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                           const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1D;
    const CeedInt ind  = indices[node + elem * P_1D * P_1D];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) atomicAdd(&d_v[ind + COMP_STRIDE * comp], r_v[comp]);
  }
}

template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard2d_Single(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt n,
                                                  const CeedInt *__restrict__ indices, const ScalarIn *__restrict__ r_v,
                                                  ScalarOut *__restrict__ d_v) {
  const CeedInt target_comp   = n / (P_1D * P_1D);
  const CeedInt target_node_x = n % P_1D;
  const CeedInt target_node_y = (n % (P_1D * P_1D)) / P_1D;

  if (data.t_id_x == target_node_x && data.t_id_y == target_node_y) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1D;
    const CeedInt ind  = indices[node + elem * P_1D * P_1D];

    atomicAdd(&d_v[ind + COMP_STRIDE * target_comp], r_v[target_comp]);
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, full assembly
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard2d_Assembly(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt in,
                                                    const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  const CeedInt elem_size  = P_1D * P_1D;
  const CeedInt in_comp    = in / elem_size;
  const CeedInt in_node_x  = in % P_1D;
  const CeedInt in_node_y  = (in % elem_size) / P_1D;
  const CeedInt e_vec_size = elem_size * NUM_COMP;

  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt in_node  = in_node_x + in_node_y * P_1D;
    const CeedInt out_node = data.t_id_x + data.t_id_y * P_1D;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      const CeedInt index = (in_comp * NUM_COMP + comp) * elem_size * elem_size + out_node * elem_size + in_node;

      d_v[elem * e_vec_size * e_vec_size + index] += r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, Qfunction assembly
//------------------------------------------------------------------------------
template <int NUM_COMP_OUT, int NUM_COMP_FIELD, int Q_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard2d_QFAssembly(SharedData_Cuda &data, const CeedInt num_elem, const CeedInt elem, const CeedInt input_offset,
                                                      const CeedInt output_offset, const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
    const CeedInt ind = (data.t_id_x + data.t_id_y * Q_1D) + elem * Q_1D * Q_1D;

    for (CeedInt comp = 0; comp < NUM_COMP_FIELD; comp++) {
      d_v[ind + (input_offset * NUM_COMP_OUT + output_offset + comp) * (Q_1D * Q_1D * num_elem)] = r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStrided2d(SharedData_Cuda &data, const CeedInt elem, const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1D;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) d_v[ind + comp * STRIDES_COMP] += r_v[comp];
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Set E-vector value
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void SetEVecStandard3d_Single(SharedData_Cuda &data, const CeedInt n, const ScalarIn value, ScalarOut *__restrict__ r_v) {
  const CeedInt target_comp   = n / (P_1D * P_1D * P_1D);
  const CeedInt target_node_x = n % P_1D;
  const CeedInt target_node_y = (n % (P_1D * P_1D)) / P_1D;
  const CeedInt target_node_z = (n % (P_1D * P_1D * P_1D)) / (P_1D * P_1D);

  if (data.t_id_x == target_node_x && data.t_id_y == target_node_y) {
    r_v[target_node_z + target_comp * P_1D] = value;
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void ReadLVecStandard3d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                          const ScalarIn *__restrict__ d_u, ScalarOut *__restrict__ r_u) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = indices[node + elem * P_1D * P_1D * P_1D];

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[z + comp * P_1D] = d_u[ind + COMP_STRIDE * comp];
    }
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM, class ScalarIn, class ScalarOut>
inline __device__ void ReadLVecStrided3d(SharedData_Cuda &data, const CeedInt elem, const ScalarIn *__restrict__ d_u, ScalarOut *__restrict__ r_u) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[z + comp * P_1D] = d_u[ind + comp * STRIDES_COMP];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, offests provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int Q_1D, class ScalarIn, class ScalarOut>
inline __device__ void ReadEVecSliceStandard3d(SharedData_Cuda &data, const CeedInt nquads, const CeedInt elem, const CeedInt q,
                                               const CeedInt *__restrict__ indices, const ScalarIn *__restrict__ d_u, ScalarOut *__restrict__ r_u) {
  if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y * Q_1D + q * Q_1D * Q_1D;
    const CeedInt ind  = indices[node + elem * Q_1D * Q_1D * Q_1D];

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + COMP_STRIDE * comp];
  }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM, class ScalarIn, class ScalarOut>
inline __device__ void ReadEVecSliceStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt q, const ScalarIn *__restrict__ d_u,
                                              ScalarOut *__restrict__ r_u) {
  if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y * Q_1D + q * Q_1D * Q_1D;
    const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) r_u[comp] = d_u[ind + comp * STRIDES_COMP];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard3d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
                                           const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = indices[node + elem * P_1D * P_1D * P_1D];

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) atomicAdd(&d_v[ind + COMP_STRIDE * comp], r_v[z + comp * P_1D]);
    }
  }
}

template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard3d_Single(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt n,
                                                  const CeedInt *__restrict__ indices, const ScalarIn *__restrict__ r_v,
                                                  ScalarOut *__restrict__ d_v) {
  const CeedInt target_comp   = n / (P_1D * P_1D * P_1D);
  const CeedInt target_node_x = n % P_1D;
  const CeedInt target_node_y = (n % (P_1D * P_1D)) / P_1D;
  const CeedInt target_node_z = (n % (P_1D * P_1D * P_1D)) / (P_1D * P_1D);

  if (data.t_id_x == target_node_x && data.t_id_y == target_node_y) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1D + target_node_z * P_1D * P_1D;
    const CeedInt ind  = indices[node + elem * P_1D * P_1D * P_1D];

    atomicAdd(&d_v[ind + COMP_STRIDE * target_comp], r_v[target_node_z + target_comp * P_1D]);
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, full assembly
//------------------------------------------------------------------------------
template <int NUM_COMP, int COMP_STRIDE, int P_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard3d_Assembly(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt in,
                                                    const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  const CeedInt elem_size  = P_1D * P_1D * P_1D;
  const CeedInt in_comp    = in / elem_size;
  const CeedInt in_node_x  = in % P_1D;
  const CeedInt in_node_y  = (in % (P_1D * P_1D)) / P_1D;
  const CeedInt in_node_z  = (in % elem_size) / (P_1D * P_1D);
  const CeedInt e_vec_size = elem_size * NUM_COMP;

  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt in_node = in_node_x + in_node_y * P_1D + in_node_z * P_1D * P_1D;
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt out_node = data.t_id_x + data.t_id_y * P_1D + z * P_1D * P_1D;

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
        const CeedInt index = (in_comp * NUM_COMP + comp) * elem_size * elem_size + out_node * elem_size + in_node;

        d_v[elem * e_vec_size * e_vec_size + index] += r_v[z + comp * P_1D];
      }
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, Qfunction assembly
//------------------------------------------------------------------------------
template <int NUM_COMP_OUT, int NUM_COMP_FIELD, int Q_1D, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStandard3d_QFAssembly(SharedData_Cuda &data, const CeedInt num_elem, const CeedInt elem, const CeedInt input_offset,
                                                      const CeedInt output_offset, const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
    for (CeedInt z = 0; z < Q_1D; z++) {
      const CeedInt ind = (data.t_id_x + data.t_id_y * Q_1D + z * Q_1D * Q_1D) + elem * Q_1D * Q_1D * Q_1D;

      for (CeedInt comp = 0; comp < NUM_COMP_FIELD; comp++) {
        d_v[ind + (input_offset * NUM_COMP_OUT + output_offset + comp) * (Q_1D * Q_1D * Q_1D * num_elem)] = r_v[z + comp * Q_1D];
      }
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int STRIDES_NODE, int STRIDES_COMP, int STRIDES_ELEM, class ScalarIn, class ScalarOut>
inline __device__ void WriteLVecStrided3d(SharedData_Cuda &data, const CeedInt elem, const ScalarIn *__restrict__ r_v, ScalarOut *__restrict__ d_v) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = node * STRIDES_NODE + elem * STRIDES_ELEM;

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) d_v[ind + comp * STRIDES_COMP] += r_v[z + comp * P_1D];
    }
  }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives computation
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void GradColloSlice3d(SharedData_Cuda &data, const CeedInt q, const ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_G,
                                        ScalarOut *__restrict__ r_V) {
  if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      __syncthreads();
      data.slice[data.t_id_x + data.t_id_y * T_1D] = r_U[q + comp * Q_1D];
      __syncthreads();
      // X derivative
      r_V[comp + 0 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[comp + 0 * NUM_COMP] += c_G[i + data.t_id_x * Q_1D] * data.slice[i + data.t_id_y * T_1D];
      }
      // Y derivative
      r_V[comp + 1 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[comp + 1 * NUM_COMP] += c_G[i + data.t_id_y * Q_1D] * data.slice[data.t_id_x + i * T_1D];
      }
      // Z derivative
      r_V[comp + 2 * NUM_COMP] = 0.0;
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[comp + 2 * NUM_COMP] += c_G[i + q * Q_1D] * r_U[i + comp * Q_1D];
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int Q_1D, int T_1D, class ScalarIn1, class ScalarIn2, class ScalarOut>
inline __device__ void GradColloSliceTranspose3d(SharedData_Cuda &data, const CeedInt q, const ScalarIn1 *__restrict__ r_U, const ScalarIn2 *c_G,
                                                 ScalarOut *__restrict__ r_V) {
  if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      __syncthreads();
      data.slice[data.t_id_x + data.t_id_y * T_1D] = r_U[comp + 0 * NUM_COMP];
      __syncthreads();
      // X derivative
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[q + comp * Q_1D] += c_G[data.t_id_x + i * Q_1D] * data.slice[i + data.t_id_y * T_1D];
      }
      // Y derivative
      __syncthreads();
      data.slice[data.t_id_x + data.t_id_y * T_1D] = r_U[comp + 1 * NUM_COMP];
      __syncthreads();
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[q + comp * Q_1D] += c_G[data.t_id_y + i * Q_1D] * data.slice[data.t_id_x + i * T_1D];
      }
      // Z derivative
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[i + comp * Q_1D] += c_G[i + q * Q_1D] * r_U[comp + 2 * NUM_COMP];
      }
    }
  }
}
