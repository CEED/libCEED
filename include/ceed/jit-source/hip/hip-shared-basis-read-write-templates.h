// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for HIP shared memory basis read/write templates
#include <ceed/types.h>

//------------------------------------------------------------------------------
// Helper function: load matrices for basis actions
//------------------------------------------------------------------------------
template <int P, int Q>
inline __device__ void LoadMatrix(SharedData_Hip &data, const CeedScalar *__restrict__ d_B, CeedScalar *B) {
  for (CeedInt i = data.t_id; i < P * Q; i += blockDim.x * blockDim.y * blockDim.z) B[i] = d_B[i];
}

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void ReadElementStrided1d(SharedData_Hip &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
                                            const CeedInt strides_elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = node * strides_node + elem * strides_elem;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = d_u[ind + comp * strides_comp];
    }
  }
}

//------------------------------------------------------------------------------
// Single element -> E-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void WriteElementStrided1d(SharedData_Hip &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
                                             const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = node * strides_node + elem * strides_elem;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] = r_v[comp];
    }
  }
}

template <int NUM_COMP, int P_1D>
inline __device__ void SumElementStrided1d(SharedData_Hip &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
                                           const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = node * strides_node + elem * strides_elem;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] += r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void ReadElementStrided2d(SharedData_Hip &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
                                            const CeedInt strides_elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1D;
    const CeedInt ind  = node * strides_node + elem * strides_elem;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = d_u[ind + comp * strides_comp];
    }
  }
}

//------------------------------------------------------------------------------
// Single element -> E-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void WriteElementStrided2d(SharedData_Hip &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
                                             const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1D;
    const CeedInt ind  = node * strides_node + elem * strides_elem;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] = r_v[comp];
    }
  }
}

template <int NUM_COMP, int P_1D>
inline __device__ void SumElementStrided2d(SharedData_Hip &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
                                           const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1D;
    const CeedInt ind  = node * strides_node + elem * strides_elem;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] += r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void ReadElementStrided3d(SharedData_Hip &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
                                            const CeedInt strides_elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = node * strides_node + elem * strides_elem;

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
        r_u[z + comp * P_1D] = d_u[ind + comp * strides_comp];
      }
    }
  }
}

//------------------------------------------------------------------------------
// Single element -> E-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void WriteElementStrided3d(SharedData_Hip &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
                                             const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = node * strides_node + elem * strides_elem;

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
        d_v[ind + comp * strides_comp] = r_v[z + comp * P_1D];
      }
    }
  }
}

template <int NUM_COMP, int P_1D>
inline __device__ void SumElementStrided3d(SharedData_Hip &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
                                           const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = data.t_id_x + data.t_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = node * strides_node + elem * strides_elem;

      for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
        d_v[ind + comp * strides_comp] += r_v[z + comp * P_1D];
      }
    }
  }
}

//------------------------------------------------------------------------------
// AtPoints
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single point
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_PTS>
inline __device__ void ReadPoint(SharedData_Hip &data, const CeedInt elem, const CeedInt p, const CeedInt points_in_elem, const CeedInt strides_point,
                                 const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *__restrict__ d_u, CeedScalar *r_u) {
  const CeedInt ind = (p % NUM_PTS) * strides_point + elem * strides_elem;

  if (p < points_in_elem) {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = d_u[ind + comp * strides_comp];
    }
  } else {
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = 0.0;
    }
  }
}

//------------------------------------------------------------------------------
// Single point -> E-vector
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_PTS>
inline __device__ void WritePoint(SharedData_Hip &data, const CeedInt elem, const CeedInt p, const CeedInt points_in_elem,
                                  const CeedInt strides_point, const CeedInt strides_comp, const CeedInt strides_elem, const CeedScalar *r_v,
                                  CeedScalar *d_v) {
  if (p < points_in_elem) {
    const CeedInt ind = (p % NUM_PTS) * strides_point + elem * strides_elem;

    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] = r_v[comp];
    }
  }
}
