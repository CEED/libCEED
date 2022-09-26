// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA shared memory basis read/write templates
#ifndef _ceed_cuda_shared_basis_read_write_templates_h
#define _ceed_cuda_shared_basis_read_write_templates_h

#include <ceed.h>

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D>
inline __device__ void ReadElementStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
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
inline __device__ void WriteElementStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
                                             const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D) {
    const CeedInt node = data.t_id_x;
    const CeedInt ind  = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] = r_v[comp];
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
inline __device__ void ReadElementStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
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
inline __device__ void WriteElementStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
                                             const CeedInt strides_elem, const CeedScalar *r_v, CeedScalar *d_v) {
  if (data.t_id_x < P_1D && data.t_id_y < P_1D) {
    const CeedInt node = data.t_id_x + data.t_id_y * P_1D;
    const CeedInt ind  = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      d_v[ind + comp * strides_comp] = r_v[comp];
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
inline __device__ void ReadElementStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
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
inline __device__ void WriteElementStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
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

//------------------------------------------------------------------------------

#endif
