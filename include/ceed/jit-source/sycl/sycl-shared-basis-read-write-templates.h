// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL shared memory basis read/write templates

#include <ceed.h>
#include "sycl-types.h"

//------------------------------------------------------------------------------
// Helper function: load matrices for basis actions
//------------------------------------------------------------------------------
inline void loadMatrix(const CeedInt N, const CeedScalar *restrict d_B, CeedScalar *restrict B) {
  const CeedInt item_id    = get_local_linear_id();
  const CeedInt group_size = get_local_size(0) * get_local_size(1) * get_local_size(2);
  for (CeedInt i = item_id; i < N; i += group_size) B[i] = d_B[i];
}

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// E-vector -> single element
//------------------------------------------------------------------------------
inline void ReadElementStrided1d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt num_elem, const CeedInt strides_node,
                                 const CeedInt strides_comp, const CeedInt strides_elem, global const CeedScalar *restrict d_u,
                                 private CeedScalar *restrict r_u) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x;
    const CeedInt ind  = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = d_u[ind + comp * strides_comp];
    }
  }
}

//------------------------------------------------------------------------------
// Single element -> E-vector
//------------------------------------------------------------------------------
inline void WriteElementStrided1d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt num_elem, const CeedInt strides_node,
                                  const CeedInt strides_comp, const CeedInt strides_elem, private const CeedScalar *restrict r_v,
                                  global CeedScalar *restrict d_v) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x;
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
inline void ReadElementStrided2d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt num_elem, const CeedInt strides_node,
                                 const CeedInt strides_comp, const CeedInt strides_elem, global const CeedScalar *restrict d_u,
                                 private CeedScalar *restrict r_u) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x + item_id_y * P_1D;
    const CeedInt ind  = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
      r_u[comp] = d_u[ind + comp * strides_comp];
    }
  }
}

//------------------------------------------------------------------------------
// Single element -> E-vector
//------------------------------------------------------------------------------
inline void WriteElementStrided2d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt num_elem, const CeedInt strides_node,
                                  const CeedInt strides_comp, const CeedInt strides_elem, private const CeedScalar *restrict r_v,
                                  global CeedScalar *restrict d_v) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x + item_id_y * P_1D;
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
inline void ReadElementStrided3d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt num_elem, const CeedInt strides_node,
                                 const CeedInt strides_comp, const CeedInt strides_elem, global const CeedScalar *restrict d_u,
                                 private CeedScalar *restrict r_u) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = item_id_x + item_id_y * P_1D + z * P_1D * P_1D;
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
inline void WriteElementStrided3d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt num_elem, const CeedInt strides_node,
                                  const CeedInt strides_comp, const CeedInt strides_elem, private const CeedScalar *restrict r_v,
                                  global CeedScalar *restrict d_v) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    for (CeedInt z = 0; z < P_1D; z++) {
      const CeedInt node = item_id_x + item_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = node * strides_node + elem * strides_elem;
      for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
        d_v[ind + comp * strides_comp] = r_v[z + comp * P_1D];
      }
    }
  }
}
