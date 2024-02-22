// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL backend macro and type definitions for JiT source
#ifndef CEED_SYCL_GEN_TEMPLATES_H
#define CEED_SYCL_GEN_TEMPLATES_H

#include <ceed.h>

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
// TODO: Handle FP32 case
typedef atomic_double CeedAtomicScalar;

//------------------------------------------------------------------------------
// Load matrices for basis actions
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
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
inline void readDofsOffset1d(const CeedInt num_comp, const CeedInt strides_comp, const CeedInt P_1D, const CeedInt num_elem,
                             const global CeedInt *restrict indices, const global CeedScalar *restrict d_u, private CeedScalar *restrict r_u) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x;
    const CeedInt ind  = indices[node + elem * P_1D];
    for (CeedInt comp = 0; comp < num_comp; ++comp) {
      r_u[comp] = d_u[ind + strides_comp * comp];
    }
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
inline void readDofsStrided1d(const CeedInt num_comp, const CeedInt P_1D, const CeedInt strides_node, const CeedInt strides_comp,
                              const CeedInt strides_elem, const CeedInt num_elem, global const CeedScalar *restrict d_u,
                              private CeedScalar *restrict r_u) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x;
    const CeedInt ind  = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < num_comp; comp++) {
      r_u[comp] = d_u[ind + comp * strides_comp];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
inline void writeDofsOffset1d(const CeedInt num_comp, const CeedInt strides_comp, const CeedInt P_1D, const CeedInt num_elem,
                              const global CeedInt *restrict indices, const private CeedScalar *restrict r_v, global CeedAtomicScalar *restrict d_v) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x;
    const CeedInt ind  = indices[node + elem * P_1D];
    for (CeedInt comp = 0; comp < num_comp; ++comp)
      atomic_fetch_add_explicit(&d_v[ind + strides_comp * comp], r_v[comp], memory_order_relaxed, memory_scope_device);
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
inline void writeDofsStrided1d(const CeedInt num_comp, const CeedInt P_1D, const CeedInt strides_node, const CeedInt strides_comp,
                               const CeedInt strides_elem, const CeedInt num_elem, private const CeedScalar *restrict r_v,
                               global CeedScalar *restrict d_v) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x;
    const CeedInt ind  = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < num_comp; comp++) {
      d_v[ind + comp * strides_comp] = r_v[comp];
    }
  }
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
inline void readDofsOffset2d(const CeedInt num_comp, const CeedInt strides_comp, const CeedInt P_1D, const CeedInt num_elem,
                             const global CeedInt *restrict indices, const global CeedScalar *restrict d_u, private CeedScalar *restrict r_u) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x + item_id_y * P_1D;
    const CeedInt ind  = indices[node + elem * P_1D * P_1D];
    for (CeedInt comp = 0; comp < num_comp; ++comp) r_u[comp] = d_u[ind + strides_comp * comp];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
inline void readDofsStrided2d(const CeedInt num_comp, const CeedInt P_1D, const CeedInt strides_node, const CeedInt strides_comp,
                              const CeedInt strides_elem, const CeedInt num_elem, const global CeedScalar *restrict d_u,
                              private CeedScalar *restrict r_u) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x + item_id_y * P_1D;
    const CeedInt ind  = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < num_comp; ++comp) r_u[comp] = d_u[ind + comp * strides_comp];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
inline void writeDofsOffset2d(const CeedInt num_comp, const CeedInt strides_comp, const CeedInt P_1D, const CeedInt num_elem,
                              const global CeedInt *restrict indices, const private CeedScalar *restrict r_v, global CeedAtomicScalar *restrict d_v) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x + item_id_y * P_1D;
    const CeedInt ind  = indices[node + elem * P_1D * P_1D];
    for (CeedInt comp = 0; comp < num_comp; ++comp)
      atomic_fetch_add_explicit(&d_v[ind + strides_comp * comp], r_v[comp], memory_order_relaxed, memory_scope_device);
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
inline void writeDofsStrided2d(const CeedInt num_comp, const CeedInt P_1D, const CeedInt strides_node, const CeedInt strides_comp,
                               const CeedInt strides_elem, const CeedInt num_elem, const private CeedScalar *restrict r_v,
                               global CeedScalar *restrict d_v) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    const CeedInt node = item_id_x + item_id_y * P_1D;
    const CeedInt ind  = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < num_comp; ++comp) d_v[ind + comp * strides_comp] += r_v[comp];
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
inline void readDofsOffset3d(const CeedInt num_comp, const CeedInt strides_comp, const CeedInt P_1D, const CeedInt num_elem,
                             const global CeedInt *restrict indices, const global CeedScalar *restrict d_u, private CeedScalar *restrict r_u) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    for (CeedInt z = 0; z < P_1D; ++z) {
      const CeedInt node = item_id_x + P_1D * (item_id_y + P_1D * z);
      const CeedInt ind  = indices[node + elem * P_1D * P_1D * P_1D];
      for (CeedInt comp = 0; comp < num_comp; ++comp) r_u[z + comp * P_1D] = d_u[ind + strides_comp * comp];
    }
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
inline void readDofsStrided3d(const CeedInt num_comp, const CeedInt P_1D, const CeedInt strides_node, const CeedInt strides_comp,
                              const CeedInt strides_elem, const CeedInt num_elem, const global CeedScalar *restrict d_u,
                              private CeedScalar *restrict r_u) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    for (CeedInt z = 0; z < P_1D; ++z) {
      const CeedInt node = item_id_x + P_1D * (item_id_y + P_1D * z);
      const CeedInt ind  = node * strides_node + elem * strides_elem;
      for (CeedInt comp = 0; comp < num_comp; ++comp) r_u[z + comp * P_1D] = d_u[ind + comp * strides_comp];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, offests provided
//------------------------------------------------------------------------------
inline void readSliceQuadsOffset3d(const CeedInt num_comp, const CeedInt strides_comp, const CeedInt Q_1D, const CeedInt num_elem, const CeedInt q,
                                   const global CeedInt *restrict indices, const global CeedScalar *restrict d_u, private CeedScalar *restrict r_u) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < Q_1D && item_id_y < Q_1D && elem < num_elem) {
    const CeedInt node = item_id_x + Q_1D * (item_id_y + Q_1D * q);
    const CeedInt ind  = indices[node + elem * Q_1D * Q_1D * Q_1D];
    for (CeedInt comp = 0; comp < num_comp; ++comp) r_u[comp] = d_u[ind + strides_comp * comp];
  }
}

//------------------------------------------------------------------------------
// E-vector -> Q-vector, strided
//------------------------------------------------------------------------------
inline void readSliceQuadsStrided3d(const CeedInt num_comp, const CeedInt Q_1D, CeedInt strides_node, CeedInt strides_comp, CeedInt strides_elem,
                                    const CeedInt num_elem, const CeedInt q, const global CeedScalar *restrict d_u,
                                    private CeedScalar *restrict r_u) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < Q_1D && item_id_y < Q_1D && elem < num_elem) {
    const CeedInt node = item_id_x + Q_1D * (item_id_y + Q_1D * q);
    const CeedInt ind  = node * strides_node + elem * strides_elem;
    for (CeedInt comp = 0; comp < num_comp; ++comp) r_u[comp] = d_u[ind + comp * strides_comp];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
inline void writeDofsOffset3d(const CeedInt num_comp, const CeedInt strides_comp, const CeedInt P_1D, const CeedInt num_elem,
                              const global CeedInt *restrict indices, const private CeedScalar *restrict r_v, global CeedAtomicScalar *restrict d_v) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    for (CeedInt z = 0; z < P_1D; ++z) {
      const CeedInt node = item_id_x + item_id_y * P_1D + z * P_1D * P_1D;
      const CeedInt ind  = indices[node + elem * P_1D * P_1D * P_1D];
      for (CeedInt comp = 0; comp < num_comp; ++comp)
        atomic_fetch_add_explicit(&d_v[ind + strides_comp * comp], r_v[z + comp * P_1D], memory_order_relaxed, memory_scope_device);
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
inline void writeDofsStrided3d(const CeedInt num_comp, const CeedInt P_1D, const CeedInt strides_node, const CeedInt strides_comp,
                               const CeedInt strides_elem, const CeedInt num_elem, const private CeedScalar *restrict r_v,
                               global CeedScalar *restrict d_v) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);
  const CeedInt elem      = get_global_id(2);

  if (item_id_x < P_1D && item_id_y < P_1D && elem < num_elem) {
    for (CeedInt z = 0; z < P_1D; ++z) {
      const CeedInt node = item_id_x + P_1D * (item_id_y + P_1D * z);
      const CeedInt ind  = node * strides_node + elem * strides_elem;
      for (CeedInt comp = 0; comp < num_comp; ++comp) d_v[ind + comp * strides_comp] += r_v[z + comp * P_1D];
    }
  }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives computation
//------------------------------------------------------------------------------
inline void gradCollo3d(const CeedInt num_comp, const CeedInt Q_1D, const CeedInt q, const private CeedScalar *restrict r_U,
                        const local CeedScalar *s_G, private CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  for (CeedInt comp = 0; comp < num_comp; ++comp) {
    if (item_id_x < Q_1D && item_id_y < Q_1D) {
      scratch[item_id_x + item_id_y * T_1D] = r_U[q + comp * Q_1D];
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if (item_id_x < Q_1D && item_id_y < Q_1D) {
      // X derivative
      r_V[comp + 0 * num_comp] = 0.0;
      for (CeedInt i = 0; i < Q_1D; ++i)
        r_V[comp + 0 * num_comp] += s_G[i + item_id_x * Q_1D] * scratch[i + item_id_y * T_1D];  // Contract x direction (X derivative)

      // Y derivative
      r_V[comp + 1 * num_comp] = 0.0;
      for (CeedInt i = 0; i < Q_1D; ++i)
        r_V[comp + 1 * num_comp] += s_G[i + item_id_y * Q_1D] * scratch[item_id_x + i * T_1D];  // Contract y direction (Y derivative)

      // Z derivative
      r_V[comp + 2 * num_comp] = 0.0;
      for (CeedInt i = 0; i < Q_1D; ++i) r_V[comp + 2 * num_comp] += s_G[i + q * Q_1D] * r_U[i + comp * Q_1D];  // Contract z direction (Z derivative)
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D collocated derivatives transpose
//------------------------------------------------------------------------------
inline void gradColloTranspose3d(const CeedInt num_comp, const CeedInt Q_1D, const CeedInt q, const private CeedScalar *restrict r_U,
                                 const local CeedScalar *restrict s_G, private CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  for (CeedInt comp = 0; comp < num_comp; ++comp) {
    // X derivative
    if (item_id_x < Q_1D && item_id_y < Q_1D) {
      scratch[item_id_x + item_id_y * T_1D] = r_U[comp + 0 * num_comp];
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if (item_id_x < Q_1D && item_id_y < Q_1D) {
      for (CeedInt i = 0; i < Q_1D; ++i)
        r_V[q + comp * Q_1D] += s_G[item_id_x + i * Q_1D] * scratch[i + item_id_y * T_1D];  // Contract x direction (X derivative)
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // Y derivative
    if (item_id_x < Q_1D && item_id_y < Q_1D) {
      scratch[item_id_x + item_id_y * T_1D] = r_U[comp + 1 * num_comp];
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if (item_id_x < Q_1D && item_id_y < Q_1D) {
      for (CeedInt i = 0; i < Q_1D; ++i)
        r_V[q + comp * Q_1D] += s_G[item_id_y + i * Q_1D] * scratch[item_id_x + i * T_1D];  // Contract y direction (Y derivative)
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // Z derivative
    if (item_id_x < Q_1D && item_id_y < Q_1D) {
      for (CeedInt i = 0; i < Q_1D; ++i)
        r_V[i + comp * Q_1D] += s_G[i + q * Q_1D] * r_U[comp + 2 * num_comp];  // PARTIAL contract z direction (Z derivative)
    }
  }
}

//------------------------------------------------------------------------------

#endif  // CEED_SYCL_GEN_TEMPLATES_H
