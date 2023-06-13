// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL shared memory tensor product basis templates
#ifndef _ceed_sycl_shared_basis_tensor_templates_h
#define _ceed_sycl_shared_basis_tensor_templates_h

#include <ceed.h>

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 1D tensor contraction x
//------------------------------------------------------------------------------
inline void ContractX1d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                        private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);

  scratch[item_id_x] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + item_id_x * P_1D] * scratch[i];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 1D transpose tensor contraction x
//------------------------------------------------------------------------------
inline void ContractTransposeX1d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                                 private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);

  scratch[item_id_x] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_x + i * P_1D] * scratch[i];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 1D interpolate to quadrature points
//------------------------------------------------------------------------------
inline void Interp1d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                     local const CeedScalar *restrict s_B, private CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX1d(P_1D, Q_1D, r_U + comp, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D interpolate transpose
//------------------------------------------------------------------------------
inline void InterpTranspose1d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                              local const CeedScalar *restrict s_B, private CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeX1d(P_1D, Q_1D, r_U + comp, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D derivatives at quadrature points
//------------------------------------------------------------------------------
inline void Grad1d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                   local const CeedScalar *restrict s_G, private CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX1d(P_1D, Q_1D, r_U + comp, s_G, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D derivatives transpose
//------------------------------------------------------------------------------
inline void GradTranspose1d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                            local const CeedScalar *restrict s_G, private CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeX1d(P_1D, Q_1D, r_U + comp, s_G, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 1D quadrature weights
//------------------------------------------------------------------------------
inline void Weight1d(const CeedInt Q_1D, const CeedScalar *restrict q_weight_1d, CeedScalar *restrict w) {
  const CeedInt item_id_x = get_local_id(0);
  *w                      = (item_id_x < Q_1D) ? q_weight_1d[item_id_x] : 0.0;
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 2D tensor contraction x
//------------------------------------------------------------------------------
inline void ContractX2d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                        private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  scratch[item_id_x + item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < Q_1D && item_id_y < P_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + item_id_x * P_1D] * scratch[i + item_id_y * T_1D];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D tensor contract y
//------------------------------------------------------------------------------
inline void ContractY2d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                        private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  scratch[item_id_x + item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < Q_1D && item_id_y < Q_1D) {
    for (CeedInt i = 0; i < P_1D; i++) {
      *V += B[i + item_id_y * P_1D] * scratch[item_id_x + i * T_1D];  // Contract y direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract y
//------------------------------------------------------------------------------
inline void ContractTransposeY2d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                                 private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  scratch[item_id_x + item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < Q_1D && item_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_y + i * P_1D] * scratch[item_id_x + i * T_1D];  // Contract y direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract x
//------------------------------------------------------------------------------
inline void ContractTransposeX2d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                                 private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  scratch[item_id_x + item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  *V = 0.0;
  if (item_id_x < P_1D && item_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_x + i * P_1D] * scratch[i + item_id_y * T_1D];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D transpose tensor contract and add x
//------------------------------------------------------------------------------
inline void ContractTransposeAddX2d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                                    private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  scratch[item_id_x + item_id_y * T_1D] = *U;
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  if (item_id_x < P_1D && item_id_y < P_1D) {
    for (CeedInt i = 0; i < Q_1D; i++) {
      *V += B[item_id_x + i * P_1D] * scratch[i + item_id_y * T_1D];  // Contract x direction
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
inline void InterpTensor2d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                           local const CeedScalar *restrict s_B, private CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  CeedScalar r_t[1];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2d(P_1D, Q_1D, r_U + comp, s_B, r_t, scratch);
    ContractY2d(P_1D, Q_1D, r_t, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 2D interpolate transpose
//------------------------------------------------------------------------------
inline void InterpTransposeTensor2d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                                    local const CeedScalar *restrict s_B, private CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  CeedScalar r_t[1];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeY2d(P_1D, Q_1D, r_U + comp, s_B, r_t, scratch);
    ContractTransposeX2d(P_1D, Q_1D, r_t, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 2D derivatives at quadrature points
//------------------------------------------------------------------------------
inline void GradTensor2d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                         local const CeedScalar *restrict s_B, local const CeedScalar *restrict s_G, private CeedScalar *restrict r_V,
                         local CeedScalar *restrict scratch) {
  CeedScalar r_t[1];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2d(P_1D, Q_1D, r_U + comp, s_G, r_t, scratch);
    ContractY2d(P_1D, Q_1D, r_t, s_B, r_V + comp + 0 * NUM_COMP, scratch);
    ContractX2d(P_1D, Q_1D, r_U + comp, s_B, r_t, scratch);
    ContractY2d(P_1D, Q_1D, r_t, s_G, r_V + comp + 1 * NUM_COMP, scratch);
  }
}

//------------------------------------------------------------------------------
// 2D derivatives transpose
//------------------------------------------------------------------------------
inline void GradTransposeTensor2d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                                  local const CeedScalar *restrict s_B, local const CeedScalar *restrict s_G, private CeedScalar *restrict r_V,
                                  local CeedScalar *restrict scratch) {
  CeedScalar r_t[1];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeY2d(P_1D, Q_1D, r_U + comp + 0 * NUM_COMP, s_B, r_t, scratch);
    ContractTransposeX2d(P_1D, Q_1D, r_t, s_G, r_V + comp, scratch);
    ContractTransposeY2d(P_1D, Q_1D, r_U + comp + 1 * NUM_COMP, s_G, r_t, scratch);
    ContractTransposeAddX2d(P_1D, Q_1D, r_t, s_B, r_V + comp, scratch);
  }
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
inline void WeightTensor2d(const CeedInt Q_1D, const CeedScalar *restrict q_weight_1d, CeedScalar *restrict w) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  *w = (item_id_x < Q_1D && item_id_y < Q_1D) ? q_weight_1d[item_id_x] * q_weight_1d[item_id_y] : 0.0;
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 3D tensor contract x
//------------------------------------------------------------------------------
inline void ContractX3d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                        private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < P_1D; i++) {
    r_B[i] = B[i + item_id_x * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    scratch[item_id_x + item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    V[k] = 0.0;
    if (item_id_x < Q_1D && item_id_y < P_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += r_B[i] * scratch[i + item_id_y * T_1D];  // Contract x direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract y
//------------------------------------------------------------------------------
inline void ContractY3d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                        private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < P_1D; i++) {
    r_B[i] = B[i + item_id_y * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    scratch[item_id_x + item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    V[k] = 0.0;
    if (item_id_x < Q_1D && item_id_y < Q_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += r_B[i] * scratch[item_id_x + i * T_1D];  // Contract y direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D tensor contract z
//------------------------------------------------------------------------------
inline void ContractZ3d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                        private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  for (CeedInt k = 0; k < Q_1D; k++) {
    V[k] = 0.0;
    if (item_id_x < Q_1D && item_id_y < Q_1D) {
      for (CeedInt i = 0; i < P_1D; i++) {
        V[k] += B[i + k * P_1D] * U[i];  // Contract z direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract z
//------------------------------------------------------------------------------
inline void ContractTransposeZ3d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                                 private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  for (CeedInt k = 0; k < P_1D; k++) {
    V[k] = 0.0;
    if (item_id_x < Q_1D && item_id_y < Q_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += B[k + i * P_1D] * U[i];  // Contract z direction
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
inline void ContractTransposeY3d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                                 private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[item_id_y + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    scratch[item_id_x + item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    V[k] = 0.0;
    if (item_id_x < Q_1D && item_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * scratch[item_id_x + i * T_1D];  // Contract y direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract y
//------------------------------------------------------------------------------
inline void ContractTransposeAddY3d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                                    private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[item_id_y + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    scratch[item_id_x + item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    if (item_id_x < Q_1D && item_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * scratch[item_id_x + i * T_1D];  // Contract y direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract x
//------------------------------------------------------------------------------
inline void ContractTransposeX3d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                                 private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[item_id_x + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    scratch[item_id_x + item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    V[k] = 0.0;
    if (item_id_x < P_1D && item_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * scratch[i + item_id_y * T_1D];  // Contract x direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract add x
//------------------------------------------------------------------------------
inline void ContractTransposeAddX3d(const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict U, local const CeedScalar *restrict B,
                                    private CeedScalar *restrict V, local CeedScalar *restrict scratch) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  CeedScalar r_B[T_1D];
  for (CeedInt i = 0; i < Q_1D; i++) {
    r_B[i] = B[item_id_x + i * P_1D];
  }

  for (CeedInt k = 0; k < P_1D; k++) {
    scratch[item_id_x + item_id_y * T_1D] = U[k];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if (item_id_x < P_1D && item_id_y < P_1D) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        V[k] += r_B[i] * scratch[i + item_id_y * T_1D];  // Contract x direction
      }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//------------------------------------------------------------------------------
// 3D interpolate to quadrature points
//------------------------------------------------------------------------------
inline void InterpTensor3d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                           local const CeedScalar *restrict s_B, private CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX3d(P_1D, Q_1D, r_U + comp * P_1D, s_B, r_t1, scratch);
    ContractY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
    ContractZ3d(P_1D, Q_1D, r_t2, s_B, r_V + comp * Q_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D interpolate transpose
//------------------------------------------------------------------------------
inline void InterpTransposeTensor3d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                                    local const CeedScalar *restrict s_B, private CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeZ3d(P_1D, Q_1D, r_U + comp * Q_1D, s_B, r_t1, scratch);
    ContractTransposeY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
    ContractTransposeX3d(P_1D, Q_1D, r_t2, s_B, r_V + comp * P_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
inline void GradTensor3d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                         local const CeedScalar *restrict s_B, local const CeedScalar *restrict s_G, private CeedScalar *restrict r_V,
                         local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX3d(P_1D, Q_1D, r_U + comp * P_1D, s_G, r_t1, scratch);
    ContractY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
    ContractZ3d(P_1D, Q_1D, r_t2, s_B, r_V + comp * Q_1D + 0 * NUM_COMP * Q_1D, scratch);
    ContractX3d(P_1D, Q_1D, r_U + comp * P_1D, s_B, r_t1, scratch);
    ContractY3d(P_1D, Q_1D, r_t1, s_G, r_t2, scratch);
    ContractZ3d(P_1D, Q_1D, r_t2, s_B, r_V + comp * Q_1D + 1 * NUM_COMP * Q_1D, scratch);
    ContractX3d(P_1D, Q_1D, r_U + comp * P_1D, s_B, r_t1, scratch);
    ContractY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
    ContractZ3d(P_1D, Q_1D, r_t2, s_G, r_V + comp * Q_1D + 2 * NUM_COMP * Q_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
inline void GradTransposeTensor3d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                                  local const CeedScalar *restrict s_B, local const CeedScalar *restrict s_G, private CeedScalar *restrict r_V,
                                  local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeZ3d(P_1D, Q_1D, r_U + comp * Q_1D + 0 * NUM_COMP * Q_1D, s_B, r_t1, scratch);
    ContractTransposeY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
    ContractTransposeX3d(P_1D, Q_1D, r_t2, s_G, r_V + comp * P_1D, scratch);
    ContractTransposeZ3d(P_1D, Q_1D, r_U + comp * Q_1D + 1 * NUM_COMP * Q_1D, s_B, r_t1, scratch);
    ContractTransposeY3d(P_1D, Q_1D, r_t1, s_G, r_t2, scratch);
    ContractTransposeAddX3d(P_1D, Q_1D, r_t2, s_B, r_V + comp * P_1D, scratch);
    ContractTransposeZ3d(P_1D, Q_1D, r_U + comp * Q_1D + 2 * NUM_COMP * Q_1D, s_G, r_t1, scratch);
    ContractTransposeY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
    ContractTransposeAddX3d(P_1D, Q_1D, r_t2, s_B, r_V + comp * P_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
inline void GradTensorCollocated3d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                                   local const CeedScalar *restrict s_B, local const CeedScalar *restrict s_G, private CeedScalar *restrict r_V,
                                   local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX3d(P_1D, Q_1D, r_U + comp * P_1D, s_B, r_t1, scratch);
    ContractY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
    ContractZ3d(P_1D, Q_1D, r_t2, s_B, r_t1, scratch);
    ContractX3d(Q_1D, Q_1D, r_t1, s_G, r_V + comp * Q_1D + 0 * NUM_COMP * Q_1D, scratch);
    ContractY3d(Q_1D, Q_1D, r_t1, s_G, r_V + comp * Q_1D + 1 * NUM_COMP * Q_1D, scratch);
    ContractZ3d(Q_1D, Q_1D, r_t1, s_G, r_V + comp * Q_1D + 2 * NUM_COMP * Q_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
inline void GradTransposeTensorCollocated3d(const CeedInt NUM_COMP, const CeedInt P_1D, const CeedInt Q_1D, private const CeedScalar *restrict r_U,
                                            local const CeedScalar *restrict s_B, local const CeedScalar *restrict s_G,
                                            private CeedScalar *restrict r_V, local CeedScalar *restrict scratch) {
  CeedScalar r_t1[T_1D];
  CeedScalar r_t2[T_1D];

  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractTransposeZ3d(Q_1D, Q_1D, r_U + comp * Q_1D + 2 * NUM_COMP * Q_1D, s_G, r_t2, scratch);
    ContractTransposeAddY3d(Q_1D, Q_1D, r_U + comp * Q_1D + 1 * NUM_COMP * Q_1D, s_G, r_t2, scratch);
    ContractTransposeAddX3d(Q_1D, Q_1D, r_U + comp * Q_1D + 0 * NUM_COMP * Q_1D, s_G, r_t2, scratch);
    ContractTransposeZ3d(P_1D, Q_1D, r_t2, s_B, r_t1, scratch);
    ContractTransposeY3d(P_1D, Q_1D, r_t1, s_B, r_t2, scratch);
    ContractTransposeX3d(P_1D, Q_1D, r_t2, s_B, r_V + comp * P_1D, scratch);
  }
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
// template <int Q_1D>
inline void WeightTensor3d(const CeedInt Q_1D, const CeedScalar *restrict q_weight_1d, CeedScalar *restrict w) {
  const CeedInt item_id_x = get_local_id(0);
  const CeedInt item_id_y = get_local_id(1);

  if (item_id_x < Q_1D && item_id_y < Q_1D) {
    const CeedScalar w_xy = q_weight_1d[item_id_x] * q_weight_1d[item_id_y];
    for (CeedInt q = 0; q < Q_1D; ++q) w[q] = w_xy * q_weight_1d[q];
  } else {
    for (CeedInt q = 0; q < Q_1D; q++) w[q] = 0.0;
  }
}

//------------------------------------------------------------------------------

#endif
