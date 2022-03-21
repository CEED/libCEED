// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-magma.h"

//////////////////////////////////////////////////////////////////////////////////////////
#ifdef __cplusplus
CEED_INTERN "C"
#endif
magma_int_t
magma_weight(
  magma_int_t Q, magma_int_t dim,
  const CeedScalar *dqweight1d,
  CeedScalar *dV, magma_int_t v_stride,
  magma_int_t nelem, magma_kernel_mode_t kernel_mode,
  magma_queue_t queue) {
  magma_int_t launch_failed = 0;

  if (kernel_mode == MAGMA_KERNEL_DIM_SPECIFIC) {
    switch(dim) {
    case 1: launch_failed = magma_weight_1d(Q, dqweight1d, dV, v_stride, nelem,
                                              queue); break;
    case 2: launch_failed = magma_weight_2d(Q, dqweight1d, dV, v_stride, nelem,
                                              queue); break;
    case 3: launch_failed = magma_weight_3d(Q, dqweight1d, dV, v_stride, nelem,
                                              queue); break;
    default: launch_failed = 1;
    }
  } else {
    launch_failed = magma_weight_generic(Q, dim, dqweight1d, dV, v_stride, nelem,
                                         queue);
  }

  return launch_failed;
}
