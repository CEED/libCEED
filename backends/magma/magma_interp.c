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
magma_interp(
  magma_int_t P, magma_int_t Q,
  magma_int_t dim, magma_int_t ncomp,
  const CeedScalar *dT, CeedTransposeMode tmode,
  const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU,
  CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV,
  magma_int_t nelem, magma_kernel_mode_t kernel_mode,
  magma_queue_t queue) {
  magma_int_t launch_failed = 0;

  if (kernel_mode == MAGMA_KERNEL_DIM_SPECIFIC) {
    switch(dim) {
    case 1: launch_failed = magma_interp_1d(P, Q, ncomp, dT, tmode, dU, estrdU,
                                              cstrdU, dV, estrdV, cstrdV, nelem, queue); break;
    case 2: launch_failed = magma_interp_2d(P, Q, ncomp, dT, tmode, dU, estrdU,
                                              cstrdU, dV, estrdV, cstrdV, nelem, queue); break;
    case 3: launch_failed = magma_interp_3d(P, Q, ncomp, dT, tmode, dU, estrdU,
                                              cstrdU, dV, estrdV, cstrdV, nelem, queue); break;
    default: launch_failed = 1;
    }
  } else {
    launch_failed = magma_interp_generic(
                      P, Q, dim, ncomp,
                      dT, tmode,
                      dU, estrdU, cstrdU,
                      dV, estrdV, cstrdV,
                      nelem, queue);
  }

  return launch_failed;
}
