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
magma_grad(
  magma_int_t P, magma_int_t Q, magma_int_t dim, magma_int_t ncomp,
  const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode,
  const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU,
  magma_int_t dstrdU,
  CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV,
  magma_int_t dstrdV,
  magma_int_t nelem, magma_kernel_mode_t kernel_mode,
  magma_queue_t queue) {
  magma_int_t launch_failed = 0;

  if (kernel_mode == MAGMA_KERNEL_DIM_SPECIFIC) {
    if (tmode == CEED_TRANSPOSE) {
      switch(dim) {
      case 1: launch_failed =  magma_grad_1d(P, Q, ncomp, dinterp1d, dgrad1d, tmode,
                                               dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem,
                                               queue); break;
      case 2: launch_failed = magma_gradt_2d(P, Q, ncomp, dinterp1d, dgrad1d, tmode,
                                               dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV,
                                               dstrdV, nelem, queue); break;
      case 3: launch_failed = magma_gradt_3d(P, Q, ncomp, dinterp1d, dgrad1d, tmode,
                                               dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV,
                                               dstrdV, nelem, queue); break;
      default: launch_failed = 1;
      }
    } else {
      switch(dim) {
      case 1: launch_failed =  magma_grad_1d(P, Q, ncomp, dinterp1d, dgrad1d, tmode,
                                               dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem,
                                               queue); break;
      case 2: launch_failed = magma_gradn_2d(P, Q, ncomp, dinterp1d, dgrad1d, tmode,
                                               dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV,
                                               dstrdV, nelem, queue); break;
      case 3: launch_failed = magma_gradn_3d(P, Q, ncomp, dinterp1d, dgrad1d, tmode,
                                               dU, estrdU, cstrdU, dstrdU, dV, estrdV, cstrdV,
                                               dstrdV, nelem, queue); break;
      default: launch_failed = 1;
      }
    }
  } else {
    launch_failed = magma_grad_generic(
                      P, Q, dim, ncomp,
                      dinterp1d, dgrad1d, tmode,
                      dU, estrdU, cstrdU, dstrdU,
                      dV, estrdV, cstrdV, dstrdV,
                      nelem, queue );
  }

  return launch_failed;
}
