// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../common/weight.h"
#include "hip/hip_runtime.h"

//////////////////////////////////////////////////////////////////////////////////////////
// NonTensor weight function
extern "C" void magma_weight_nontensor(magma_int_t grid, magma_int_t threads, magma_int_t nelem, magma_int_t Q, CeedScalar *dqweight, CeedScalar *dv,
                                       magma_queue_t queue) {
  hipLaunchKernelGGL(magma_weight_nontensor_kernel, dim3(grid), dim3(threads), 0, magma_queue_get_hip_stream(queue), nelem, Q, dqweight, dv);
}
