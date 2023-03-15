// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef wall_dist_func_h
#define wall_dist_func_h

#include <ceed.h>

#include "newtonian_state.h"
#include "utils.h"

CEED_QFUNCTION_HELPER int DistanceFunction(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // inputs
  const CeedScalar(*q)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*x)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  //  Outputs
  CeedScalar(*v)[CEED_Q_VLA]  = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*dv)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];

  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;

  return 0;
}

#endif  // wall_dist_func_h