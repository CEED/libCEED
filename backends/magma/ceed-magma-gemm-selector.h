// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_MAGMA_GEMM_SELECTOR_H
#define CEED_MAGMA_GEMM_SELECTOR_H

#include "ceed-magma.h"

////////////////////////////////////////////////////////////////////////////////
CEED_INTERN void gemm_selector(int gpu_arch, char precision, char trans_A, int m, int n, int k, int *n_batch, int *use_magma);

////////////////////////////////////////////////////////////////////////////////
CEED_INTERN CeedInt nontensor_rtc_get_nb(int gpu_arch, char trans_A, int q_comp, int P, int Q, int N);

#endif  // CEED_MAGMA_GEMM_SELECTOR_H
