// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>

#define CEED_AVX_EVEN_ODD_CACHE_MAX 8
#define CEED_AVX_EVEN_ODD_MIN_DIM 4

typedef struct {
  const CeedScalar *t_ptr;
  CeedTransposeMode t_mode;
  CeedInt           B, J, B_half, J_half;
  int               symmetry;
  CeedScalar       *t_copy;
  CeedScalar       *t_even;
  CeedScalar       *t_odd;
} CeedTensorContract_Avx_CacheEntry;

typedef struct {
  CeedInt                           n_cached;
  CeedTensorContract_Avx_CacheEntry cache[CEED_AVX_EVEN_ODD_CACHE_MAX];
} CeedTensorContract_Avx;

CEED_INTERN int CeedTensorContractCreate_Avx(CeedTensorContract contract);
