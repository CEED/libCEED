// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CPU JiT utilities
#pragma once

#include <ceed/types.h>

#define CeedCall(...)        \
  do {                       \
    int ierr_ = __VA_ARGS__; \
    if (ierr_) return ierr_; \
  } while (0)

constexpr CeedInt CeedIntMin(CeedInt a, CeedInt b) { return a < b ? a : b; }

constexpr CeedInt CeedIntMax(CeedInt a, CeedInt b) { return a > b ? a : b; }
