// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_XSMM_H
#define CEED_XSMM_H

#include <ceed.h>
#include <ceed/backend.h>

#if ((LIBXSMM_VERSION_MAJOR > 1 || \
      (LIBXSMM_VERSION_MAJOR == 1 && (LIBXSMM_VERSION_MINOR > 17 || (LIBXSMM_VERSION_MINOR == 17 && LIBXSMM_VERSION_PATCH >= 3710)))))
#define libxsmm_dispatch_gemm(a, b, c) libxsmm_dispatch_gemm(a, b, c)
#else
#define libxsmm_dispatch_gemm(a, b, c) libxsmm_dispatch_gemm_v2(a, b, c)
#endif

CEED_INTERN int CeedTensorContractCreate_Xsmm(CeedTensorContract contract);

#endif  // CEED_XSMM_H
