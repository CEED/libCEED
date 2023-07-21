// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_magma_gemm_h
#define _ceed_magma_gemm_h

#include <ceed.h>
#include <ceed/backend.h>

CEED_INTERN void gemm_selector(int gpu_arch, char precision, char transA, int m, int n, int k, int *nbatch, int *use_magma);

CEED_INTERN CeedInt nontensor_rtc_get_nb(int gpu_arch, char precision, CeedEvalMode emode, CeedTransposeMode tmode, int P_, int N, int Q_);

#endif
