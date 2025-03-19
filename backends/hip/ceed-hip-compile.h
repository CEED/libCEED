// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>

static inline CeedInt CeedDivUpInt(CeedInt numerator, CeedInt denominator) { return (numerator + denominator - 1) / denominator; }

CEED_INTERN int CeedCompile_Hip(Ceed ceed, const char *source, hipModule_t *module, const CeedInt num_defines, ...);
CEED_INTERN int CeedTryCompile_Hip(Ceed ceed, const char *source, bool *is_compile_good, hipModule_t *module, const CeedInt num_defines, ...);

CEED_INTERN int CeedGetKernel_Hip(Ceed ceed, hipModule_t module, const char *name, hipFunction_t *kernel);

CEED_INTERN int CeedRunKernel_Hip(Ceed ceed, hipFunction_t kernel, int grid_size, int block_size, void **args);

CEED_INTERN int CeedRunKernelDim_Hip(Ceed ceed, hipFunction_t kernel, int grid_size, int block_size_x, int block_size_y, int block_size_z,
                                     void **args);

CEED_INTERN int CeedRunKernelDimShared_Hip(Ceed ceed, hipFunction_t kernel, hipStream_t stream, int grid_size, int block_size_x, int block_size_y,
                                           int block_size_z, int shared_mem_size, void **args);
CEED_INTERN int CeedTryRunKernelDimShared_Hip(Ceed ceed, hipFunction_t kernel, hipStream_t stream, int grid_size, int block_size_x, int block_size_y,
                                              int block_size_z, int shared_mem_size, bool *is_good_run, void **args);
