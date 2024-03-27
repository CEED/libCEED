// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <cuda.h>

static inline CeedInt CeedDivUpInt(CeedInt numerator, CeedInt denominator) { return (numerator + denominator - 1) / denominator; }

CEED_INTERN int CeedCompile_Cuda(Ceed ceed, const char *source, CUmodule *module, const CeedInt num_defines, ...);

CEED_INTERN int CeedGetKernel_Cuda(Ceed ceed, CUmodule module, const char *name, CUfunction *kernel);

CEED_INTERN int CeedRunKernel_Cuda(Ceed ceed, CUfunction kernel, int grid_size, int block_size, void **args);

CEED_INTERN int CeedRunKernelAutoblockCuda(Ceed ceed, CUfunction kernel, size_t points, void **args);

CEED_INTERN int CeedRunKernelDim_Cuda(Ceed ceed, CUfunction kernel, int grid_size, int block_size_x, int block_size_y, int block_size_z, void **args);

CEED_INTERN int CeedRunKernelDimShared_Cuda(Ceed ceed, CUfunction kernel, int grid_size, int block_size_x, int block_size_y, int block_size_z,
                                            int shared_mem_size, void **args);
