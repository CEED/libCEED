// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_cuda_compile_h
#define _ceed_cuda_compile_h

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <cuda.h>
#include <nvrtc.h>

static inline CeedInt CeedDivUpInt(CeedInt numerator, CeedInt denominator) { return (numerator + denominator - 1) / denominator; }

CEED_INTERN int CeedCompileCuda(Ceed ceed, const char *source, CUmodule *module, const CeedInt num_defines, ...);

CEED_INTERN int CeedGetKernelCuda(Ceed ceed, CUmodule module, const char *name, CUfunction *kernel);

CEED_INTERN int CeedRunKernelCuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size, void **args);

CEED_INTERN int CeedRunKernelAutoblockCuda(Ceed ceed, CUfunction kernel, size_t size, void **args);

CEED_INTERN int CeedRunKernelDimCuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y,
                                     const int block_size_z, void **args);

CEED_INTERN int CeedRunKernelDimSharedCuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y,
                                           const int block_size_z, const int shared_mem_size, void **args);

#endif  // _ceed_cuda_compile_h
