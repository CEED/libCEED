// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_hip_compile_h
#define _ceed_hip_compile_h

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <hip/hip_runtime.h>

static inline CeedInt CeedDivUpInt(CeedInt numerator, CeedInt denominator) { return (numerator + denominator - 1) / denominator; }

CEED_INTERN int CeedCompileHip(Ceed ceed, const char *source, hipModule_t *module, const CeedInt num_defines, ...);

CEED_INTERN int CeedGetKernelHip(Ceed ceed, hipModule_t module, const char *name, hipFunction_t *kernel);

CEED_INTERN int CeedRunKernelHip(Ceed ceed, hipFunction_t kernel, const int grid_size, const int block_size, void **args);

CEED_INTERN int CeedRunKernelDimHip(Ceed ceed, hipFunction_t kernel, const int grid_size, const int block_size_x, const int block_size_y,
                                    const int block_size_z, void **args);

CEED_INTERN int CeedRunKernelDimSharedHip(Ceed ceed, hipFunction_t kernel, const int grid_size, const int block_size_x, const int block_size_y,
                                          const int block_size_z, const int shared_mem_size, void **args);

#endif  // _ceed_hip_compile_h
