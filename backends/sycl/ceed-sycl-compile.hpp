// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_sycl_compile_hpp
#define _ceed_sycl_compile_hpp

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <sycl/sycl.hpp>

// static inline CeedInt CeedDivUpInt(CeedInt numerator, CeedInt denominator) { return (numerator + denominator - 1) / denominator; }

// CEED_INTERN int CeedCompileSycl(Ceed ceed, const char *source, CUmodule *module, const CeedInt num_defines, ...);

// CEED_INTERN int CeedGetKernelSycl(Ceed ceed, CUmodule module, const char *name, CUfunction *kernel);

// CEED_INTERN int CeedRunKernelSycl(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size, void **args);

// CEED_INTERN int CeedRunKernelAutoblockSycl(Ceed ceed, CUfunction kernel, size_t size, void **args);

// CEED_INTERN int CeedRunKernelDimSycl(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y,
//                                      const int block_size_z, void **args);

// CEED_INTERN int CeedRunKernelDimSharedSycl(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y,
//                                            const int block_size_z, const int shared_mem_size, void **args);

#endif  // _ceed_sycl_compile_h
