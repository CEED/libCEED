// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-compile.hpp"

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <sstream>
#include <sycl/sycl.hpp>

#include "ceed-sycl-common.hpp"

// Kris: These functions may not be needed for sycl.

//------------------------------------------------------------------------------
// Compile SYCL kernel
//------------------------------------------------------------------------------
// int CeedCompileSycl(Ceed ceed, const char *source, CUmodule *module, const CeedInt num_defines, ...) {
//   return CEED_ERROR_SUCCESS;
// }

//------------------------------------------------------------------------------
// Get SYCL kernel
//------------------------------------------------------------------------------
// int CeedGetKernelSycl(Ceed ceed, CUmodule module, const char *name, CUfunction *kernel) {
//   return CEED_ERROR_SUCCESS;
// }

// Run kernel with block size selected automatically based on the kernel (which may use enough registers to require a smaller block size than the
// hardware is capable).
// int CeedRunKernelAutoblockSycl(Ceed ceed, CUfunction kernel, size_t points, void **args) {
//   int min_grid_size, max_block_size;
//   CeedCallSycl(ceed, cuOccupancyMaxPotentialBlockSize(&min_grid_size, &max_block_size, kernel, NULL, 0, 0x10000));
//   CeedCallBackend(CeedRunKernelSycl(ceed, kernel, CeedDivUpInt(points, max_block_size), max_block_size, args));
//   return 0;
// }

//------------------------------------------------------------------------------
// Run SYCL kernel
//------------------------------------------------------------------------------
// int CeedRunKernelSycl(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size, void **args) {
//   return CEED_ERROR_SUCCESS;
// }

//------------------------------------------------------------------------------
// Run SYCL kernel for spatial dimension
//------------------------------------------------------------------------------
// int CeedRunKernelDimSycl(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y, const int block_size_z,
//                          void **args) {
//   return CEED_ERROR_SUCCESS;
// }

//------------------------------------------------------------------------------
// Run SYCL kernel for spatial dimension with shared memory
//------------------------------------------------------------------------------
// int CeedRunKernelDimSharedSycl(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y,
//                                const int block_size_z, const int shared_mem_size, void **args) {
//   return CEED_ERROR_SUCCESS;
// }

//------------------------------------------------------------------------------
