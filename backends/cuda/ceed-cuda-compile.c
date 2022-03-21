// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <string.h>
#include "ceed-cuda-common.h"
#include "ceed-cuda-compile.h"

#define CeedChk_Nvrtc(ceed, x) \
do { \
  nvrtcResult result = x; \
  if (result != NVRTC_SUCCESS) \
    return CeedError((ceed), CEED_ERROR_BACKEND, nvrtcGetErrorString(result)); \
} while (0)

//------------------------------------------------------------------------------
// Compile CUDA kernel
//------------------------------------------------------------------------------
int CeedCompileCuda(Ceed ceed, const char *source, CUmodule *module,
                    const CeedInt num_opts, ...) {
  int ierr;
  cudaFree(0); // Make sure a Context exists for nvrtc
  nvrtcProgram prog;
  CeedChk_Nvrtc(ceed, nvrtcCreateProgram(&prog, source, NULL, 0, NULL, NULL));

  // Get kernel specific options, such as kernel constants
  const int opts_len = 32;
  const int opts_extra = 4;
  const char *opts[num_opts + opts_extra];
  char buf[num_opts][opts_len];
  if (num_opts > 0) {
    va_list args;
    va_start(args, num_opts);
    char *name;
    int val;
    for (int i = 0; i < num_opts; i++) {
      name = va_arg(args, char *);
      val = va_arg(args, int);
      snprintf(&buf[i][0], opts_len,"-D%s=%d", name, val);
      opts[i] = &buf[i][0];
    }
    va_end(args);
  }

  // Standard backend options
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
    opts[num_opts]   = "-DCeedScalar=float";
  } else {
    opts[num_opts]   = "-DCeedScalar=double";
  }
  opts[num_opts + 1] = "-DCeedInt=int";
  opts[num_opts + 2] = "-default-device";
  struct cudaDeviceProp prop;
  Ceed_Cuda *ceed_data;
  ierr = CeedGetData(ceed, &ceed_data); CeedChkBackend(ierr);
  ierr = cudaGetDeviceProperties(&prop, ceed_data->device_id);
  CeedChk_Cu(ceed, ierr);
  char buff[opts_len];
  snprintf(buff, opts_len,"-arch=compute_%d%d", prop.major, prop.minor);
  opts[num_opts + 3] = buff;

  // Compile kernel
  nvrtcResult result = nvrtcCompileProgram(prog, num_opts + opts_extra, opts);
  if (result != NVRTC_SUCCESS) {
    size_t log_size;
    CeedChk_Nvrtc(ceed, nvrtcGetProgramLogSize(prog, &log_size));
    char *log;
    ierr = CeedMalloc(log_size, &log); CeedChkBackend(ierr);
    CeedChk_Nvrtc(ceed, nvrtcGetProgramLog(prog, log));
    return CeedError(ceed, CEED_ERROR_BACKEND, "%s\n%s",
                     nvrtcGetErrorString(result), log);
  }

  size_t ptx_size;
  CeedChk_Nvrtc(ceed, nvrtcGetPTXSize(prog, &ptx_size));
  char *ptx;
  ierr = CeedMalloc(ptx_size, &ptx); CeedChkBackend(ierr);
  CeedChk_Nvrtc(ceed, nvrtcGetPTX(prog, ptx));
  CeedChk_Nvrtc(ceed, nvrtcDestroyProgram(&prog));

  CeedChk_Cu(ceed, cuModuleLoadData(module, ptx));
  ierr = CeedFree(&ptx); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get CUDA kernel
//------------------------------------------------------------------------------
int CeedGetKernelCuda(Ceed ceed, CUmodule module, const char *name,
                      CUfunction *kernel) {
  CeedChk_Cu(ceed, cuModuleGetFunction(kernel, module, name));
  return CEED_ERROR_SUCCESS;
}

// Run kernel with block size selected automatically based on the kernel (which
// may use enough registers to require a smaller block size than the hardware is
// capable).
int CeedRunKernelAutoblockCuda(Ceed ceed, CUfunction kernel, size_t points,
                               void **args) {
  int min_grid_size, max_block_size;
  CeedChk_Cu(ceed, cuOccupancyMaxPotentialBlockSize(&min_grid_size,
             &max_block_size, kernel, NULL, 0, 0x10000));
  CeedChkBackend(CeedRunKernelCuda(ceed, kernel, CeedDivUpInt(points,
                                   max_block_size), max_block_size, args));
  return 0;
}

//------------------------------------------------------------------------------
// Run CUDA kernel
//------------------------------------------------------------------------------
int CeedRunKernelCuda(Ceed ceed, CUfunction kernel, const int grid_size,
                      const int block_size, void **args) {
  CeedChkBackend(CeedRunKernelDimSharedCuda(ceed, kernel, grid_size,
                 block_size, 1, 1, 0, args));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel for spatial dimension
//------------------------------------------------------------------------------
int CeedRunKernelDimCuda(Ceed ceed, CUfunction kernel, const int grid_size,
                         const int block_size_x, const int block_size_y,
                         const int block_size_z, void **args) {
  CeedChkBackend(CeedRunKernelDimSharedCuda(ceed, kernel, grid_size,
                 block_size_x, block_size_y, block_size_z,
                 0, args));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel for spatial dimension with sharde memory
//------------------------------------------------------------------------------
int CeedRunKernelDimSharedCuda(Ceed ceed, CUfunction kernel,
                               const int grid_size, const int block_size_x,
                               const int block_size_y, const int block_size_z,
                               const int shared_mem_size, void **args) {
  CUresult result = cuLaunchKernel(kernel, grid_size, 1, 1,
                                   block_size_x, block_size_y, block_size_z,
                                   shared_mem_size, NULL, args, NULL);
  if (result == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) {
    int max_threads_per_block, shared_size_bytes, num_regs;
    cuFuncGetAttribute(&max_threads_per_block,
                       CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel);
    cuFuncGetAttribute(&shared_size_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                       kernel);
    cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: max_threads_per_block %d on block size (%d,%d,%d), shared_size %d, num_regs %d",
                     max_threads_per_block, block_size_x, block_size_y, block_size_z,
                     shared_size_bytes, num_regs);
  } else CeedChk_Cu(ceed, result);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
