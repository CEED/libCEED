// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cuda-compile.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <stdarg.h>
#include <string.h>

#include <sstream>
#include <iostream>
#include <fstream>

#include "ceed-cuda-common.h"

#define CeedChk_Nvrtc(ceed, x)                                                                              \
  do {                                                                                                      \
    nvrtcResult result = static_cast<nvrtcResult>(x);                                                       \
    if (result != NVRTC_SUCCESS) return CeedError((ceed), CEED_ERROR_BACKEND, nvrtcGetErrorString(result)); \
  } while (0)

#define CeedCallNvrtc(ceed, ...)  \
  do {                            \
    int ierr_q_ = __VA_ARGS__;    \
    CeedChk_Nvrtc(ceed, ierr_q_); \
  } while (0)

//------------------------------------------------------------------------------
// Compile CUDA kernel
//------------------------------------------------------------------------------
using std::ifstream;
using std::ofstream;
using std::ostringstream;

static int CeedCompileCore_Cuda(Ceed ceed, const char *source, const bool throw_error, bool *is_compile_good, CUmodule *module,
                                const CeedInt num_defines, va_list args) {
  size_t                ptx_size;
  char                 *ptx;
  const int             num_opts            = 4;
  CeedInt               num_jit_source_dirs = 0, num_jit_defines = 0;
  const char          **opts;
  nvrtcProgram          prog;
  struct cudaDeviceProp prop;
  Ceed_Cuda            *ceed_data;


  cudaFree(0);  // Make sure a Context exists for nvrtc

  std::ostringstream code;

  // Get kernel specific options, such as kernel constants
  if (num_defines > 0) {
    char *name;
    int   val;

    for (int i = 0; i < num_defines; i++) {
      name = va_arg(args, char *);
      val  = va_arg(args, int);
      code << "#define " << name << " " << val << "\n";
    }
  }

  // Standard libCEED definitions for CUDA backends
  code << "#include <ceed/jit-source/cuda/cuda-jit.h>\n\n";
  //code << source;
  // Non-macro options
  CeedCallBackend(CeedCalloc(num_opts, &opts));
  opts[0] = "-default-device";
  CeedCallBackend(CeedGetData(ceed, &ceed_data));
  CeedCallCuda(ceed, cudaGetDeviceProperties(&prop, ceed_data->device_id));
  std::string arch_arg =
#if CUDA_VERSION >= 11010
      // NVRTC used to support only virtual architectures through the option
      // -arch, since it was only emitting PTX. It will now support actual
      // architectures as well to emit SASS.
      // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#dynamic-code-generation
      "-arch=sm_"
#else
      "-arch=compute_"
#endif
      + std::to_string(prop.major) + std::to_string(prop.minor);
  opts[1] = arch_arg.c_str();
  std::cout << "opt1: " << opts[1] << std::endl;
  opts[2] = "-Dint32_t=int";
  opts[3] = "-DCEED_RUNNING_JIT_PASS=1";
  // Additional include dirs
  {
    const char **jit_source_dirs;

    CeedCallBackend(CeedGetJitSourceRoots(ceed, &num_jit_source_dirs, &jit_source_dirs));
    CeedCallBackend(CeedRealloc(num_opts + num_jit_source_dirs, &opts));
    for (CeedInt i = 0; i < num_jit_source_dirs; i++) {
      std::ostringstream include_dir_arg;

      include_dir_arg << "-I" << jit_source_dirs[i];
      CeedCallBackend(CeedStringAllocCopy(include_dir_arg.str().c_str(), (char **)&opts[num_opts + i]));
      std::cout << "Opt is " << opts[num_opts + i] << std::endl;
    }
    CeedCallBackend(CeedRestoreJitSourceRoots(ceed, &jit_source_dirs));
  }
  // User defines
  {
    const char **jit_defines;

    CeedCallBackend(CeedGetJitDefines(ceed, &num_jit_defines, &jit_defines));
    CeedCallBackend(CeedRealloc(num_opts + num_jit_source_dirs + num_jit_defines, &opts));
    for (CeedInt i = 0; i < num_jit_defines; i++) {
      std::ostringstream define_arg;

      define_arg << "-D" << jit_defines[i];
      CeedCallBackend(CeedStringAllocCopy(define_arg.str().c_str(), (char **)&opts[num_opts + num_jit_source_dirs + i]));
      std::cout << "Opt is " << opts[num_opts + num_jit_source_dirs + i] << std::endl;
    }
    CeedCallBackend(CeedRestoreJitDefines(ceed, &jit_defines));
  }

  // Add string source argument provided in call
  code << source;

  // Create Program
  //CeedCallNvrtc(ceed, nvrtcCreateProgram(&prog, code.str().c_str(), NULL, 0, NULL, NULL));

  //std::cout << "prog is " << prog << "code is {" << code.str().c_str() << "}" << std::endl;

  // Compile kernel
  CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- ATTEMPTING TO COMPILE JIT SOURCE ----------\n");
  CeedDebug(ceed, "Source:\n%s\n", code.str().c_str());
  CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- END OF JIT SOURCE ----------\n");

  //nvrtcResult result = nvrtcCompileProgram(prog, num_opts + num_jit_source_dirs + num_jit_defines, opts);





  /*for (CeedInt i = 0; i < num_jit_source_dirs; i++) {
    CeedCallBackend(CeedFree(&opts[num_opts + i]));
  }
  for (CeedInt i = 0; i < num_jit_defines; i++) {
    CeedCallBackend(CeedFree(&opts[num_opts + num_jit_source_dirs + i]));
  }
  CeedCallBackend(CeedFree(&opts));
  *is_compile_good = result == NVRTC_SUCCESS;
  if (!*is_compile_good) {
    char  *log;
    size_t log_size;

    CeedCallNvrtc(ceed, nvrtcGetProgramLogSize(prog, &log_size));
    CeedCallBackend(CeedMalloc(log_size, &log));
    CeedCallNvrtc(ceed, nvrtcGetProgramLog(prog, log));
    if (throw_error) {
      return CeedError(ceed, CEED_ERROR_BACKEND, "%s\n%s", nvrtcGetErrorString(result), log);
    } else {
      CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- COMPILE ERROR DETECTED ----------\n");
      CeedDebug(ceed, "Error: %s\nCompile log:\n%s\n", nvrtcGetErrorString(result), log);
      CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- BACKEND MAY FALLBACK ----------\n");
      CeedCallBackend(CeedFree(&log));
      CeedCallNvrtc(ceed, nvrtcDestroyProgram(&prog));
      return CEED_ERROR_SUCCESS;
    }
  }

#if CUDA_VERSION >= 11010
  CeedCallNvrtc(ceed, nvrtcGetCUBINSize(prog, &ptx_size));
  CeedCallBackend(CeedMalloc(ptx_size, &ptx));
  CeedCallNvrtc(ceed, nvrtcGetCUBIN(prog, ptx));
#else
  CeedCallNvrtc(ceed, nvrtcGetPTXSize(prog, &ptx_size));
  CeedCallBackend(CeedMalloc(ptx_size, &ptx));
  CeedCallNvrtc(ceed, nvrtcGetPTX(prog, ptx));
#endif
  CeedCallNvrtc(ceed, nvrtcDestroyProgram(&prog));*/
/////////////////////////////
    const char* full_filename = "temp-jit.cu";
    FILE* file = fopen(full_filename, "w");
    if (!file) {
        perror("Failed to create file");
        return 1;
    }
    fputs(code.str().c_str(), file);
    fclose(file);

    // Compile command

    int err;

    //err = system("clang++ -c temp-jit.cu -DCEED_RUNNING_JIT_PASS=1 -I/home/alma4974/spur/libCEED/include --cuda-gpu-arch=sm_80  -default-device -o kern.o -flto=thin -fuse-ld=lld");

    //err = system("clang++ -c temp-jit.cu -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -Wl,-rpath,/home/alma4974/spur/libCEED/lib -I/home/alma4974/spur/libCEED/include -L../../lib -lceed -DCEED_RUNNING_JIT_PASS=1 --cuda-gpu-arch=sm_80 --cuda-device-only -default-device -o kern.o -L . -lbruhh -flto=thin -fuse-ld=lld");

    err = system("clang++ -flto=thin --cuda-gpu-arch=sm_80 -I/home/alma4974/spur/libCEED/include --cuda-device-only -emit-llvm -S temp-jit.cu -o kern.ll -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -Wl,-rpath,/home/alma4974/spur/libCEED/lib -I/home/alma4974/spur/libCEED/include -L../../lib -lceed");

    if(err){
        printf("Failed task 1\n");
        abort();
    }

    //system("/usr/local/cuda/bin/ptxas -m64 --gpu-name sm_80 kern.ptx -o kern.elf");
    err = system("llvm-link kern.ll --internalize --only-needed -S -o kern2.ll ");


    printf("HERE\n");

    //err = system("clang++ -S kern.o -L/usr/local/cuda/lib64 -lcudart_static -flto=thin -fuse-ld=lld -o kern.ptx");
    //err = system("clang++ kern.o -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -Wl,-rpath,/home/alma4974/spur/libCEED/lib -I/home/alma4974/spur/libCEED/include -L../../lib -lceed -DCEED_RUNNING_JIT_PASS=1 --cuda-gpu-arch=sm_80 -default-device -L . -lbruhh -flto=thin -fuse-ld=lld");
    //err = system("llc kern.ll -mcpu=sm_80 -o kern.ptx");

    if(err){
        printf("Failed task 2\n");
        abort();
    }

    err = system("opt --passes internalize,inline kern2.ll -o kern3.ll");

    if(err){
        printf("Failed task 3\n");
        abort();
    }

    err = system("llc -O3 -mcpu=sm_80 kern3.ll -o kern.ptx");

    if(err){
        printf("Failed task 4\n");
        abort();
    }

    //system("clang++ -x cuda kern.ptx --cuda-gpu-arch=sm_80 -L . -lbruhh -flto=thin -fuse-ld=lld --cuda-device-only -o kern.elf");

    //system("ptxas kern.ll -o output.ptx");

    //ofstream out("correct-output.ptx");
    //out.write(ptx, ptx_size);
    //out.close();


    ifstream           ptxfile("kern.ptx");

    ostringstream      sstr;
    sstr << ptxfile.rdbuf();
    auto ptx_data = sstr.str();

    ptx_size = ptx_data.length();

    //std::cout << "THING is " << sstr.str() << std::endl;

    printf("JITTED = %d\n", ptx_size);
    CeedCallCuda(ceed, cuModuleLoadData(module, ptx_data.c_str()));
    //printf("JITTED = %s\n", ptx_data);
  CeedCallBackend(CeedFree(&ptx_data));
  return CEED_ERROR_SUCCESS;
}

int CeedCompile_Cuda(Ceed ceed, const char *source, CUmodule *module, const CeedInt num_defines, ...) {
  bool    is_compile_good = true;
  va_list args;

  va_start(args, num_defines);
  const CeedInt ierr = CeedCompileCore_Cuda(ceed, source, true, &is_compile_good, module, num_defines, args);

  va_end(args);
  CeedCallBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

int CeedTryCompile_Cuda(Ceed ceed, const char *source, bool *is_compile_good, CUmodule *module, const CeedInt num_defines, ...) {
  va_list args;

  va_start(args, num_defines);
  const CeedInt ierr = CeedCompileCore_Cuda(ceed, source, false, is_compile_good, module, num_defines, args);

  va_end(args);
  CeedCallBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get CUDA kernel
//------------------------------------------------------------------------------
int CeedGetKernel_Cuda(Ceed ceed, CUmodule module, const char *name, CUfunction *kernel) {

    CeedCallCuda(ceed, cuModuleGetFunction(kernel, module, name));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel with block size selected automatically based on the kernel
//     (which may use enough registers to require a smaller block size than the
//      hardware is capable)
//------------------------------------------------------------------------------
int CeedRunKernelAutoblockCuda(Ceed ceed, CUfunction kernel, size_t points, void **args) {
  int min_grid_size, max_block_size;

  CeedCallCuda(ceed, cuOccupancyMaxPotentialBlockSize(&min_grid_size, &max_block_size, kernel, NULL, 0, 0x10000));
  CeedCallBackend(CeedRunKernel_Cuda(ceed, kernel, CeedDivUpInt(points, max_block_size), max_block_size, args));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel
//------------------------------------------------------------------------------
int CeedRunKernel_Cuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size, void **args) {
  CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, kernel, NULL, grid_size, block_size, 1, 1, 0, args));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel for spatial dimension
//------------------------------------------------------------------------------
int CeedRunKernelDim_Cuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y, const int block_size_z,
                          void **args) {
  CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, kernel, NULL, grid_size, block_size_x, block_size_y, block_size_z, 0, args));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel for spatial dimension with shared memory
//------------------------------------------------------------------------------
static int CeedRunKernelDimSharedCore_Cuda(Ceed ceed, CUfunction kernel, CUstream stream, const int grid_size, const int block_size_x,
                                           const int block_size_y, const int block_size_z, const int shared_mem_size, const bool throw_error,
                                           bool *is_good_run, void **args) {
#if CUDA_VERSION >= 9000
  cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_mem_size);
#endif
  CUresult result = cuLaunchKernel(kernel, grid_size, 1, 1, block_size_x, block_size_y, block_size_z, shared_mem_size, stream, args, NULL);

  if (result == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) {
    *is_good_run = false;
    if (throw_error) {
      int max_threads_per_block, shared_size_bytes, num_regs;

      cuFuncGetAttribute(&max_threads_per_block, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel);
      cuFuncGetAttribute(&shared_size_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
      cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: max_threads_per_block %d on block size (%d,%d,%d), shared_size %d, num_regs %d",
                       max_threads_per_block, block_size_x, block_size_y, block_size_z, shared_size_bytes, num_regs);
    }
  } else CeedChk_Cu(ceed, result);
  return CEED_ERROR_SUCCESS;
}

int CeedRunKernelDimShared_Cuda(Ceed ceed, CUfunction kernel, CUstream stream, const int grid_size, const int block_size_x, const int block_size_y,
                                const int block_size_z, const int shared_mem_size, void **args) {
  bool is_good_run = true;

  CeedCallBackend(CeedRunKernelDimSharedCore_Cuda(ceed, kernel, stream, grid_size, block_size_x, block_size_y, block_size_z, shared_mem_size, true,
                                                  &is_good_run, args));
  return CEED_ERROR_SUCCESS;
}

int CeedTryRunKernelDimShared_Cuda(Ceed ceed, CUfunction kernel, CUstream stream, const int grid_size, const int block_size_x, const int block_size_y,
                                   const int block_size_z, const int shared_mem_size, bool *is_good_run, void **args) {
  CeedCallBackend(CeedRunKernelDimSharedCore_Cuda(ceed, kernel, stream, grid_size, block_size_x, block_size_y, block_size_z, shared_mem_size, false,
                                                  is_good_run, args));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
