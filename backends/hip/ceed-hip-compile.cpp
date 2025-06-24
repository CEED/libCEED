// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-hip-compile.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <stdarg.h>
#include <string.h>
#include <hip/hiprtc.h>

#include <sstream>

#include "ceed-hip-common.h"

#define CeedChk_hiprtc(ceed, x)                                                                               \
  do {                                                                                                        \
    hiprtcResult result = static_cast<hiprtcResult>(x);                                                       \
    if (result != HIPRTC_SUCCESS) return CeedError((ceed), CEED_ERROR_BACKEND, hiprtcGetErrorString(result)); \
  } while (0)

#define CeedCallHiprtc(ceed, ...)  \
  do {                             \
    int ierr_q_ = __VA_ARGS__;     \
    CeedChk_hiprtc(ceed, ierr_q_); \
  } while (0)

//------------------------------------------------------------------------------
// Compile HIP kernel
//------------------------------------------------------------------------------
static int CeedCompileCore_Hip(Ceed ceed, const char *source, const bool throw_error, bool *is_compile_good, hipModule_t *module,
                               const CeedInt num_defines, va_list args) {
  size_t                 ptx_size;
  char                  *ptx;
  const int              num_opts            = 4;
  CeedInt                num_jit_source_dirs = 0, num_jit_defines = 0;
  const char           **opts;
  int                    runtime_version;
  hiprtcProgram          prog;
  struct hipDeviceProp_t prop;
  Ceed_Hip              *ceed_data;

  hipFree(0);  // Make sure a Context exists for hiprtc

  std::ostringstream code;

  // Add hip runtime include statement for generation if runtime < 40400000 (implies ROCm < 4.5)
  CeedCallHip(ceed, hipRuntimeGetVersion(&runtime_version));
  if (runtime_version < 40400000) {
    code << "\n#include <hip/hip_runtime.h>\n";
  }
  // With ROCm 4.5, need to include these definitions specifically for hiprtc (but cannot include the runtime header)
  else {
    code << "#include <stddef.h>\n";
    code << "#define __forceinline__ inline __attribute__((always_inline))\n";
    code << "#define HIP_DYNAMIC_SHARED(type, var) extern __shared__ type var[];\n";
  }

  // Kernel specific options, such as kernel constants
  if (num_defines > 0) {
    char *name;
    int   val;

    for (int i = 0; i < num_defines; i++) {
      name = va_arg(args, char *);
      val  = va_arg(args, int);
      code << "#define " << name << " " << val << "\n";
    }
  }

  // Standard libCEED definitions for HIP backends
  code << "#include <ceed/jit-source/hip/hip-jit.h>\n\n";

  // Non-macro options
  CeedCallBackend(CeedCalloc(num_opts, &opts));
  opts[0] = "-default-device";
  CeedCallBackend(CeedGetData(ceed, (void **)&ceed_data));
  CeedCallHip(ceed, hipGetDeviceProperties(&prop, ceed_data->device_id));
  std::string arch_arg = "--gpu-architecture=" + std::string(prop.gcnArchName);
  opts[1]              = arch_arg.c_str();
  opts[2]              = "-munsafe-fp-atomics";
  opts[3]              = "-DCEED_RUNNING_JIT_PASS=1";
  // Additional include dirs
  {
    const char **jit_source_dirs;

    CeedCallBackend(CeedGetJitSourceRoots(ceed, &num_jit_source_dirs, &jit_source_dirs));
    CeedCallBackend(CeedRealloc(num_opts + num_jit_source_dirs, &opts));
    for (CeedInt i = 0; i < num_jit_source_dirs; i++) {
      std::ostringstream include_dir_arg;

      include_dir_arg << "-I" << jit_source_dirs[i];
      CeedCallBackend(CeedStringAllocCopy(include_dir_arg.str().c_str(), (char **)&opts[num_opts + i]));
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
    }
    CeedCallBackend(CeedRestoreJitDefines(ceed, &jit_defines));
  }

  // Add string source argument provided in call
  code << source;

  // Create Program
  CeedCallHiprtc(ceed, hiprtcCreateProgram(&prog, code.str().c_str(), NULL, 0, NULL, NULL));

  // Compile kernel
  CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- ATTEMPTING TO COMPILE JIT SOURCE ----------\n");
  CeedDebug(ceed, "Source:\n%s\n", code.str().c_str());
  CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- END OF JIT SOURCE ----------\n");
  if (CeedDebugFlag(ceed)) {
    // LCOV_EXCL_START
    CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- JiT COMPILER OPTIONS ----------\n");
    for (CeedInt i = 0; i < num_opts + num_jit_source_dirs + num_jit_defines; i++) {
      CeedDebug(ceed, "Option %d: %s", i, opts[i]);
    }
    CeedDebug(ceed, "");
    CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- END OF JiT COMPILER OPTIONS ----------\n");
    // LCOV_EXCL_STOP
  }
  hiprtcResult result = hiprtcCompileProgram(prog, num_opts + num_jit_source_dirs + num_jit_defines, opts);

  for (CeedInt i = 0; i < num_jit_source_dirs; i++) {
    CeedCallBackend(CeedFree(&opts[num_opts + i]));
  }
  for (CeedInt i = 0; i < num_jit_defines; i++) {
    CeedCallBackend(CeedFree(&opts[num_opts + num_jit_source_dirs + i]));
  }
  CeedCallBackend(CeedFree(&opts));
  *is_compile_good = result == HIPRTC_SUCCESS;
  if (!*is_compile_good) {
    size_t log_size;
    char  *log;

    CeedChk_hiprtc(ceed, hiprtcGetProgramLogSize(prog, &log_size));
    CeedCallBackend(CeedMalloc(log_size, &log));
    CeedCallHiprtc(ceed, hiprtcGetProgramLog(prog, log));
    if (throw_error) {
      return CeedError(ceed, CEED_ERROR_BACKEND, "%s\n%s", hiprtcGetErrorString(result), log);
    } else {
      CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- COMPILE ERROR DETECTED ----------\n");
      CeedDebug(ceed, "Error: %s\nCompile log:\n%s\n", hiprtcGetErrorString(result), log);
      CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- BACKEND MAY FALLBACK ----------\n");
      CeedCallBackend(CeedFree(&log));
      CeedCallHiprtc(ceed, hiprtcDestroyProgram(&prog));
      return CEED_ERROR_SUCCESS;
    }
  }

  CeedCallHiprtc(ceed, hiprtcGetCodeSize(prog, &ptx_size));
  CeedCallBackend(CeedMalloc(ptx_size, &ptx));
  CeedCallHiprtc(ceed, hiprtcGetCode(prog, ptx));
  CeedCallHiprtc(ceed, hiprtcDestroyProgram(&prog));

  CeedCallHip(ceed, hipModuleLoadData(module, ptx));
  CeedCallBackend(CeedFree(&ptx));
  return CEED_ERROR_SUCCESS;
}

int CeedCompile_Hip(Ceed ceed, const char *source, hipModule_t *module, const CeedInt num_defines, ...) {
  bool    is_compile_good = true;
  va_list args;

  va_start(args, num_defines);
  const CeedInt ierr = CeedCompileCore_Hip(ceed, source, true, &is_compile_good, module, num_defines, args);

  va_end(args);
  CeedCallBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

int CeedTryCompile_Hip(Ceed ceed, const char *source, bool *is_compile_good, hipModule_t *module, const CeedInt num_defines, ...) {
  va_list args;

  va_start(args, num_defines);
  const CeedInt ierr = CeedCompileCore_Hip(ceed, source, false, is_compile_good, module, num_defines, args);

  va_end(args);
  CeedCallBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get HIP kernel
//------------------------------------------------------------------------------
int CeedGetKernel_Hip(Ceed ceed, hipModule_t module, const char *name, hipFunction_t *kernel) {
  CeedCallHip(ceed, hipModuleGetFunction(kernel, module, name));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run HIP kernel
//------------------------------------------------------------------------------
int CeedRunKernel_Hip(Ceed ceed, hipFunction_t kernel, const int grid_size, const int block_size, void **args) {
  CeedCallHip(ceed, hipModuleLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, NULL, args, NULL));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run HIP kernel for spatial dimension
//------------------------------------------------------------------------------
int CeedRunKernelDim_Hip(Ceed ceed, hipFunction_t kernel, const int grid_size, const int block_size_x, const int block_size_y, const int block_size_z,
                         void **args) {
  CeedCallHip(ceed, hipModuleLaunchKernel(kernel, grid_size, 1, 1, block_size_x, block_size_y, block_size_z, 0, NULL, args, NULL));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run HIP kernel for spatial dimension with shared memory
//------------------------------------------------------------------------------
static int CeedRunKernelDimSharedCore_Hip(Ceed ceed, hipFunction_t kernel, hipStream_t stream, const int grid_size, const int block_size_x,
                                          const int block_size_y, const int block_size_z, const int shared_mem_size, const bool throw_error,
                                          bool *is_good_run, void **args) {
  hipError_t result = hipModuleLaunchKernel(kernel, grid_size, 1, 1, block_size_x, block_size_y, block_size_z, shared_mem_size, stream, args, NULL);

  *is_good_run = result == hipSuccess;
  if (throw_error) CeedCallHip(ceed, result);
  return CEED_ERROR_SUCCESS;
}

int CeedRunKernelDimShared_Hip(Ceed ceed, hipFunction_t kernel, hipStream_t stream, const int grid_size, const int block_size_x,
                               const int block_size_y, const int block_size_z, const int shared_mem_size, void **args) {
  bool is_good_run = true;

  CeedCallBackend(CeedRunKernelDimSharedCore_Hip(ceed, kernel, stream, grid_size, block_size_x, block_size_y, block_size_z, shared_mem_size, true,
                                                 &is_good_run, args));
  return CEED_ERROR_SUCCESS;
}

int CeedTryRunKernelDimShared_Hip(Ceed ceed, hipFunction_t kernel, hipStream_t stream, const int grid_size, const int block_size_x,
                                  const int block_size_y, const int block_size_z, const int shared_mem_size, bool *is_good_run, void **args) {
  CeedCallBackend(CeedRunKernelDimSharedCore_Hip(ceed, kernel, stream, grid_size, block_size_x, block_size_y, block_size_z, shared_mem_size, false,
                                                 is_good_run, args));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
