// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-hip-compile.h"

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>
#include <hip/hiprtc.h>
#include <stdarg.h>
#include <string.h>

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
  } while (0);

//------------------------------------------------------------------------------
// Compile HIP kernel
//------------------------------------------------------------------------------
int CeedCompileHip(Ceed ceed, const char *source, hipModule_t *module, const CeedInt num_defines, ...) {
  hipFree(0);  // Make sure a Context exists for hiprtc
  hiprtcProgram prog;

  std::ostringstream code;

  // Add hip runtime include statement for generation if runtime < 40400000
  // (implies ROCm < 4.5)
  int runtime_version;
  CeedCallHip(ceed, hipRuntimeGetVersion(&runtime_version));
  if (runtime_version < 40400000) {
    code << "\n#include <hip/hip_runtime.h>\n";
  }
  // With ROCm 4.5, need to include these definitions specifically for hiprtc
  // (but cannot include the runtime header)
  else {
    code << "#include <stddef.h>\n";
    code << "#define __forceinline__ inline __attribute__((always_inline))\n";
    code << "#define HIP_DYNAMIC_SHARED(type, var) extern __shared__ type var[];\n";
  }

  // Kernel specific options, such as kernel constants
  if (num_defines > 0) {
    va_list args;
    va_start(args, num_defines);
    char *name;
    int   val;
    for (int i = 0; i < num_defines; i++) {
      name = va_arg(args, char *);
      val  = va_arg(args, int);
      code << "#define " << name << " " << val << "\n";
    }
    va_end(args);
  }

  // Standard libCEED definitions for HIP backends
  char *jit_defs_path, *jit_defs_source;
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-jit.h", &jit_defs_path));
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, jit_defs_path, &jit_defs_source));
  code << jit_defs_source;
  code << "\n\n";
  CeedCallBackend(CeedFree(&jit_defs_path));
  CeedCallBackend(CeedFree(&jit_defs_source));

  // Non-macro options
  const int   num_opts = 3;
  const char *opts[num_opts];
  opts[0] = "-default-device";
  struct hipDeviceProp_t prop;
  Ceed_Hip              *ceed_data;
  CeedCallBackend(CeedGetData(ceed, (void **)&ceed_data));
  CeedCallHip(ceed, hipGetDeviceProperties(&prop, ceed_data->device_id));
  std::string arch_arg = "--gpu-architecture=" + std::string(prop.gcnArchName);
  opts[1]              = arch_arg.c_str();
  opts[2]              = "-munsafe-fp-atomics";

  // Add string source argument provided in call
  code << source;

  // Create Program
  CeedCallHiprtc(ceed, hiprtcCreateProgram(&prog, code.str().c_str(), NULL, 0, NULL, NULL));

  // Compile kernel
  hiprtcResult result = hiprtcCompileProgram(prog, num_opts, opts);
  if (result != HIPRTC_SUCCESS) {
    size_t log_size;
    CeedChk_hiprtc(ceed, hiprtcGetProgramLogSize(prog, &log_size));
    char *log;
    CeedCallBackend(CeedMalloc(log_size, &log));
    CeedCallHiprtc(ceed, hiprtcGetProgramLog(prog, log));
    return CeedError(ceed, CEED_ERROR_BACKEND, "%s\n%s", hiprtcGetErrorString(result), log);
  }

  size_t ptx_size;
  CeedCallHiprtc(ceed, hiprtcGetCodeSize(prog, &ptx_size));
  char *ptx;
  CeedCallBackend(CeedMalloc(ptx_size, &ptx));
  CeedCallHiprtc(ceed, hiprtcGetCode(prog, ptx));
  CeedCallHiprtc(ceed, hiprtcDestroyProgram(&prog));

  CeedCallHip(ceed, hipModuleLoadData(module, ptx));
  CeedCallBackend(CeedFree(&ptx));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get HIP kernel
//------------------------------------------------------------------------------
int CeedGetKernelHip(Ceed ceed, hipModule_t module, const char *name, hipFunction_t *kernel) {
  CeedCallHip(ceed, hipModuleGetFunction(kernel, module, name));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run HIP kernel
//------------------------------------------------------------------------------
int CeedRunKernelHip(Ceed ceed, hipFunction_t kernel, const int grid_size, const int block_size, void **args) {
  CeedCallHip(ceed, hipModuleLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, NULL, args, NULL));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run HIP kernel for spatial dimension
//------------------------------------------------------------------------------
int CeedRunKernelDimHip(Ceed ceed, hipFunction_t kernel, const int grid_size, const int block_size_x, const int block_size_y, const int block_size_z,
                        void **args) {
  CeedCallHip(ceed, hipModuleLaunchKernel(kernel, grid_size, 1, 1, block_size_x, block_size_y, block_size_z, 0, NULL, args, NULL));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run HIP kernel for spatial dimension with shared memory
//------------------------------------------------------------------------------
int CeedRunKernelDimSharedHip(Ceed ceed, hipFunction_t kernel, const int grid_size, const int block_size_x, const int block_size_y,
                              const int block_size_z, const int shared_mem_size, void **args) {
  CeedCallHip(ceed, hipModuleLaunchKernel(kernel, grid_size, 1, 1, block_size_x, block_size_y, block_size_z, shared_mem_size, NULL, args, NULL));
  return CEED_ERROR_SUCCESS;
}
