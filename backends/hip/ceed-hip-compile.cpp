// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <sstream>
#include <stdarg.h>
#include <string.h>
#include <hip/hiprtc.h>
#include "ceed-hip-common.h"
#include "ceed-hip-compile.h"

#define CeedChk_hiprtc(ceed, x) \
do { \
  hiprtcResult result = static_cast<hiprtcResult>(x); \
  if (result != HIPRTC_SUCCESS) \
    return CeedError((ceed), CEED_ERROR_BACKEND, hiprtcGetErrorString(result)); \
} while (0)

//------------------------------------------------------------------------------
// Compile HIP kernel
//------------------------------------------------------------------------------
int CeedCompileHip(Ceed ceed, const char *source, hipModule_t *module,
                    const CeedInt num_opts, ...) {
  int ierr;
  hipFree(0); // Make sure a Context exists for hiprtc 
  hiprtcProgram prog;

  std::ostringstream code;
  // Add hip runtime include statement for generation if runtime < 40400000
  // (implies ROCm < 4.5)
  int runtime_version;
  CeedChk_Hip(ceed, hipRuntimeGetVersion(&runtime_version));
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

  // Macro definitions
  // Get kernel specific options, such as kernel constants
  const int opts_size = 2;
  const char *opts[opts_size];
  if (num_opts > 0) {
    va_list args;
    va_start(args, num_opts);
    char *name;
    int val;
    for (int i = 0; i < num_opts; i++) {
      name = va_arg(args, char *);
      val = va_arg(args, int);
      code << "#define " << name << " " << val << "\n";
    }
    va_end(args);
  }

  // Standard backend options
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) { 
    code << "#define CeedScalar float\n";
  }
  else {
    code << "#define CeedScalar double\n";
  }
  code << "#define CeedInt int\n";
  code << "#define CEED_ERROR_SUCCESS 0\n\n";
 
  // Non-macro options     
  opts[0] = "-default-device";
  struct hipDeviceProp_t prop;
  Ceed_Hip *ceed_data;
  ierr = CeedGetData(ceed, (void **)&ceed_data); CeedChkBackend(ierr);
  CeedChk_Hip(ceed, hipGetDeviceProperties(&prop, ceed_data->device_id));
  std::string arch_arg = "--gpu-architecture="  + std::string(prop.gcnArchName);
  opts[1] = arch_arg.c_str();

  // Add string source argument provided in call
  code << source;

  // Create Program
  CeedChk_hiprtc(ceed, hiprtcCreateProgram(&prog, code.str().c_str(), NULL, 0, NULL, NULL));

  // Compile kernel
  hiprtcResult result = hiprtcCompileProgram(prog, opts_size, opts);
  if (result != HIPRTC_SUCCESS) {
    size_t log_size;
    CeedChk_hiprtc(ceed, hiprtcGetProgramLogSize(prog, &log_size));
    char *log;
    ierr = CeedMalloc(log_size, &log); CeedChkBackend(ierr);
    CeedChk_hiprtc(ceed, hiprtcGetProgramLog(prog, log));
    return CeedError(ceed, CEED_ERROR_BACKEND, "%s\n%s",
                     hiprtcGetErrorString(result), log);
  }

  size_t ptx_size;
  CeedChk_hiprtc(ceed, hiprtcGetCodeSize(prog, &ptx_size));
  char *ptx;
  ierr = CeedMalloc(ptx_size, &ptx); CeedChkBackend(ierr);
  CeedChk_hiprtc(ceed, hiprtcGetCode(prog, ptx));
  CeedChk_hiprtc(ceed, hiprtcDestroyProgram(&prog));

  CeedChk_Hip(ceed, hipModuleLoadData(module, ptx));
  ierr = CeedFree(&ptx); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get HIP kernel
//------------------------------------------------------------------------------
int CeedGetKernelHip(Ceed ceed, hipModule_t module, const char *name,
                      hipFunction_t *kernel) {

  CeedChk_Hip(ceed, hipModuleGetFunction(kernel, module, name));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run HIP kernel
//------------------------------------------------------------------------------
int CeedRunKernelHip(Ceed ceed, hipFunction_t kernel, const int grid_size,
                      const int block_size, void **args) {
  CeedChk_Hip(ceed, hipModuleLaunchKernel(kernel, grid_size, 1, 1, block_size, 1,
                                  1, 0, NULL, args, NULL));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run HIP kernel for spatial dimension
//------------------------------------------------------------------------------
int CeedRunKernelDimHip(Ceed ceed, hipFunction_t kernel, const int grid_size,
                         const int block_size_x, const int block_size_y,
                         const int block_size_z, void **args) {
  CeedChk_Hip(ceed, hipModuleLaunchKernel(kernel, grid_size, 1, 1,
                                  block_size_x, block_size_y, block_size_z,
                                  0, NULL, args, NULL));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run HIP kernel for spatial dimension with shared memory
//------------------------------------------------------------------------------
int CeedRunKernelDimSharedHip(Ceed ceed, hipFunction_t kernel, const int grid_size,
                               const int block_size_x, const int block_size_y,
                               const int block_size_z, const int shared_mem_size,
                               void **args) {
  CeedChk_Hip(ceed, hipModuleLaunchKernel(kernel, grid_size, 1, 1,
                                  block_size_x, block_size_y, block_size_z,
                                  shared_mem_size, NULL, args, NULL));
  return CEED_ERROR_SUCCESS;
}
