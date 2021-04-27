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
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ceed-cuda.h"

//------------------------------------------------------------------------------
// Compile CUDA kernel
//------------------------------------------------------------------------------
int CeedCompileCuda(Ceed ceed, const char *source, CUmodule *module,
                    const CeedInt numopts, ...) {
  int ierr;
  cudaFree(0); // Make sure a Context exists for nvrtc
  nvrtcProgram prog;
  CeedChk_Nvrtc(ceed, nvrtcCreateProgram(&prog, source, NULL, 0, NULL, NULL));

  // Get kernel specific options, such as kernel constants
  const int optslen = 32;
  const int optsextra = 4;
  const char *opts[numopts + optsextra];
  char buf[numopts][optslen];
  if (numopts > 0) {
    va_list args;
    va_start(args, numopts);
    char *name;
    int val;
    for (int i = 0; i < numopts; i++) {
      name = va_arg(args, char *);
      val = va_arg(args, int);
      snprintf(&buf[i][0], optslen,"-D%s=%d", name, val);
      opts[i] = &buf[i][0];
    }
    va_end(args);
  }

  // Standard backend options
  opts[numopts]     = "-DCeedScalar=double";
  opts[numopts + 1] = "-DCeedInt=int";
  opts[numopts + 2] = "-default-device";
  struct cudaDeviceProp prop;
  Ceed_Cuda *ceed_data;
  ierr = CeedGetData(ceed, &ceed_data); CeedChkBackend(ierr);
  ierr = cudaGetDeviceProperties(&prop, ceed_data->deviceId);
  CeedChk_Cu(ceed, ierr);
  char buff[optslen];
  snprintf(buff, optslen,"-arch=compute_%d%d", prop.major, prop.minor);
  opts[numopts + 3] = buff;

  // Compile kernel
  nvrtcResult result = nvrtcCompileProgram(prog, numopts + optsextra, opts);
  if (result != NVRTC_SUCCESS) {
    size_t logsize;
    CeedChk_Nvrtc(ceed, nvrtcGetProgramLogSize(prog, &logsize));
    char *log;
    ierr = CeedMalloc(logsize, &log); CeedChkBackend(ierr);
    CeedChk_Nvrtc(ceed, nvrtcGetProgramLog(prog, log));
    return CeedError(ceed, CEED_ERROR_BACKEND, "%s\n%s",
                     nvrtcGetErrorString(result), log);
  }

  size_t ptxsize;
  CeedChk_Nvrtc(ceed, nvrtcGetPTXSize(prog, &ptxsize));
  char *ptx;
  ierr = CeedMalloc(ptxsize, &ptx); CeedChkBackend(ierr);
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

//------------------------------------------------------------------------------
// Run CUDA kernel
//------------------------------------------------------------------------------
int CeedRunKernelCuda(Ceed ceed, CUfunction kernel, const int gridSize,
                      const int blockSize, void **args) {
  CeedChk_Cu(ceed, cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1,
                                  1, 0, NULL, args, NULL));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel for spatial dimension
//------------------------------------------------------------------------------
int CeedRunKernelDimCuda(Ceed ceed, CUfunction kernel, const int gridSize,
                         const int blockSizeX, const int blockSizeY,
                         const int blockSizeZ, void **args) {
  CeedChk_Cu(ceed, cuLaunchKernel(kernel, gridSize, 1, 1,
                                  blockSizeX, blockSizeY, blockSizeZ,
                                  0, NULL, args, NULL));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel for spatial dimension with sharde memory
//------------------------------------------------------------------------------
int CeedRunKernelDimSharedCuda(Ceed ceed, CUfunction kernel, const int gridSize,
                               const int blockSizeX, const int blockSizeY,
                               const int blockSizeZ, const int sharedMemSize,
                               void **args) {
  CeedChk_Cu(ceed, cuLaunchKernel(kernel, gridSize, 1, 1,
                                  blockSizeX, blockSizeY, blockSizeZ,
                                  sharedMemSize, NULL, args, NULL));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// CUDA preferred MemType
//------------------------------------------------------------------------------
static int CeedGetPreferredMemType_Cuda(CeedMemType *type) {
  *type = CEED_MEM_DEVICE;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedCudaInit(Ceed ceed, const char *resource, int nrc) {
  int ierr;
  const char *device_spec = strstr(resource, ":device_id=");
  const int deviceID = (device_spec) ? atoi(device_spec+11) : -1;

  int currentDeviceID;
  ierr = cudaGetDevice(&currentDeviceID); CeedChk_Cu(ceed,ierr);
  if (deviceID >= 0 && currentDeviceID != deviceID) {
    ierr = cudaSetDevice(deviceID); CeedChk_Cu(ceed,ierr);
    currentDeviceID = deviceID;
  }
  struct cudaDeviceProp deviceProp;
  ierr = cudaGetDeviceProperties(&deviceProp, currentDeviceID);
  CeedChk_Cu(ceed,ierr);

  Ceed_Cuda *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  data->deviceId = currentDeviceID;
  data->optblocksize = deviceProp.maxThreadsPerBlock;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get CUBLAS handle
//------------------------------------------------------------------------------
int CeedCudaGetCublasHandle(Ceed ceed, cublasHandle_t *handle) {
  int ierr;
  Ceed_Cuda *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);

  if (!data->cublasHandle) {
    ierr = cublasCreate(&data->cublasHandle); CeedChk_Cublas(ceed, ierr);
  }
  *handle = data->cublasHandle;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend destroy
//------------------------------------------------------------------------------
int CeedDestroy_Cuda(Ceed ceed) {
  int ierr;
  Ceed_Cuda *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  if (data->cublasHandle) {
    ierr = cublasDestroy(data->cublasHandle); CeedChk_Cublas(ceed, ierr);
  }
  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Cuda(const char *resource, Ceed ceed) {
  int ierr;
  const int nrc = 9; // number of characters in resource
  if (strncmp(resource, "/gpu/cuda/ref", nrc))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Cuda backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChk(ierr);

  Ceed_Cuda *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedSetData(ceed, data); CeedChkBackend(ierr);
  ierr = CeedCudaInit(ceed, resource, nrc); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "GetPreferredMemType",
                                CeedGetPreferredMemType_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate",
                                CeedVectorCreate_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate",
                                CeedElemRestrictionCreate_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateBlocked",
                                CeedElemRestrictionCreateBlocked_Cuda);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionContextCreate",
                                CeedQFunctionContextCreate_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "CompositeOperatorCreate",
                                CeedCompositeOperatorCreate_Cuda); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Cuda); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Cuda(void) {
  return CeedRegister("/gpu/cuda/ref", CeedInit_Cuda, 40);
}
//------------------------------------------------------------------------------
