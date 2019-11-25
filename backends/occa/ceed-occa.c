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
#define CEED_DEBUG_COLOR 10
#include "ceed-occa.h"

// *****************************************************************************
// * OCCA modes, default device_id is 0, but can be changed with /ocl/occa/1
// *****************************************************************************
static const char *occaCPU = "mode: 'Serial'";
static const char *occaOMP = "mode: 'OpenMP'";
static const char *occaGPU = "mode: 'CUDA', device_id: %d";
static const char *occaOCL = "mode: 'OpenCL', platform_id: 0, device_id: %d";

// *****************************************************************************
// * CeedError_Occa
// *****************************************************************************
static int CeedError_Occa(Ceed ceed,
                          const char *file, int line,
                          const char *func, int code,
                          const char *format, va_list args) {
  fprintf(stderr, "CEED-OCCA error @ %s:%d %s\n", file, line, func);
  vfprintf(stderr, format, args);
  fprintf(stderr,"\n");
  fflush(stderr);
  abort();
  return code;
}

// *****************************************************************************
// * CeedDestroy_Occa
// *****************************************************************************
static int CeedDestroy_Occa(Ceed ceed) {
  int ierr;
  Ceed_Occa *data;
  ierr = CeedGetData(ceed, (void *)&data); CeedChk(ierr);
  dbg("[CeedDestroy]");
  ierr = CeedFree(&data->occa_cache_dir); CeedChk(ierr);
  occaFree(data->device);
  ierr = CeedFree(&data->libceed_dir); CeedChk(ierr);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedDebugImpl256
// *****************************************************************************
void CeedDebugImpl256(const Ceed ceed,
                      const unsigned char color,
                      const char *format,...) {
  const Ceed_Occa *data;
  CeedGetData(ceed, (void *)&data);
  if (!data->debug) return;
  va_list args;
  va_start(args, format);
  fflush(stdout);
  fprintf(stdout,"\033[38;5;%dm",color);
  vfprintf(stdout,format,args);
  fprintf(stdout,"\033[m");
  fprintf(stdout,"\n");
  fflush(stdout);
  va_end(args);
}

// *****************************************************************************
// * CeedDebugImpl
// *****************************************************************************
void CeedDebugImpl(const Ceed ceed,
                   const char *format,...) {
  const Ceed_Occa *data;
  CeedGetData(ceed, (void *)&data);
  if (!data->debug) return;
  va_list args;
  va_start(args, format);
  CeedDebugImpl256(ceed,0,format,args);
  va_end(args);
}


// *****************************************************************************
// * INIT
// *****************************************************************************
static int CeedInit_Occa(const char *resource, Ceed ceed) {
  int ierr;
  Ceed_Occa *data;
  const int nrc = 9; // number of characters in resource
  const bool cpu = !strncmp(resource,"/cpu/occa",nrc);
  const bool omp = !strncmp(resource,"/omp/occa",nrc);
  const bool ocl = !strncmp(resource,"/ocl/occa",nrc);
  const bool gpu = !strncmp(resource,"/gpu/occa",nrc);
  const int rlen = strlen(resource);
  const bool slash = (rlen>nrc)?resource[nrc]=='/'?true:false:false;
  const int deviceID = slash?(rlen>nrc+1)?atoi(&resource[nrc+1]):0:0;
  // Warning: "backend cannot use resource" is used to grep in test/tap.sh
  if (!cpu && !omp && !ocl && !gpu)
    return CeedError(ceed, 1, "OCCA backend cannot use resource: %s", resource);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Error",
                                CeedError_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate",
                                CeedVectorCreate_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate",
                                CeedElemRestrictionCreate_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateBlocked",
                                CeedElemRestrictionCreateBlocked_Occa);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Occa); CeedChk(ierr);
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  // push env variables CEED_DEBUG or DBG to our data
  data->debug=!!getenv("CEED_DEBUG") || !!getenv("DBG");
  // push ocl to our data, to be able to check it later for the kernels
  data->ocl = ocl;
  data->libceed_dir = NULL;
  data->occa_cache_dir = NULL;
  if (data->debug) {
    occaPropertiesSet(occaSettings(), "device/verbose", occaBool(1));
    occaPropertiesSet(occaSettings(), "kernel/verbose", occaBool(1));
  }
  ierr = CeedSetData(ceed, (void *)&data); CeedChk(ierr);
  // Now that we can dbg, output resource and deviceID
  dbg("[CeedInit] resource: %s", resource);
  dbg("[CeedInit] deviceID: %d", deviceID);
  const char *mode_format = gpu?occaGPU : omp?occaOMP : ocl ? occaOCL : occaCPU;
  char mode[CEED_MAX_RESOURCE_LEN] = {0};
  // Push deviceID for CUDA and OpenCL mode
  if (ocl || gpu) sprintf(mode,mode_format,deviceID);
  else memcpy(mode,mode_format,strlen(mode_format));
  dbg("[CeedInit] mode: %s", mode);
  // Now creating OCCA device
  data->device = occaCreateDevice(occaString(mode));
  const char *deviceMode = occaDeviceMode(data->device);
  dbg("[CeedInit] returned deviceMode: %s", deviceMode);
  // Warning: "OCCA backend failed" is used to grep in test/tap.sh
  if (cpu && strcmp(occaDeviceMode(data->device), "Serial"))
    return CeedError(ceed,1, "OCCA backend failed to use Serial resource");
  if (omp && strcmp(occaDeviceMode(data->device), "OpenMP"))
    return CeedError(ceed,1, "OCCA backend failed to use OpenMP resource");
  if (gpu && strcmp(occaDeviceMode(data->device), "CUDA"))
    return CeedError(ceed,1, "OCCA backend failed to use CUDA resource");
  if (ocl && strcmp(occaDeviceMode(data->device), "OpenCL"))
    return CeedError(ceed,1, "OCCA backend failed to use OpenCL resource");
  // populating our data struct with libceed_dir
  ierr = CeedOklDladdr_Occa(ceed); CeedChk(ierr);
  if (data->libceed_dir)
    dbg("[CeedInit] libceed_dir: %s", data->libceed_dir);
  // populating our data struct with occa_cache_dir
  char occa_cache_home[OCCA_PATH_MAX];
  const char *HOME = getenv("HOME");
  if (!HOME) return CeedError(ceed, 1, "Cannot get env HOME");
  ierr = sprintf(occa_cache_home,"%s/.occa",HOME); CeedChk(!ierr);
  const char *OCCA_CACHE_DIR = getenv("OCCA_CACHE_DIR");
  const char *occa_cache_dir = OCCA_CACHE_DIR?OCCA_CACHE_DIR:occa_cache_home;
  const int occa_cache_dir_len = strlen(occa_cache_dir);
  ierr = CeedCalloc(occa_cache_dir_len+1,&data->occa_cache_dir); CeedChk(ierr);
  memcpy(data->occa_cache_dir,occa_cache_dir,occa_cache_dir_len+1);
  dbg("[CeedInit] occa_cache_dir: %s", data->occa_cache_dir);
  return 0;
}

// *****************************************************************************
// * REGISTER
// *****************************************************************************
__attribute__((constructor))
static void Register(void) {
  CeedRegister("/cpu/occa", CeedInit_Occa, 20);
  CeedRegister("/gpu/occa", CeedInit_Occa, 20);
  CeedRegister("/omp/occa", CeedInit_Occa, 20);
  CeedRegister("/ocl/occa", CeedInit_Occa, 20);
}
