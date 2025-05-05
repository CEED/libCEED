// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include "ceed-backend-init.h"

#define CEED_PRIORITY_UNCOMPILED 1024  // Revisit if we ever use 4 digit priority values

// LCOV_EXCL_START
// This function provides improved error messages for uncompiled backends
static int CeedInit_Uncompiled(const char *resource, Ceed ceed) {
  return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend not currently compiled: %s\nConsult the installation instructions to compile this backend",
                   resource);
}
// LCOV_EXCL_STOP

//------------------------------------------------------------------------------
// Native CPU Backends
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Ref_Serial(void) {
  const char   *prefix   = "/cpu/self/ref/serial";
  const CeedInt priority = 50;

  return
      //! [Register]
      CeedRegister(prefix, CeedInit_Ref_Serial, priority);
  //! [Register]
}

CEED_INTERN int CeedRegister_Ref_Blocked(void) {
  const char   *prefix   = "/cpu/self/ref/blocked";
  const CeedInt priority = 55;

  return CeedRegister(prefix, CeedInit_Ref_Blocked, priority);
}

CEED_INTERN int CeedRegister_Opt_Blocked(void) {
  const char   *prefix   = "/cpu/self/opt/blocked";
  const CeedInt priority = 40;

  return CeedRegister(prefix, CeedInit_Opt_Blocked, priority);
}

CEED_INTERN int CeedRegister_Opt_Serial(void) {
  const char   *prefix   = "/cpu/self/opt/serial";
  const CeedInt priority = 45;

  return CeedRegister(prefix, CeedInit_Opt_Serial, priority);
}

//------------------------------------------------------------------------------
// Memcheck Backends
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Memcheck_Blocked(void) {
  const char *prefix = "/cpu/self/memcheck/blocked";
#ifdef CEED_USE_MEMCHECK
  const CeedInt priority = 110;
  return CeedRegister(prefix, CeedInit_Memcheck_Blocked, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

CEED_INTERN int CeedRegister_Memcheck_Serial(void) {
  const char *prefix = "/cpu/self/memcheck/serial";
#ifdef CEED_USE_MEMCHECK
  const CeedInt priority = 100;
  return CeedRegister(prefix, CeedInit_Memcheck_Serial, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

//------------------------------------------------------------------------------
// AVX Backends
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Avx_Blocked(void) {
  const char *prefix = "/cpu/self/avx/blocked";
#ifdef CEED_USE_AVX
  const CeedInt priority = 30;
  return CeedRegister(prefix, CeedInit_Avx_Blocked, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

CEED_INTERN int CeedRegister_Avx_Serial(void) {
  const char *prefix = "/cpu/self/avx/serial";
#ifdef CEED_USE_AVX
  const CeedInt priority = 35;
  return CeedRegister(prefix, CeedInit_Avx_Serial, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

//------------------------------------------------------------------------------
// XSMM Backends
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Xsmm_Blocked(void) {
  const char *prefix = "/cpu/self/xsmm/blocked";
#ifdef CEED_USE_XSMM
  const CeedInt priority = 20;
  return CeedRegister(prefix, CeedInit_Xsmm_Blocked, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

CEED_INTERN int CeedRegister_Xsmm_Serial(void) {
  const char *prefix = "/cpu/self/xsmm/serial";
#ifdef CEED_USE_XSMM
  const CeedInt priority = 25;
  return CeedRegister(prefix, CeedInit_Xsmm_Serial, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

//------------------------------------------------------------------------------
// Cuda Backends
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Cuda_Ref(void) {
  const char *prefix = "/gpu/cuda/ref";
#ifdef CEED_USE_CUDA
  const CeedInt priority = 40;
  return CeedRegister(prefix, CeedInit_Cuda_Ref, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

CEED_INTERN int CeedRegister_Cuda_Shared(void) {
  const char *prefix = "/gpu/cuda/shared";
#ifdef CEED_USE_CUDA
  const CeedInt priority = 25;
  return CeedRegister(prefix, CeedInit_Cuda_Shared, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

CEED_INTERN int CeedRegister_Cuda_Gen(void) {
  const char *prefix = "/gpu/cuda/gen";
#ifdef CEED_USE_CUDA
  const CeedInt priority = 20;
  return CeedRegister(prefix, CeedInit_Cuda_Gen, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

//------------------------------------------------------------------------------
// Hip Backends
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Hip_Ref(void) {
  const char *prefix = "/gpu/hip/ref";
#ifdef CEED_USE_HIP
  const CeedInt priority = 40;
  return CeedRegister(prefix, CeedInit_Hip_Ref, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

CEED_INTERN int CeedRegister_Hip_Shared(void) {
  const char *prefix = "/gpu/hip/shared";
#ifdef CEED_USE_HIP
  const CeedInt priority = 25;
  return CeedRegister(prefix, CeedInit_Hip_Shared, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

CEED_INTERN int CeedRegister_Hip_Gen(void) {
  const char *prefix = "/gpu/hip/gen";
#ifdef CEED_USE_HIP
  const CeedInt priority = 20;
  return CeedRegister(prefix, CeedInit_Hip_Gen, priority);
#else
  CeedDebugEnv("Weak Register   : %s", prefix);
  return CeedRegister(prefix, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

//------------------------------------------------------------------------------
// Sycl Backends
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Sycl_Ref(void) {
  const char *prefix_cpu = "/cpu/sycl/ref";
  const char *prefix_gpu = "/gpu/sycl/ref";
#ifdef CEED_USE_SYCL
  const CeedInt priority_cpu = 50;
  const CeedInt priority_gpu = 40;
  CeedRegister(prefix_cpu, CeedInit_Sycl_Ref, priority_cpu);
  return CeedRegister(prefix_gpu, CeedInit_Sycl_Ref, priority_gpu);
#else
  CeedDebugEnv("Weak Register   : %s", prefix_cpu);
  CeedRegister(prefix_cpu, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
  CeedDebugEnv("Weak Register   : %s", prefix_gpu);
  return CeedRegister(prefix_gpu, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

CEED_INTERN int CeedRegister_Sycl_Shared(void) {
  const char *prefix_cpu = "/cpu/sycl/shared";
  const char *prefix_gpu = "/gpu/sycl/shared";
#ifdef CEED_USE_SYCL
  const CeedInt priority_cpu = 35;
  const CeedInt priority_gpu = 25;
  CeedRegister(prefix, CeedInit_Sycl_Shared, priority_cpu);
  return CeedRegister(prefix, CeedInit_Sycl_Shared, priority_gpu);
#else
  CeedDebugEnv("Weak Register   : %s", prefix_cpu);
  CeedRegister(prefix_cpu, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
  CeedDebugEnv("Weak Register   : %s", prefix_gpu);
  return CeedRegister(prefix_gpu, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

CEED_INTERN int CeedRegister_Sycl_Gen(void) {
  const char *prefix_gpu = "/gpu/sycl/gen";
#ifdef CEED_USE_SYCL
  const CeedInt priority_gpu = 20;
  return CeedRegister(prefix_gpu, CeedInit_Sycl_Gen, priority_gpu);
#else
  CeedDebugEnv("Weak Register   : %s", prefix_gpu);
  return CeedRegister(prefix_gpu, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif
}

//------------------------------------------------------------------------------
// Magma Backends
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Magma(void) {
  const char *prefix_cuda = "/gpu/cuda/magma";
  const char *prefix_hip  = "/gpu/hip/magma";
#ifdef CEED_USE_MAGMA
  const CeedInt priority = 120;
#ifdef CEED_MAGMA_USE_HIP
  CeedRegister(prefix_hip, CeedInit_Magma, priority);
  CeedDebugEnv("Weak Register   : %s", prefix_cuda);
  return CeedRegister(prefix_cuda, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#else
  CeedDebugEnv("Weak Register   : %s", prefix_hip);
  CeedRegister(prefix_hip, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
  return CeedRegister(prefix_cuda, CeedInit_Magma, priority);
#endif  // CEED_MAGMA_USE_HIP
#else   // CEED_USE_MAGMA
  CeedDebugEnv("Weak Register   : %s", prefix_hip);
  CeedRegister(prefix_hip, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
  CeedDebugEnv("Weak Register   : %s", prefix_cuda);
  return CeedRegister(prefix_cuda, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif  // CEED_USE_MAGMA
}

CEED_INTERN int CeedRegister_Magma_Det(void) {
  const char *prefix_cuda = "/gpu/cuda/magma/det";
  const char *prefix_hip  = "/gpu/hip/magma/det";
#ifdef CEED_USE_MAGMA
  const CeedInt priority = 125;
#ifdef CEED_MAGMA_USE_HIP
  CeedRegister(prefix_hip, CeedInit_Magma_Det, priority);
  CeedDebugEnv("Weak Register   : %s", prefix_cuda);
  return CeedRegister(prefix_cuda, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#else
  CeedDebugEnv("Weak Register   : %s", prefix_hip);
  CeedRegister(prefix_hip, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
  return CeedRegister(prefix_cuda, CeedInit_Magma_Det, priority);
#endif  // CEED_MAGMA_USE_HIP
#else   // CEED_USE_MAGMA
  CeedDebugEnv("Weak Register   : %s", prefix_hip);
  CeedRegister(prefix_hip, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
  CeedDebugEnv("Weak Register   : %s", prefix_cuda);
  return CeedRegister(prefix_cuda, CeedInit_Uncompiled, CEED_PRIORITY_UNCOMPILED);
#endif  // CEED_USE_MAGMA
}

//------------------------------------------------------------------------------
