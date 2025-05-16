// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// This header does not have guards because it is included multiple times.

// List each backend function suffix once here.
// The CEED_BACKEND(name, suffix, is_enabled, num_prefixes, prefix_0, priority_0, ...) macro is used in several places.
//
// This list will be expanded inside ceed-backend-init.h to declare all initialization functions of the form CeedInit_##name(resource, ceed).
// These functions must be defined in their respective backends.
//
// This list will be expanded inside ceed-backend-register.h to declare all initialization functions of the form CeedRegister_##name##suffix(void).
// These functions are defined ceed-register.c.
//
// In ceed-register.c, if the backend is enabled, then CeedRegister_##name##suffix(void) registers each prefix with the corresponding priority and uses CeedInit_##name.
// If the backend is not enabled, then CeedRegister_##name##suffix(void) registers each prefix with an initialization function that gives an error message.
//
// This list is expanded inside CeedRegisterAll() to call each registration function in the order listed.

CEED_BACKEND(Ref_Serial, , true, 1, "/cpu/self/ref/serial", 50)
CEED_BACKEND(Ref_Blocked, , true, 1, "/cpu/self/ref/blocked", 55)
CEED_BACKEND(Opt_Serial, , true, 1, "/cpu/self/opt/serial", 40)
CEED_BACKEND(Opt_Blocked, , true, 1, "/cpu/self/opt/blocked", 45)

#ifndef CEED_USE_MEMCHECK
#define CEED_USE_MEMCHECK false
#endif
CEED_BACKEND(Memcheck_Serial, , CEED_USE_MEMCHECK, 1, "/cpu/self/memcheck/serial", 100)
CEED_BACKEND(Memcheck_Blocked, , CEED_USE_MEMCHECK, 1, "/cpu/self/memcheck/blocked", 110)

#ifndef CEED_USE_AVX
#define CEED_USE_AVX false
#endif
CEED_BACKEND(Avx_Serial, , CEED_USE_AVX, 1, "/cpu/self/avx/serial", 30)
CEED_BACKEND(Avx_Blocked, , CEED_USE_AVX, 1, "/cpu/self/avx/blocked", 30)

#ifndef CEED_USE_XSMM
#define CEED_USE_XSMM false
#endif
CEED_BACKEND(Xsmm_Serial, , CEED_USE_XSMM, 1, "/cpu/self/xsmm/serial", 20)
CEED_BACKEND(Xsmm_Blocked, , CEED_USE_XSMM, 1, "/cpu/self/xsmm/blocked", 25)

#ifndef CEED_USE_CUDA
#define CEED_USE_CUDA false
#endif
CEED_BACKEND(Cuda_Ref, , CEED_USE_CUDA, 1, "/gpu/cuda/ref", 40)
CEED_BACKEND(Cuda_Shared, , CEED_USE_CUDA, 1, "/gpu/cuda/shared", 30)
CEED_BACKEND(Cuda_Gen, , CEED_USE_CUDA, 1, "/gpu/cuda/gen", 20)

#ifndef CEED_USE_HIP
#define CEED_USE_HIP false
#endif
CEED_BACKEND(Hip_Ref, , CEED_USE_HIP, 1, "/gpu/hip/ref", 40)
CEED_BACKEND(Hip_Shared, , CEED_USE_HIP, 1, "/gpu/hip/shared", 30)
CEED_BACKEND(Hip_Gen, , CEED_USE_HIP, 1, "/gpu/hip/gen", 20)

#ifndef CEED_USE_SYCL
#define CEED_USE_SYCL false
#endif
CEED_BACKEND(Sycl_Ref, , CEED_USE_SYCL, 2, "/gpu/sycl/ref", 40, "/cpu/sycl/ref", 45)
CEED_BACKEND(Sycl_Shared, , CEED_USE_SYCL, 2, "/gpu/sycl/shared", 30, "/cpu/sycl/shared", 35)
CEED_BACKEND(Sycl_Gen, , CEED_USE_SYCL, 1, "/gpu/sycl/gen", 20)

#ifndef CEED_MAGMA_USE_CUDA
#define CEED_MAGMA_USE_CUDA false
#endif
CEED_BACKEND(Magma, _Cuda, CEED_MAGMA_USE_CUDA, 1, "/gpu/cuda/magma", 120)
CEED_BACKEND(Magma_Det, _Cuda, CEED_MAGMA_USE_CUDA, 1, "/gpu/cuda/magma/det", 125)

#ifndef CEED_MAGMA_USE_HIP
#define CEED_MAGMA_USE_HIP false
#endif
CEED_BACKEND(Magma, _Hip, CEED_MAGMA_USE_HIP, 1, "/gpu/hip/magma", 120)
CEED_BACKEND(Magma_Det, _Hip, CEED_MAGMA_USE_HIP, 1, "/gpu/hip/magma/det", 125)
