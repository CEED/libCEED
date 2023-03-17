// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// This header does not have guards because it is included multiple times.

// List each backend registration function once here.
// This will be expanded inside CeedRegisterAll() to call each registration function in the order listed, and also to define weak symbol aliases for
// backends that are not configured.

CEED_BACKEND(CeedRegister_Avx_Blocked, 1, "/cpu/self/avx/blocked")
CEED_BACKEND(CeedRegister_Avx_Serial, 1, "/cpu/self/avx/serial")
CEED_BACKEND(CeedRegister_Cuda, 1, "/gpu/cuda/ref")
CEED_BACKEND(CeedRegister_Cuda_Gen, 1, "/gpu/cuda/gen")
CEED_BACKEND(CeedRegister_Cuda_Shared, 1, "/gpu/cuda/shared")
CEED_BACKEND(CeedRegister_Hip, 1, "/gpu/hip/ref")
CEED_BACKEND(CeedRegister_Hip_Gen, 1, "/gpu/hip/gen")
CEED_BACKEND(CeedRegister_Hip_Shared, 1, "/gpu/hip/shared")
CEED_BACKEND(CeedRegister_Magma, 2, "/gpu/cuda/magma", "/gpu/hip/magma")
CEED_BACKEND(CeedRegister_Magma_Det, 2, "/gpu/cuda/magma/det", "/gpu/hip/magma/det")
CEED_BACKEND(CeedRegister_Memcheck_Blocked, 1, "/cpu/self/memcheck/blocked")
CEED_BACKEND(CeedRegister_Memcheck_Serial, 1, "/cpu/self/memcheck/serial")
CEED_BACKEND(CeedRegister_Occa, 6, "/cpu/self/occa", "/cpu/openmp/occa", "/gpu/dpcpp/occa", "/gpu/opencl/occa", "/gpu/hip/occa", "/gpu/cuda/occa")
CEED_BACKEND(CeedRegister_Opt_Blocked, 1, "/cpu/self/opt/blocked")
CEED_BACKEND(CeedRegister_Opt_Serial, 1, "/cpu/self/opt/serial")
CEED_BACKEND(CeedRegister_Ref, 1, "/cpu/self/ref/serial")
CEED_BACKEND(CeedRegister_Ref_Blocked, 1, "/cpu/self/ref/blocked")
CEED_BACKEND(CeedRegister_Xsmm_Blocked, 1, "/cpu/self/xsmm/blocked")
CEED_BACKEND(CeedRegister_Xsmm_Serial, 1, "/cpu/self/xsmm/serial")
