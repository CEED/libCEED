// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// This header does not have guards because it is included multiple times.

// List each backend registration function once here. This will be expanded
// inside CeedRegisterAll() to call each registration function in the order
// listed, and also to define weak symbol aliases for backends that are not
// configured.

MACRO(CeedRegister_Avx_Blocked, 1, "/cpu/self/avx/blocked")
MACRO(CeedRegister_Avx_Serial, 1, "/cpu/self/avx/serial")
MACRO(CeedRegister_Cuda, 1, "/gpu/cuda/ref")
MACRO(CeedRegister_Cuda_Gen, 1, "/gpu/cuda/gen")
MACRO(CeedRegister_Cuda_Shared, 1, "/gpu/cuda/shared")
MACRO(CeedRegister_Hip, 1, "/gpu/hip/ref")
MACRO(CeedRegister_Hip_Gen, 1, "/gpu/hip/gen")
MACRO(CeedRegister_Hip_Shared, 1, "/gpu/hip/shared")
MACRO(CeedRegister_Magma, 2, "/gpu/cuda/magma", "/gpu/hip/magma")
MACRO(CeedRegister_Magma_Det, 2, "/gpu/cuda/magma/det", "/gpu/hip/magma/det")
MACRO(CeedRegister_Memcheck_Blocked, 1, "/cpu/self/memcheck/blocked")
MACRO(CeedRegister_Memcheck_Serial, 1, "/cpu/self/memcheck/serial")
MACRO(CeedRegister_Opt_Blocked, 1, "/cpu/self/opt/blocked")
MACRO(CeedRegister_Opt_Serial, 1, "/cpu/self/opt/serial")
MACRO(CeedRegister_Ref, 1, "/cpu/self/ref/serial")
MACRO(CeedRegister_Ref_Blocked, 1, "/cpu/self/ref/blocked")
MACRO(CeedRegister_Xsmm_Blocked, 1, "/cpu/self/xsmm/blocked")
MACRO(CeedRegister_Xsmm_Serial, 1, "/cpu/self/xsmm/serial")
