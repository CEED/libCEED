// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// This header does not have guards because it is included multiple times.

// List each backend function suffix once here.
// The CEED_BACKEND(name) macro is used in several places.
//
// This list will be expanded inside ceed-backend-init.h to declare all initialization functions of the form CeedInit_##name(resource, ceed).
// These functions must be defined in their respective backends.
//
// This list will be expanded inside ceed-register.c to list all initialization functions of the form CeedRegister_##name(void).
// These functions must all be defined in ceed-backends.c.
// This list will then be expanded inside CeedRegisterAll() to call each registration function in the order listed.

CEED_BACKEND(Ref_Serial)
CEED_BACKEND(Ref_Blocked)
CEED_BACKEND(Opt_Serial)
CEED_BACKEND(Opt_Blocked)

CEED_BACKEND(Memcheck_Serial)
CEED_BACKEND(Memcheck_Blocked)

CEED_BACKEND(Avx_Serial)
CEED_BACKEND(Avx_Blocked)

CEED_BACKEND(Xsmm_Serial)
CEED_BACKEND(Xsmm_Blocked)

CEED_BACKEND(Cuda_Ref)
CEED_BACKEND(Cuda_Shared)
CEED_BACKEND(Cuda_Gen)

CEED_BACKEND(Hip_Ref)
CEED_BACKEND(Hip_Shared)
CEED_BACKEND(Hip_Gen)

CEED_BACKEND(Sycl_Ref)
CEED_BACKEND(Sycl_Shared)
CEED_BACKEND(Sycl_Gen)

CEED_BACKEND(Magma)
CEED_BACKEND(Magma_Det)
