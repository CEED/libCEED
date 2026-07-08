// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// This header does not have guards because it is included multiple times.
// This will be expanded inside CeedRegisterAll() to call each registration function.
// This is also used to create weakly linked registration functions in `backends/weak/ceed-*-weak.c'.

CEED_BACKEND(CeedRegister_Magma, 2, "/gpu/cuda/magma", "/gpu/hip/magma")
CEED_BACKEND(CeedRegister_Magma_Det, 2, "/gpu/cuda/magma/det", "/gpu/hip/magma/det")
