// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_MAGMA_COMMON_DEFS_H
#define CEED_MAGMA_COMMON_DEFS_H

#ifdef CEED_MAGMA_USE_HIP
#define MAGMA_DEVICE_SHARED(type, name) HIP_DYNAMIC_SHARED(type, name)
#else
#define MAGMA_DEVICE_SHARED(type, name) extern __shared__ type name[];
#endif

typedef enum {
    MagmaNoTrans       = 111,
    MagmaTrans         = 112,
    MagmaConjTrans     = 113,
    Magma_ConjTrans    = MagmaConjTrans
} magma_trans_t;

#define MAGMA_D_ZERO              0.0
#define MAGMA_D_ONE               1.0

#define MAGMA_CEILDIV(A, B)    ( ((A) + (B) - 1) / (B) )
#define MAGMA_ROUNDUP(A, B)    MAGMA_CEILDIV((A), (B)) * (B)
#define MAGMA_MAX(A, B)        ((A) > (B) ? (A) : (B))

#endif // CEED_MAGMA_COMMON_DEFS_H
