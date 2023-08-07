// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_magma_sycl_h
#define _ceed_magma_sycl_h

#include <ceed.h>

#include "ceed-magma.h"

CEED_INTERN int CeedInitMagma_Sycl(Ceed ceed, int deviceID);

CEED_INTERN int CeedMagmaGetSyclHandle(Ceed ceed, void **handle);

CEED_INTERN void CeedMagmaQueueSync_Sycl(magma_queue_t queue);

CEED_INTERN int mkl_gemm_batched_strided(magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                                         const CeedScalar *dA, magma_int_t ldda, magma_int_t strideA, const CeedScalar *dB, magma_int_t lddb,
                                         magma_int_t strideB, CeedScalar beta, CeedScalar *dC, magma_int_t lddc, magma_int_t strideC,
                                         magma_int_t batchCount, magma_queue_t queue);

#endif  // _ceed_magma_sycl_h
