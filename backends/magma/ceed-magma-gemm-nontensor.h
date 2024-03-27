// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include "ceed-magma.h"

////////////////////////////////////////////////////////////////////////////////
CEED_INTERN int magma_gemm_nontensor(magma_trans_t trans_A, magma_trans_t trans_B, magma_int_t m, magma_int_t n, magma_int_t k, CeedScalar alpha,
                                     const CeedScalar *d_A, magma_int_t ldda, const CeedScalar *d_B, magma_int_t lddb, CeedScalar beta,
                                     CeedScalar *d_C, magma_int_t lddc, magma_queue_t queue);
