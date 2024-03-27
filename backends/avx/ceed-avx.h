// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_AVX_H
#define CEED_AVX_H

#include <ceed.h>
#include <ceed/backend.h>

CEED_INTERN int CeedTensorContractCreate_Avx(CeedTensorContract contract);

#endif  // CEED_AVX_H
