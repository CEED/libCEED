// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_HIP_H
#define CEED_HIP_H

#include <ceed.h>
#include <hip/hip_runtime.h>

CEED_EXTERN int CeedQFunctionSetHIPUserFunction(CeedQFunction qf, hipFunction_t f);

#endif  // CEED_HIP_H
