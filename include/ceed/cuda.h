// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_CUDA_H
#define CEED_CUDA_H

#include <ceed.h>
#include <cuda.h>

CEED_EXTERN int CeedQFunctionSetCUDAUserFunction(CeedQFunction qf, CUfunction f);

#endif  // CEED_CUDA_H
