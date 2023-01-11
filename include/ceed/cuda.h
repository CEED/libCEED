// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_cuda_h
#define _ceed_cuda_h

#include <ceed/ceed.h>
#include <cuda.h>

CEED_EXTERN int CeedQFunctionSetCUDAUserFunction(CeedQFunction qf, CUfunction f);

#endif
