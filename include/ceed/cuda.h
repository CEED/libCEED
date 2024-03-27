// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Public header for CUDA utility components of libCEED
#pragma once

#include <ceed.h>
#include <cuda.h>

CEED_EXTERN int CeedQFunctionSetCUDAUserFunction(CeedQFunction qf, CUfunction f);
