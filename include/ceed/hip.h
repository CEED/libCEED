// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_hip_h
#define _ceed_hip_h

#include <ceed/ceed.h>
#include <hip/hip_runtime.h>

CEED_EXTERN int CeedQFunctionSetHIPUserFunction(CeedQFunction qf, hipFunction_t f);

#endif
