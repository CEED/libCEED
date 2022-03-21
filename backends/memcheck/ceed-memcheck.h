// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_memcheck_h
#define _ceed_memcheck_h

#include <ceed/ceed.h>
#include <ceed/backend.h>

typedef struct {
  const CeedScalar **inputs;
  CeedScalar **outputs;
  bool setup_done;
} CeedQFunction_Memcheck;

CEED_INTERN int CeedQFunctionCreate_Memcheck(CeedQFunction qf);

#endif // _ceed_memcheck_h
