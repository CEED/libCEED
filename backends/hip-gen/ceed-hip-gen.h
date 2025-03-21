// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/hip/hip-types.h>
#include <hip/hip_runtime.h>

typedef struct {
  bool          use_fallback;
  CeedInt       dim;
  CeedInt       Q, Q_1d;
  CeedInt       max_P_1d;
  CeedInt       thread_1d;
  hipModule_t   module;
  hipFunction_t op;
  FieldsInt_Hip indices;
  Fields_Hip    fields;
  Fields_Hip    B;
  Fields_Hip    G;
  CeedScalar   *W;
  Points_Hip    points;
} CeedOperator_Hip_gen;

typedef struct {
  const char *qfunction_name;
  void       *d_c;
} CeedQFunction_Hip_gen;

CEED_INTERN int CeedQFunctionCreate_Hip_gen(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Hip_gen(CeedOperator op);
