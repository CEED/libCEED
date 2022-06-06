// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_hip_gen_h
#define _ceed_hip_gen_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>
#include "../hip/ceed-hip-common.h"

typedef struct { const CeedScalar *in[CEED_FIELD_MAX]; CeedScalar *out[CEED_FIELD_MAX]; } HipFields;
typedef struct { CeedInt *in[CEED_FIELD_MAX]; CeedInt *out[CEED_FIELD_MAX]; } HipFieldsInt;

typedef struct {
  CeedInt dim;
  CeedInt Q1d;
  CeedInt maxP1d;
  hipModule_t module;
  hipFunction_t op;
  HipFieldsInt indices;
  HipFields fields;
  HipFields B;
  HipFields G;
  CeedScalar *W;
} CeedOperator_Hip_gen;

typedef struct {
  char *qFunctionName;
  char *qFunctionSource;
  void *d_c;
} CeedQFunction_Hip_gen;

CEED_INTERN int CeedQFunctionCreate_Hip_gen(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Hip_gen(CeedOperator op);

#endif // _ceed_hip_gen_h
