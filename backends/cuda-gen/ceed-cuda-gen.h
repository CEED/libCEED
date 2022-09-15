// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_cuda_gen_h
#define _ceed_cuda_gen_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cuda.h>
#include "../cuda/ceed-cuda-common.h"

typedef struct { const CeedScalar *inputs[CEED_FIELD_MAX]; CeedScalar *outputs[CEED_FIELD_MAX]; } CudaFields;
typedef struct { CeedInt *inputs[CEED_FIELD_MAX]; CeedInt *outputs[CEED_FIELD_MAX]; } CudaFieldsInt;

typedef struct {
  CeedInt dim;
  CeedInt Q_1d;
  CeedInt max_P_1d;
  CUmodule module;
  CUfunction op;
  CudaFieldsInt indices;
  CudaFields fields;
  CudaFields B;
  CudaFields G;
  CeedScalar *W;
} CeedOperator_Cuda_gen;

typedef struct {
  char *q_function_name;
  char *q_function_source;
  void *d_c;
} CeedQFunction_Cuda_gen;

CEED_INTERN int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Cuda_gen(CeedOperator op);

#endif // _ceed_cuda_gen_h
