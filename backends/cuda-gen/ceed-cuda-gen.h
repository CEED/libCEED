// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_CUDA_GEN_H
#define CEED_CUDA_GEN_H

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/cuda/cuda-types.h>
#include <cuda.h>

typedef struct {
  CeedInt        dim;
  CeedInt        Q_1d;
  CeedInt        max_P_1d;
  CUmodule       module;
  CUfunction     op;
  FieldsInt_Cuda indices;
  Fields_Cuda    fields;
  Fields_Cuda    B;
  Fields_Cuda    G;
  CeedScalar    *W;
} CeedOperator_Cuda_gen;

typedef struct {
  const char *qfunction_name;
  const char *qfunction_source;
  void       *d_c;
} CeedQFunction_Cuda_gen;

CEED_INTERN int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Cuda_gen(CeedOperator op);

#endif  // CEED_CUDA_GEN_H
