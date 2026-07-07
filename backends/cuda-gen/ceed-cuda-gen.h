// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/cuda/cuda-types.h>
#include <cuda.h>
#include <cuda_runtime.h>


typedef struct {
  bool           use_fallback, use_assembly_fallback;
  CeedInt        dim;
  CeedInt        Q, Q_1d;
  CeedInt        max_P_1d;
  CeedInt        thread_1d;
  CUmodule       module, module_assemble_full, module_assemble_diagonal, module_assemble_qfunction;
  CUfunction     op, assemble_full, assemble_diagonal, assemble_qfunction;
  FieldsInt_Cuda indices;
  Fields_Cuda    fields;
  Fields_Cuda    B;
  Fields_Cuda    G;
  CeedScalar    *W;
  Points_Cuda    points;
  
  // CUDA graph state
  bool              graph_created;
  bool              warmup_done;
  cudaGraph_t       graph;
  cudaGraphExec_t   graph_instance;
  int               graph_launches;
  int               fallbacks;
  const CeedScalar *captured_input_ptr;  // device address at capture; checked before each replay
} CeedOperator_Cuda_gen;

typedef struct {
  const char *qfunction_name;
  void       *d_c;
} CeedQFunction_Cuda_gen;

CEED_INTERN int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Cuda_gen(CeedOperator op);
