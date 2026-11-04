// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
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

CEED_INTERN int CeedGenModuleRetire_Cuda_gen(Ceed ceed, CUmodule *module);

typedef enum { CEED_CUDA_GEN_TENSOR_SLICES, CEED_CUDA_GEN_TENSOR_REGISTERS, CEED_CUDA_GEN_TENSOR_PREFETCHED_SLICES } CeedTensorPipeline_Cuda_gen;

// Operator readiness freezes field roles, not vector storage or context values.
// Preparation retains their public handles; submission borrows current arrays.
typedef struct {
  CeedInt       num_elem, num_inputs, num_outputs;
  CeedQFunction qf;
  CeedVector    inputs[CEED_FIELD_MAX], outputs[CEED_FIELD_MAX], coords;
} CeedApplyBindings_Cuda_gen;

typedef struct {
  bool           use_fallback, use_assembly_fallback;
  CeedInt        dim;
  CeedInt        Q, Q_1d;
  CeedInt        max_P_1d;
  CeedInt        thread_1d;
  CUmodule       module, module_assemble_full, module_assemble_diagonal, module_assemble_qfunction, pending_module;
  CUfunction     op, assemble_full, assemble_diagonal, assemble_qfunction;
  FieldsInt_Cuda indices;
  Fields_Cuda    fields;
  Fields_Cuda    B;
  Fields_Cuda    G;
  CeedScalar    *W;
  Points_Cuda    points;

  CeedApplyBindings_Cuda_gen  apply;
  CeedTensorPipeline_Cuda_gen tensor_pipeline;
  CeedInt                     scratch_planes;
  int                         block[3], grid;
  size_t                      shared_mem;

  // Graph capture data
  bool              use_graph;
  bool              graph_created;
  bool              warmup_done;
  cudaGraph_t       graph;
  cudaGraphExec_t   graph_instance;
  const CeedScalar *captured_input_ptr;
  CeedScalar       *captured_output_ptr;
} CeedOperator_Cuda_gen;

typedef struct {
  const char *qfunction_name;
  void       *d_c;
} CeedQFunction_Cuda_gen;

CEED_INTERN int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Cuda_gen(CeedOperator op);
