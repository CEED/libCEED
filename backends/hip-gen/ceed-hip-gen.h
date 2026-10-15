// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
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

CEED_INTERN int CeedGenModuleRetire_Hip_gen(Ceed ceed, hipModule_t *module);

typedef enum { CEED_HIP_GEN_TENSOR_SLICES, CEED_HIP_GEN_TENSOR_REGISTERS, CEED_HIP_GEN_TENSOR_PREFETCHED_SLICES } CeedTensorPipeline_Hip_gen;

typedef struct {
  bool          use_fallback, use_assembly_fallback;
  CeedInt       dim;
  CeedInt       Q, Q_1d;
  CeedInt       max_P_1d;
  CeedInt       thread_1d;
  hipStream_t   streams[CEED_COMPOSITE_MAX];
  hipModule_t   module, module_assemble_full, module_assemble_diagonal, module_assemble_qfunction, pending_module;
  hipFunction_t op, assemble_full, assemble_diagonal, assemble_qfunction;
  FieldsInt_Hip indices;
  Fields_Hip    fields;
  Fields_Hip    B;
  Fields_Hip    G;
  CeedScalar   *W;
  Points_Hip    points;

  CeedInt                    scratch_planes;
  CeedTensorPipeline_Hip_gen tensor_pipeline;
} CeedOperator_Hip_gen;

typedef struct {
  const char *qfunction_name;
  void       *d_c;
} CeedQFunction_Hip_gen;

CEED_INTERN int CeedQFunctionCreate_Hip_gen(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Hip_gen(CeedOperator op);
