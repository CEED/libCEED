// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/cpu-gen/cpu-gen-types.h>
#include <stdbool.h>

typedef struct {
  CeedInt block_size;
  char   *cxx;
} Ceed_Cpu_Gen;

typedef struct {
  bool                    use_fallback;
  void                   *handle;
  const char             *op_function_name;
  CeedElemRestriction     inputs_block_elem_rstr[CEED_FIELD_MAX];
  CeedElemRestriction     outputs_block_elem_rstr[CEED_FIELD_MAX];
  InputFieldData_Cpu_Gen  inputs[CEED_FIELD_MAX];
  OutputFieldData_Cpu_Gen outputs[CEED_FIELD_MAX];
} CeedOperator_Cpu_Gen;

CEED_INTERN int CeedOperatorCreate_Cpu_Gen(CeedOperator op);
