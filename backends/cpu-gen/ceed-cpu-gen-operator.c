// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <dlfcn.h>

#include "ceed-cpu-compile.h"
#include "ceed-cpu-gen-operator-build.h"
#include "ceed-cpu-gen.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Cpu_Gen(CeedOperator op) {
  Ceed                  ceed;
  CeedOperator_Cpu_Gen *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  if (impl->handle) dlclose(impl->handle);
  CeedCallBackend(CeedFree(&impl->op_function_name));
  for (CeedInt i = 0; i < CEED_FIELD_MAX; i++) {
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->inputs_block_elem_rstr[i]));
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->outputs_block_elem_rstr[i]));
  }
  CeedCallBackend(CeedFree(&impl));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Cpu_Gen(CeedOperator op, CeedVector input_vec, CeedVector output_vec, CeedRequest *request) {
  bool                  is_run_good = false;
  Ceed                  ceed;
  CeedOperator_Cpu_Gen *impl;

  // Backend data
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Build operator
  if (!impl->handle && !impl->use_fallback) {
    bool is_build_good = false;

    CeedCallBackend(CeedOperatorBuildKernel_Cpu_Gen(op, &is_build_good));
    if (!is_build_good) {
      is_run_good = false;
      CeedDebug(ceed, "Build failure; falling back to /cpu/self/opt");
    }
  }

  // Try to run kernel
  if (!impl->use_fallback) {
    void                        *ctx        = NULL;
    const CeedScalar            *input_arr  = NULL;
    CeedScalar                  *output_arr = NULL;
    CeedInt                      num_input_fields, num_output_fields;
    CeedOperatorField           *op_input_fields, *op_output_fields;
    CeedQFunction                qf;
    CeedOperatorFunction_Cpu_Gen op_function;

    CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));

    // Get active l-vecs
    if (input_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(input_vec, CEED_MEM_HOST, &input_arr));
    if (output_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArray(output_vec, CEED_MEM_HOST, &output_arr));

    // Get input l-vecs
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        impl->inputs[i].l_vec = input_arr;
      } else if (vec != CEED_VECTOR_NONE) {
        CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_HOST, &impl->inputs[i].l_vec));
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }

    // Writable access to output l-vecs
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        impl->outputs[i].l_vec = output_arr;
      } else if (vec != CEED_VECTOR_NONE) {
        CeedCallBackend(CeedVectorGetArray(vec, CEED_MEM_HOST, &impl->outputs[i].l_vec));
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }

    // Context
    CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
    CeedCallBackend(CeedQFunctionGetContextData(qf, CEED_MEM_HOST, &ctx));

    // Run JiTed function
    CeedRunFunction_Cpu(ceed, impl->handle, impl->op_function_name, op_function, ctx, impl->inputs, impl->outputs);

    // Restore context
    CeedCallBackend(CeedQFunctionRestoreContextData(qf, &ctx));

    // Restore input l-vecs
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        impl->inputs[i].l_vec = NULL;
      } else if (vec != CEED_VECTOR_NONE) {
        CeedCallBackend(CeedVectorRestoreArrayRead(vec, &impl->inputs[i].l_vec));
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }

    // Restore output l-vecs
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        impl->outputs[i].l_vec = NULL;
      } else if (vec != CEED_VECTOR_NONE) {
        CeedCallBackend(CeedVectorRestoreArray(vec, &impl->outputs[i].l_vec));
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }

    // And restore active l-vecs
    if (input_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorRestoreArrayRead(input_vec, &input_arr));
    if (output_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorRestoreArray(output_vec, &output_arr));

    // Cleanup
    CeedCallBackend(CeedQFunctionDestroy(&qf));
    is_run_good = true;
  }

  // Fallback
  if (!is_run_good) {
    CeedOperator op_fallback;

    CeedDebug(CeedOperatorReturnCeed(op), "\nFalling back to /cpu/self/opt CeedOperator for ApplyAdd\n");
    CeedCallBackend(CeedOperatorGetFallback(op, &op_fallback));
    CeedCallBackend(CeedOperatorApplyAdd(op_fallback, input_vec, output_vec, request));
  }
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Create
//------------------------------------------------------------------------------
int CeedOperatorCreate_Cpu_Gen(CeedOperator op) {
  Ceed                  ceed;
  CeedOperator_Cpu_Gen *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Cpu_Gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Cpu_Gen));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Create At Points
//------------------------------------------------------------------------------
int CeedOperatorCreateAtPoints_Cpu_Gen(CeedOperator op) {
  Ceed                  ceed;
  CeedOperator_Cpu_Gen *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Cpu_Gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Cpu_Gen));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
