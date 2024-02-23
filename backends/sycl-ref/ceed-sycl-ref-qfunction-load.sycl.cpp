// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <sycl/sycl.hpp>
#include <vector>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref.hpp"

#define SUB_GROUP_SIZE_QF 16

//------------------------------------------------------------------------------
// Build QFunction kernel
//
// TODO: Refactor
//------------------------------------------------------------------------------
extern "C" int CeedQFunctionBuildKernel_Sycl(CeedQFunction qf) {
  Ceed                ceed;
  Ceed_Sycl          *data;
  const char         *read_write_kernel_path, *read_write_kernel_source;
  const char         *qfunction_name, *qfunction_source;
  CeedInt             num_input_fields, num_output_fields;
  CeedQFunctionField *input_fields, *output_fields;
  CeedQFunction_Sycl *impl;

  CeedCallBackend(CeedQFunctionGetData(qf, (void **)&impl));
  // QFunction is built
  if (impl->QFunction) return CEED_ERROR_SUCCESS;

  CeedQFunctionGetCeed(qf, &ceed);
  CeedCallBackend(CeedGetData(ceed, &data));

  // QFunction kernel generation
  CeedCallBackend(CeedQFunctionGetFields(qf, &num_input_fields, &input_fields, &num_output_fields, &output_fields));

  std::vector<CeedInt> input_sizes(num_input_fields);
  CeedQFunctionField  *input_i = input_fields;

  for (auto &size_i : input_sizes) {
    CeedCallBackend(CeedQFunctionFieldGetSize(*input_i, &size_i));
    ++input_i;
  }

  std::vector<CeedInt> output_sizes(num_output_fields);
  CeedQFunctionField  *output_i = output_fields;

  for (auto &size_i : output_sizes) {
    CeedCallBackend(CeedQFunctionFieldGetSize(*output_i, &size_i));
    ++output_i;
  }

  CeedCallBackend(CeedQFunctionGetKernelName(qf, &qfunction_name));

  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading QFunction User Source -----\n");
  CeedCallBackend(CeedQFunctionLoadSourceToBuffer(qf, &qfunction_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading QFunction User Source Complete! -----\n");

  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/sycl/sycl-ref-qfunction.h", &read_write_kernel_path));

  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading QFunction Read/Write Kernel Source -----\n");
  {
    char *tmp;
    CeedCallBackend(CeedLoadSourceToBuffer(ceed, read_write_kernel_path, &tmp));
    read_write_kernel_source = tmp;
  }
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading QFunction Read/Write Kernel Source Complete! -----\n");

  std::string_view  qf_name_view(qfunction_name);
  std::string_view  qf_source_view(qfunction_source);
  std::string_view  rw_source_view(read_write_kernel_source);
  const std::string kernel_name = "CeedKernelSyclRefQFunction_" + std::string(qf_name_view);

  // Defintions
  std::ostringstream code;
  code << rw_source_view;
  code << qf_source_view;
  code << "\n";

  // Kernel function
  // Here we are fixing a lower sub-group size value to avoid register spills
  // This needs to be revisited if all qfunctions require this.
  code << "__attribute__((intel_reqd_sub_group_size(" << SUB_GROUP_SIZE_QF << "))) __kernel void " << kernel_name
       << "(__global void *ctx, CeedInt Q,\n";

  // OpenCL doesn't allow for structs with pointers.
  // We will need to pass all of the arguments individually.
  // Input parameters
  for (CeedInt i = 0; i < num_input_fields; ++i) {
    code << "  "
         << "__global const CeedScalar *in_" << i << ",\n";
  }

  // Output parameters
  code << "  "
       << "__global CeedScalar *out_0";
  for (CeedInt i = 1; i < num_output_fields; ++i) {
    code << "\n,  "
         << "__global CeedScalar *out_" << i;
  }
  // Begin kernel function body
  code << ") {\n\n";

  // Inputs
  code << "  // Input fields\n";
  for (CeedInt i = 0; i < num_input_fields; ++i) {
    code << "  CeedScalar U_" << i << "[" << input_sizes[i] << "];\n";
  }
  code << "  const CeedScalar *inputs[" << num_input_fields << "] = {U_0";
  for (CeedInt i = 1; i < num_input_fields; i++) {
    code << ", U_" << i << "\n";
  }
  code << "};\n\n";

  // Outputs
  code << "  // Output fields\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "  CeedScalar V_" << i << "[" << output_sizes[i] << "];\n";
  }
  code << "  CeedScalar *outputs[" << num_output_fields << "] = {V_0";
  for (CeedInt i = 1; i < num_output_fields; i++) {
    code << ", V_" << i << "\n";
  }
  code << "};\n\n";

  code << "  const CeedInt q = get_global_linear_id();\n\n";

  code << "if(q < Q){ \n\n";

  // Load inputs
  code << "  // -- Load inputs\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "  readQuads(" << input_sizes[i] << ", Q, q, "
         << "in_" << i << ", U_" << i << ");\n";
  }
  code << "\n";

  // QFunction
  code << "  // -- Call QFunction\n";
  code << "  " << qf_name_view << "(ctx, 1, inputs, outputs);\n\n";

  // Write outputs
  code << "  // -- Write outputs\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "  writeQuads(" << output_sizes[i] << ", Q, q, "
         << "V_" << i << ", out_" << i << ");\n";
  }
  code << "\n";

  // End kernel function body
  code << "}\n";
  code << "}\n";

  // View kernel for debugging
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Generated QFunction Kernels:\n");
  CeedDebug(ceed, code.str().c_str());

  // Compile kernel
  CeedCallBackend(CeedBuildModule_Sycl(ceed, code.str(), &impl->sycl_module));
  CeedCallBackend(CeedGetKernel_Sycl(ceed, impl->sycl_module, kernel_name, &impl->QFunction));

  // Cleanup
  CeedCallBackend(CeedFree(&qfunction_source));
  CeedCallBackend(CeedFree(&read_write_kernel_path));
  CeedCallBackend(CeedFree(&read_write_kernel_source));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
