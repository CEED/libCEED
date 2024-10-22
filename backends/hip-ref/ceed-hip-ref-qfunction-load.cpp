// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <string.h>

#include <iostream>
#include <sstream>

#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Build QFunction kernel
//------------------------------------------------------------------------------
extern "C" int CeedQFunctionBuildKernel_Hip_ref(CeedQFunction qf) {
  using std::ostringstream;
  using std::string;

  Ceed                ceed;
  Ceed_Hip           *ceed_Hip;
  CeedInt             num_input_fields, num_output_fields, size;
  CeedQFunctionField *input_fields, *output_fields;
  CeedQFunction_Hip  *data;

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_Hip));
  CeedCallBackend(CeedQFunctionGetData(qf, (void **)&data));

  // QFunction is built
  if (data->QFunction) return CEED_ERROR_SUCCESS;

  // QFunction kernel generation
  CeedCallBackend(CeedQFunctionGetFields(qf, &num_input_fields, &input_fields, &num_output_fields, &output_fields));

  // Build strings for final kernel
  string        qfunction_name(data->qfunction_name);
  string        kernel_name = "CeedKernelHipRefQFunction_" + qfunction_name;
  ostringstream code;

  // Definitions
  code << "// QFunction source\n";
  code << "#include <ceed/jit-source/hip/hip-ref-qfunction.h>\n\n";
  {
    const char *source_path;

    CeedCallBackend(CeedQFunctionGetSourcePath(qf, &source_path));
    CeedCheck(source_path, ceed, CEED_ERROR_BACKEND, "No QFunction source or hipFunction_t provided.");

    code << "// User QFunction source\n";
    code << "#include \"" << source_path << "\"\n\n";
  }
  code << "extern \"C\" __launch_bounds__(BLOCK_SIZE)\n";
  code << "__global__ void " << kernel_name << "(void *ctx, CeedInt Q, Fields_Hip fields) {\n";

  // Inputs
  code << "  // Input fields\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetSize(input_fields[i], &size));
    code << "  const CeedInt size_input_" << i << " = " << size << ";\n";
    code << "  CeedScalar input_" << i << "[size_input_" << i << "];\n";
  }
  code << "  const CeedScalar *inputs[" << CeedIntMax(num_input_fields, 1) << "];\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "  inputs[" << i << "] = input_" << i << ";\n";
  }
  code << "\n";

  // Outputs
  code << "  // Output fields\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetSize(output_fields[i], &size));
    code << "  const CeedInt size_output_" << i << " = " << size << ";\n";
    code << "  CeedScalar output_" << i << "[size_output_" << i << "];\n";
  }
  code << "  CeedScalar *outputs[" << CeedIntMax(num_output_fields, 1) << "];\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "  outputs[" << i << "] = output_" << i << ";\n";
  }
  code << "\n";

  // Loop over quadrature points
  code << "  // Loop over quadrature points\n";
  code << "  for (CeedInt q = blockIdx.x * blockDim.x + threadIdx.x; q < Q; q += blockDim.x * gridDim.x) {\n";

  // Load inputs
  code << "    // -- Load inputs\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "    readQuads<size_input_" << i << ">(q, Q, fields.inputs[" << i << "], input_" << i << ");\n";
  }
  code << "\n";

  // QFunction
  code << "    // -- Call QFunction\n";
  code << "    " << qfunction_name << "(ctx, 1, inputs, outputs);\n\n";

  // Write outputs
  code << "    // -- Write outputs\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    code << "    writeQuads<size_output_" << i << ">(q, Q, output_" << i << ", fields.outputs[" << i << "]);\n";
  }
  code << "  }\n";
  code << "}\n";

  // Compile kernel
  CeedCallBackend(CeedCompile_Hip(ceed, code.str().c_str(), &data->module, 1, "BLOCK_SIZE", ceed_Hip->opt_block_size));
  CeedCallBackend(CeedGetKernel_Hip(ceed, data->module, kernel_name.c_str(), &data->QFunction));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
