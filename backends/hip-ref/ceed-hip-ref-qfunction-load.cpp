// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include "ceed-hip-ref.h"
#include "../hip/ceed-hip-compile.h"

//------------------------------------------------------------------------------
// Build QFunction kernel
//------------------------------------------------------------------------------
extern "C" int CeedHipBuildQFunction(CeedQFunction qf) {
  CeedInt ierr;
  using std::ostringstream;
  using std::string;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  Ceed_Hip *ceed_Hip;
  ierr = CeedGetData(ceed, &ceed_Hip); CeedChkBackend(ierr);
  CeedQFunction_Hip *data;
  ierr = CeedQFunctionGetData(qf, (void **)&data); CeedChkBackend(ierr);

  // QFunction is built
  if (data->QFunction)
    return CEED_ERROR_SUCCESS;

  if (!data->qfunction_source)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "No QFunction source or hipFunction_t provided.");
  // LCOV_EXCL_STOP
  
  // QFunction kernel generation
  CeedInt num_input_fields, num_output_fields, size;
  CeedQFunctionField *input_fields, *output_fields;
  ierr = CeedQFunctionGetFields(qf, &num_input_fields, &input_fields,
                                &num_output_fields, &output_fields);
  CeedChkBackend(ierr);

  // Build strings for final kernel
  char *read_write_kernel_path, *read_write_kernel_source;
  ierr = CeedPathConcatenate(ceed, __FILE__, "kernels/hip-ref-qfunction.h",
                             &read_write_kernel_path); CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading QFunction Read/Write Kernel Source -----\n");
  ierr = CeedLoadSourceToBuffer(ceed, read_write_kernel_path, &read_write_kernel_source);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading QFunction Read/Write Kernel Source Complete! -----\n");
  string qfunction_source(data->qfunction_source);
  string qfunction_name(data->qfunction_name);
  string read_write(read_write_kernel_source);
  string kernel_name = "CeedKernel_Hip_ref_" + qfunction_name;
  ostringstream code;

  // Defintions
  code << "\n#define CEED_QFUNCTION(name) inline __device__ int name\n";
  code << "#define CEED_QFUNCTION_HELPER inline __device__ __forceinline__\n";
  code << "#define CeedPragmaSIMD\n";
  code << "#define CEED_ERROR_SUCCESS 0\n";
  code << "#define CEED_Q_VLA 1\n\n";
  code << "typedef struct { const CeedScalar* inputs[16]; CeedScalar* outputs[16]; } Fields_Hip;\n";
  code << read_write;
  code << qfunction_source;
  code << "\n";
  code << "extern \"C\" __launch_bounds__(BLOCK_SIZE)\n";
  code << "__global__ void " << kernel_name << "(void *ctx, CeedInt Q, Fields_Hip fields) {\n";
  
  // Inputs
  code << "  // Input fields\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    ierr = CeedQFunctionFieldGetSize(input_fields[i], &size); CeedChkBackend(ierr);
    code << "  const CeedInt size_input_" << i << " = " << size << ";\n";
    code << "  CeedScalar input_" << i << "[size_input_" << i << "];\n";
  }
  code << "  const CeedScalar* inputs[" << num_input_fields << "];\n";
  for (CeedInt i = 0; i < num_input_fields; i++) {
    code << "  inputs[" << i << "] = input_" << i << ";\n";
  }
  code << "\n";

  // Outputs
  code << "  // Output fields\n";
  for (CeedInt i = 0; i < num_output_fields; i++) {
    ierr = CeedQFunctionFieldGetSize(output_fields[i], &size); CeedChkBackend(ierr);
    code << "  const CeedInt size_output_" << i << " = " << size << ";\n";
    code << "  CeedScalar output_" << i << "[size_output_" << i << "];\n";
  }
  code << "  CeedScalar* outputs[" << num_output_fields << "];\n";
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

  // View kernel for debugging
  CeedDebug256(ceed, 2, "Generated QFunction Kernels:\n");
  CeedDebug(ceed, code.str().c_str());
 
  // Compile kernel
  ierr = CeedCompileHip(ceed, code.str().c_str(), &data->module,
		        1, "BLOCK_SIZE", ceed_Hip->opt_block_size);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, kernel_name.c_str(), &data->QFunction);
  CeedChkBackend(ierr);

  // Cleanup
  ierr = CeedFree(&data->qfunction_source); CeedChkBackend(ierr);
  ierr = CeedFree(&read_write_kernel_path); CeedChkBackend(ierr);
  ierr = CeedFree(&read_write_kernel_source); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
