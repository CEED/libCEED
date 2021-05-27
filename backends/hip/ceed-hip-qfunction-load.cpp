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
#include <iostream>
#include <sstream>
#include <string.h>
#include "ceed-hip.h"
#include "ceed-hip-compile.h"

static const char *qReadWrite = QUOTE(
template <int SIZE>
//------------------------------------------------------------------------------
// Read from quadrature points
//------------------------------------------------------------------------------
inline __device__ void readQuads(const CeedInt quad, const CeedInt nquads, const CeedScalar* d_u, CeedScalar* r_u) {
  for(CeedInt comp = 0; comp < SIZE; ++comp) {
    r_u[comp] = d_u[quad + nquads * comp];
  }
}

//------------------------------------------------------------------------------
// Write at quadrature points
//------------------------------------------------------------------------------
template <int SIZE>
inline __device__ void writeQuads(const CeedInt quad, const CeedInt nquads, const CeedScalar* r_v, CeedScalar* d_v) {
  for(CeedInt comp = 0; comp < SIZE; ++comp) {
    d_v[quad + nquads * comp] = r_v[comp];
  }
}
);

//------------------------------------------------------------------------------
// Build QFunction kernel
//------------------------------------------------------------------------------
extern "C" int CeedHipBuildQFunction(CeedQFunction qf) {
  CeedInt ierr;
  using std::ostringstream;
  using std::string;
  CeedQFunction_Hip *data;
  ierr = CeedQFunctionGetData(qf, (void **)&data); CeedChkBackend(ierr);
  // QFunction is built
  if (!data->qFunctionSource)
    return CEED_ERROR_SUCCESS;
  
  // QFunction kernel generation
  CeedInt numinputfields, numoutputfields, size;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChkBackend(ierr);

  // Build strings for final kernel
  string qFunction(data->qFunctionSource);
  string qReadWriteS(qReadWrite);
  ostringstream code;
  string qFunctionName(data->qFunctionName);
  string kernelName;
  kernelName = "CeedKernel_Hip_ref_" + qFunctionName;

  // Defintions
  code << "\n#define CEED_QFUNCTION(name) inline __device__ int name\n";
  code << "#define CEED_QFUNCTION_HELPER inline __device__ __forceinline__\n";
  code << "#define CeedPragmaSIMD\n";
  code << "#define CEED_ERROR_SUCCESS 0\n";
  code << "#define CEED_Q_VLA 1\n\n";
  code << "typedef struct { const CeedScalar* inputs[16]; CeedScalar* outputs[16]; } Fields_Hip;\n";
  code << qReadWriteS;
  code << qFunction;
  code << "extern \"C\" __global__ void " << kernelName << "(void *ctx, CeedInt Q, Fields_Hip fields) {\n";
  
  // Inputs
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "// Input field "<<i<<"\n";
    ierr = CeedQFunctionFieldGetSize(qfinputfields[i], &size); CeedChkBackend(ierr);
    code << "  const CeedInt size_in_"<<i<<" = "<<size<<";\n";
    code << "  CeedScalar r_q"<<i<<"[size_in_"<<i<<"];\n";
  }

  // Outputs
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "// Output field "<<i<<"\n";
    ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &size); CeedChkBackend(ierr);
    code << "  const CeedInt size_out_"<<i<<" = "<<size<<";\n";
    code << "  CeedScalar r_qq"<<i<<"[size_out_"<<i<<"];\n";
  }

  // Setup input/output arrays
  code << "  const CeedScalar* in["<<numinputfields<<"];\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "    in["<<i<<"] = r_q"<<i<<";\n";
  }
  code << "  CeedScalar* out["<<numoutputfields<<"];\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "    out["<<i<<"] = r_qq"<<i<<";\n";
  }

  // Loop over quadrature points
  code << "  for (CeedInt q = blockIdx.x * blockDim.x + threadIdx.x; q < Q; q += blockDim.x * gridDim.x) {\n";

  // Load inputs
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "// Input field "<<i<<"\n";
    code << "  readQuads<size_in_"<<i<<">(q, Q, fields.inputs["<<i<<"], r_q"<<i<<");\n";
  }
  // QFunction
  code << "// QFunction\n";
  code << "    "<<qFunctionName<<"(ctx, 1, in, out);\n";

  // Write outputs
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "// Output field "<<i<<"\n";
    code << "  writeQuads<size_out_"<<i<<">(q, Q, r_qq"<<i<<", fields.outputs["<<i<<"]);\n";
  }
  code << "  }\n";
  code << "}\n";

  // View kernel for debugging
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedDebug(code.str().c_str());
 
  // Compile kernel
  ierr = CeedCompileHip(ceed, code.str().c_str(), &data->module, 0);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, kernelName.c_str(), &data->qFunction);
  CeedChkBackend(ierr);

  // Cleanup
  ierr = CeedFree(&data->qFunctionSource); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
