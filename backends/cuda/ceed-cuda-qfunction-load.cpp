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

#include <ceed-backend.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include "../cuda/ceed-cuda.h"


static const char *qReadWrite = QUOTE(
template <int SIZE>
inline __device__ void readQuads(const CeedInt quad, const CeedInt nquads, const CeedScalar* d_u, CeedScalar* r_u) {
  for(CeedInt comp = 0; comp < SIZE; ++comp) {
    r_u[comp] = d_u[quad + nquads * comp];
  }
}

template <int SIZE>
inline __device__ void writeQuads(const CeedInt quad, const CeedInt nquads, const CeedScalar* r_v, CeedScalar* d_v) {
  for(CeedInt comp = 0; comp < SIZE; ++comp) {
    d_v[quad + nquads * comp] = r_v[comp];
  }
}
);

extern "C" int CeedCudaBuildQFunction(CeedQFunction qf) {
  CeedInt ierr;
  using std::ostringstream;
  using std::string;
  CeedQFunction_Cuda *data;
  ierr = CeedQFunctionGetData(qf, (void **)&data); CeedChk(ierr);
  if (!data->qFunctionSource) //qFunction is build
  {
    return 0;
  }
  //qFunction kernel generation
  CeedInt numinputfields, numoutputfields, size;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChk(ierr);

  string qFunction(data->qFunctionSource);
  string qReadWriteS(qReadWrite);

  ostringstream code;

  code << "\n#define CEED_QFUNCTION(name) inline __device__ int name\n";
  code << "\n#define CeedPragmaSIMD\n";
  code << "typedef struct { const CeedScalar* inputs[16]; CeedScalar* outputs[16]; } Fields_Cuda;\n";
  code << qReadWriteS;
  code << qFunction;
  code << "extern \"C\" __global__ void qfunction(void *ctx, CeedInt Q, Fields_Cuda fields) {\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "// Input field "<<i<<"\n";
    ierr = CeedQFunctionFieldGetSize(qfinputfields[i], &size); CeedChk(ierr);
    code << "  const CeedInt size_in_"<<i<<" = "<<size<<";\n";
    code << "  CeedScalar r_q"<<i<<"[size_in_"<<i<<"];\n";
  }
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "// Output field "<<i<<"\n";
    ierr = CeedQFunctionFieldGetSize(qfoutputfields[i], &size); CeedChk(ierr);
    code << "  const CeedInt size_out_"<<i<<" = "<<size<<";\n";
    code << "  CeedScalar r_qq"<<i<<"[size_out_"<<i<<"];\n";
  }
  code << "  const CeedScalar* in["<<numinputfields<<"];\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "    in["<<i<<"] = r_q"<<i<<";\n";
  }
  code << "  CeedScalar* out["<<numoutputfields<<"];\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "    out["<<i<<"] = r_qq"<<i<<";\n";
  }
  code << "  for (CeedInt q = blockIdx.x * blockDim.x + threadIdx.x; q < Q; q += blockDim.x * gridDim.x) {\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "// Input field "<<i<<"\n";
    code << "  readQuads<size_in_"<<i<<">(q, Q, fields.inputs["<<i<<"], r_q"<<i<<");\n";
  }
  code << "//QFunction\n";
  string qFunctionName(data->qFunctionName);
  code << "    "<<qFunctionName<<"(ctx, 1, in, out);\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "// Output field "<<i<<"\n";
    code << "  writeQuads<size_out_"<<i<<">(q, Q, r_qq"<<i<<", fields.outputs["<<i<<"]);\n";
  }
  code << "  }\n";
  code << "}\n";

  // std::cout << code.str();

  //********************
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  ierr = CeedCompileCuda(ceed, code.str().c_str(), &data->module, 0); CeedChk(ierr);
  ierr = CeedGetKernelCuda(ceed, data->module, "qfunction", &data->qFunction);
  CeedChk(ierr);
  ierr = CeedFree(&data->qFunctionSource); CeedChk(ierr);

  return 0;
}
