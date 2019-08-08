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
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);

  string qFunction(data->qFunctionSource);

  ostringstream code;

  code << "\n#define CEED_QFUNCTION(name) inline __device__ int name\n";
  code << qFunction;
  code << "typedef struct { const CeedScalar* inputs[16]; CeedScalar* outputs[16]; } Fields_Cuda;\n";
  code << "extern \"C\" __global__ void qfunction(void *ctx, CeedInt Q, Fields_Cuda fields) {\n";
  code << "  const CeedScalar* in["<<numinputfields<<"];\n";
  for (CeedInt i = 0; i < numinputfields; i++) {
    code << "  in["<<i<<"] = fields.inputs["<<i<<"];\n";
  }
  code << "  CeedScalar* out["<<numoutputfields<<"];\n";
  for (CeedInt i = 0; i < numoutputfields; i++) {
    code << "  out["<<i<<"] = fields.outputs["<<i<<"];\n";
  }
  string qFunctionName(data->qFunctionName);
  code << "  "<<qFunctionName<<"(ctx, Q, ";
  code << "in, out";
  code << ");\n";
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
