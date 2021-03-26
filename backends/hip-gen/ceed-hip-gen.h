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

#ifndef _ceed_hip_gen_h
#define _ceed_hip_gen_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>
#include "../hip/ceed-hip.h"

typedef struct { const CeedScalar *in[16]; CeedScalar *out[16]; } HipFields;
typedef struct { CeedInt *in[16]; CeedInt *out[16]; } HipFieldsInt;

typedef struct {
  CeedInt dim;
  CeedInt Q1d;
  CeedInt maxP1d;
  hipModule_t module;
  hipFunction_t op;
  HipFieldsInt indices;
  HipFields fields;
  HipFields B;
  HipFields G;
  CeedScalar *W;
} CeedOperator_Hip_gen;

typedef struct {
  char *qFunctionName;
  char *qFunctionSource;
  void *d_c;
} CeedQFunction_Hip_gen;

typedef struct {
  Ceed_Hip base;
} Ceed_Hip_gen;

CEED_INTERN int CeedQFunctionCreate_Hip_gen(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Hip_gen(CeedOperator op);

CEED_INTERN int CeedCompositeOperatorCreate_Hip_gen(CeedOperator op);

#endif // _ceed_hip_gen_h
