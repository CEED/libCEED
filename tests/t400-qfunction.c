// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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

/// @file
/// Test creation, evaluation, and destruction for qfunction
/// \test Test creation, evaluation, and destruction for qfunction
#include <ceed.h>
#include <math.h>
#include "t400-qfunction.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector in[16], out[16];
  CeedVector Qdata, W, U, V;
  CeedQFunction qf_setup, qf_mass;
  CeedInt Q = 8;
  const CeedScalar *vv;
  CeedScalar w[Q], u[Q], v[Q];


  CeedInit(argv[1], &ceed);

  CeedQFunctionCreateInterior(ceed, 1, setup, __FILE__ ":setup", &qf_setup);
  CeedQFunctionAddInput(qf_setup, "w", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_setup, "qdata", 1, CEED_EVAL_INTERP);

  CeedQFunctionCreateInterior(ceed, 1, mass, __FILE__ ":mass", &qf_mass);
  CeedQFunctionAddInput(qf_mass, "qdata", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  for (CeedInt i=0; i<Q; i++) {
    CeedScalar x = 2.*i/(Q-1) - 1;
    w[i] = 1 - x*x;
    u[i] = 2 + 3*x + 5*x*x;
    v[i] = w[i] * u[i];
  }

  CeedVectorCreate(ceed, Q, &W);
  CeedVectorSetArray(W, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)&w);
  CeedVectorCreate(ceed, Q, &U);
  CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)&u);
  CeedVectorCreate(ceed, Q, &V);
  CeedVectorSetValue(V, 0);
  CeedVectorCreate(ceed, Q, &Qdata);
  CeedVectorSetValue(Qdata, 0);

  {
    in[0] = W;
    out[0] = Qdata;
    CeedQFunctionApply(qf_setup, Q, in, out);
  }
  {
    in[0] = W;
    in[1] = U;
    out[0] = V;
    CeedQFunctionApply(qf_mass, Q, in, out);
  }

  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &vv);
  for (CeedInt i=0; i<Q; i++) {
    if (fabs(v[i] - vv[i]) > 1.e-14)
      printf("[%d] v %f != vv %f\n",i, v[i], vv[i]);
  }
  CeedVectorRestoreArrayRead(V, &vv);

  CeedVectorDestroy(&W);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&Qdata);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedDestroy(&ceed);
  return 0;
}
