/// @file
/// Test creation, evaluation, and destruction for qfunction
/// \test Test creation, evaluation, and destruction for qfunction
#include <ceed.h>

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

  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "w", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_setup, "qdata", 1, CEED_EVAL_INTERP);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
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
  CeedVectorSetArray(W, CEED_MEM_HOST, CEED_USE_POINTER, w);
  CeedVectorCreate(ceed, Q, &U);
  CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);
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
  for (CeedInt i=0; i<Q; i++)
    if (v[i] != vv[i])
      // LCOV_EXCL_START
      printf("[%d] v %f != vv %f\n",i, v[i], vv[i]);
  // LCOV_EXCL_STOP
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
