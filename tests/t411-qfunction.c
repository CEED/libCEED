/// @file
/// Test creation, evaluation, and destruction of identity qfunction
/// \test Test creation, evaluation, and destruction of identity qfunction
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector in[16], out[16];
  CeedVector U, V;
  CeedQFunction qf;
  CeedInt Q = 8;
  const CeedScalar *v;
  CeedScalar u[Q];

  CeedInit(argv[1], &ceed);

  CeedQFunctionCreateIdentity(ceed, 1, CEED_EVAL_INTERP, CEED_EVAL_INTERP, &qf);

  for (CeedInt i=0; i<Q; i++)
    u[i] = i*i;

  CeedVectorCreate(ceed, Q, &U);
  CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);
  CeedVectorCreate(ceed, Q, &V);
  CeedVectorSetValue(V, 0);

  {
    in[0] = U;
    out[0] = V;
    CeedQFunctionApply(qf, Q, in, out);
  }

  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &v);
  for (CeedInt i=0; i<Q; i++)
    if (fabs(v[i] - u[i])>1e-14)
      // LCOV_EXCL_START
      printf("[%d] v %f != u %f\n",i, v[i], u[i]);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(V, &v);

  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedQFunctionDestroy(&qf);
  CeedDestroy(&ceed);
  return 0;
}
