/// @file
/// Test creation, evaluation, and destruction of identity QFunction
/// \test Test creation, evaluation, and destruction of identity QFunction
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    in[16], out[16];
  CeedVector    u, v;
  CeedQFunction qf;
  CeedInt       q = 8;
  CeedScalar    u_array[q];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, q, &u);
  for (CeedInt i = 0; i < q; i++) u_array[i] = i * i;
  CeedVectorSetArray(u, CEED_MEM_HOST, CEED_USE_POINTER, u_array);
  CeedVectorCreate(ceed, q, &v);
  CeedVectorSetValue(v, 0);

  CeedQFunctionCreateIdentity(ceed, 1, CEED_EVAL_INTERP, CEED_EVAL_INTERP, &qf);
  {
    in[0]  = u;
    out[0] = v;
    CeedQFunctionApply(qf, q, in, out);
  }

  // Verify results
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < q; i++) {
      if (fabs(v_array[i] - u_array[i]) > 1e-14) printf("[%" CeedInt_FMT "] v %f != u %f\n", i, v_array[i], u_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedQFunctionDestroy(&qf);
  CeedDestroy(&ceed);
  return 0;
}
