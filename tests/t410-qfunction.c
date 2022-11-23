/// @file
/// Test creation, evaluation, and destruction for QFunction by name
/// \test Test creation, evaluation, and destruction for QFunction by name
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        in[16], out[16];
  CeedVector        Q_data, J, W, U, V;
  CeedQFunction     qf_setup, qf_mass;
  CeedInt           Q = 8;
  const CeedScalar *vv;
  CeedScalar        j[Q], w[Q], u[Q], v[Q];

  CeedInit(argv[1], &ceed);

  CeedQFunctionCreateInteriorByName(ceed, "Mass1DBuild", &qf_setup);
  CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_mass);

  for (CeedInt i = 0; i < Q; i++) {
    CeedScalar x = 2. * i / (Q - 1) - 1;
    j[i]         = 1;
    w[i]         = 1 - x * x;
    u[i]         = 2 + 3 * x + 5 * x * x;
    v[i]         = w[i] * u[i];
  }

  CeedVectorCreate(ceed, Q, &J);
  CeedVectorSetArray(J, CEED_MEM_HOST, CEED_USE_POINTER, j);
  CeedVectorCreate(ceed, Q, &W);
  CeedVectorSetArray(W, CEED_MEM_HOST, CEED_USE_POINTER, w);
  CeedVectorCreate(ceed, Q, &U);
  CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);
  CeedVectorCreate(ceed, Q, &V);
  CeedVectorSetValue(V, 0);
  CeedVectorCreate(ceed, Q, &Q_data);
  CeedVectorSetValue(Q_data, 0);

  {
    in[0]  = J;
    in[1]  = W;
    out[0] = Q_data;
    CeedQFunctionApply(qf_setup, Q, in, out);
  }
  {
    in[0]  = W;
    in[1]  = U;
    out[0] = V;
    CeedQFunctionApply(qf_mass, Q, in, out);
  }

  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &vv);
  for (CeedInt i = 0; i < Q; i++) {
    if (v[i] != vv[i]) printf("[%" CeedInt_FMT "] v %f != vv %f\n", i, v[i], vv[i]);
  }
  CeedVectorRestoreArrayRead(V, &vv);

  CeedVectorDestroy(&J);
  CeedVectorDestroy(&W);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&Q_data);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedDestroy(&ceed);
  return 0;
}
