// Test qfunction evaluation
#include <ceed.h>

static int setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
                 CeedScalar *const *out) {
  const CeedScalar *w = in[0];
  CeedScalar *qdata = out[0];
  for (CeedInt i=0; i<Q; i++) {
    qdata[i] = w[i];
  }
  return 0;
}

static int mass(void *ctx, CeedInt Q, const CeedScalar *const *in,
                CeedScalar *const *out) {
  const CeedScalar *qdata = in[0], *u = in[1];
  CeedScalar *v = out[0];
  for (CeedInt i=0; i<Q; i++) {
    v[i] = qdata[i] * u[i];
  }
  return 0;
}

int main(int argc, char **argv) {
  Ceed ceed;
  CeedQFunction qf_setup, qf_mass;
  CeedInt Q = 8;
  CeedScalar qdata[Q], w[Q], u[Q], v[Q], vv[Q];


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
  {
    const CeedScalar *const in[1] = {w};
    CeedScalar *const out[1] = {qdata};
    CeedQFunctionApply(qf_setup, Q, in, out);
  }
  {
    const CeedScalar *const in[2] = {qdata, u};
    CeedScalar *const out[1] = {vv};
    CeedQFunctionApply(qf_mass, Q, in, out);
  }
  for (CeedInt i=0; i<Q; i++) {
    if (v[i] != vv[i]) printf("[%d] v %f != vv %f\n",i, v[i], vv[i]);
  }
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedDestroy(&ceed);
  return 0;
}
