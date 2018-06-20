// Test qfunction evaluation
#include <ceed.h>

static int setup(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u,
                 CeedScalar *const *v) {
  CeedScalar *w = qdata;
  for (CeedInt i=0; i<Q; i++) {
    w[i] = u[0][i];
  }
  return 0;
}

static int mass(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u,
                CeedScalar *const *v) {
  const CeedScalar *w = qdata;
  for (CeedInt i=0; i<Q; i++) {
    v[0][i] = w[i] * u[0][i];
  }
  return 0;
}

int main(int argc, char **argv) {
  Ceed ceed;
  CeedQFunction qf_setup, qf_mass;
  CeedInt Q = 8;
  CeedScalar qdata[Q], w[Q], u[Q], v[Q], vv[Q];

  CeedInit(argv[1], &ceed);
  CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                              CEED_EVAL_WEIGHT, CEED_EVAL_NONE,
                              setup, NULL, __FILE__ ":setup", &qf_setup);
  CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                              CEED_EVAL_INTERP, CEED_EVAL_INTERP,
                              mass, NULL, __FILE__ ":mass", &qf_mass);
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar x = 2.*i/(Q-1) - 1;
    w[i] = 1 - x*x;
    u[i] = 2 + 3*x + 5*x*x;
    v[i] = w[i] * u[i];
  }
  {
    const CeedScalar *const up = w;
    CeedQFunctionApply(qf_setup, qdata, Q, &up, NULL);
  }
  {
    const CeedScalar *const up = u;
    CeedScalar *const vp = vv;
    CeedQFunctionApply(qf_mass, qdata, Q, &up, &vp);
  }
  for (CeedInt i=0; i<Q; i++) {
    if (v[i] != vv[i]) printf("[%d] v %f != vv %f\n",i, v[i], vv[i]);
  }
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedDestroy(&ceed);
  return 0;
}
