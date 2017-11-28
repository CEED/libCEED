// Test qfunction evaluation
#include <feme.h>

static int setup(void *ctx, void *qdata, FemeInt Q, const FemeScalar *const *u, FemeScalar *const *v) {
  FemeScalar *w = qdata;
  for (FemeInt i=0; i<Q; i++) {
    w[i] = 1.0;
  }
  return 0;
}

static int mass(void *ctx, void *qdata, FemeInt Q, const FemeScalar *const *u, FemeScalar *const *v) {
  const FemeScalar *w = qdata;
  for (FemeInt i=0; i<Q; i++) {
    v[0][i] = w[i] * u[0][i];
  }
  return 0;
}

int main(int argc, char **argv) {
  Feme feme;
  FemeQFunction qf_setup, qf_mass;

  FemeInit("/cpu/self", &feme);
  FemeQFunctionCreateInterior(feme, 1, 1, sizeof(FemeScalar), FEME_EVAL_NONE, FEME_EVAL_NONE, setup, __FILE__ ":setup", &qf_setup);
  FemeQFunctionCreateInterior(feme, 1, 1, sizeof(FemeScalar), FEME_EVAL_INTERP, FEME_EVAL_INTERP, mass, __FILE__ ":mass", &qf_mass);
  FemeQFunctionDestroy(&qf_setup);
  FemeQFunctionDestroy(&qf_mass);
  FemeDestroy(&feme);
  return 0;
}
