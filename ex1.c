#include "feme.h"
#include <stdlib.h>
#include <math.h>

static int f_mass(void *ctx, void *qdata, FemeInt Q, const FemeScalar *const *u, FemeScalar *const *v) {
  const FemeScalar *w = qdata;
  for (FemeInt i=0; i<Q; i++) v[0][i] = w[i] * u[0][i];
  return 0;
}

static int f_poisson3d(void *ctx, void *qdata, FemeInt Q, const FemeScalar *const *u, FemeScalar *const *v) {
  // Q is guaranteed to be a multiple of 8 (because of how we call FemeQFunctionCreateInterior) so we can tell the compiler
  Q = 8*(Q/8);
  // qdata can be packed arbitrarily, but we'll choose a vector-friendly ordering here
  const FemeScalar *rhs = qdata;
  const FemeScalar (*K)[Q] = (const FemeScalar(*)[Q])(rhs + Q);  // Probably symmetric but we don't have to exploit it
  for (FemeInt i=0; i<Q; i++) {
    v[0][i] = -rhs[i];
    for (FemeInt d=0; d<3; d++) {
      v[1][d*Q+i] = K[d*3+0][i] * u[1][0*Q+i] + K[d*3+1][i] * u[1][1*Q+i] + K[d*3+2][i] * u[1][2*Q+i];
    }
  }
  return 0;
}

static int f_buildcoeffs(void *ctx, void *qdata, FemeInt Q, const FemeScalar *const *u, FemeScalar *const *v) {
  FemeScalar *rhs = qdata;
  FemeScalar (*K)[Q] = (FemeScalar(*)[Q])(rhs + Q);
  for (FemeInt i=0; i<Q; i++) {
    // RHS as an analytic function of the coordinates
    rhs[i] = cos(u[0][0*Q+i]) * cos(u[0][1*Q+i]) * cos(u[0][1*Q+i]);
    // ... set K using the gradient of coordinates u[1][...]
    for (FemeInt d=0; d<3; d++) {
      for (FemeInt e=0; e<3; e++) {
        K[d*3+e][i] = (u[1][(d*3+0)*Q+i] * u[1][(e*3+0)*Q+i] +
                       u[1][(d*3+1)*Q+i] * u[1][(e*3+1)*Q+i] +
                       u[1][(d*3+2)*Q+i] * u[1][(e*3+2)*Q+i]); // quadrature weight elided
      }
    }
  }
  return 0;
}

int main(int argc, char **argv)
{
  Feme feme;
  FemeVector u, r, xcoord, qdata;
  FemeInt *Eindices;
  FemeElemRestriction Erestrict;
  FemeBasis Basis;
  FemeQFunction qf_mass, qf_poisson3d, qf_buildcoeffs;
  FemeOperator op_mass, op_poisson3d, op_buildcoeffs;

  FemeInit("/cpu/self", &feme); // implementation aborts on error by default
  FemeVectorCreate(feme, 1234, &u);
  FemeVectorCreate(feme, 1234, &r);
  FemeVectorCreate(feme, 1234*3, &xcoord);

  Eindices = malloc(123 * 125 * sizeof(Eindices[0]));
  // call function to initialize Eindices...
  FemeElemRestrictionCreate(feme, 123, 125, FEME_MEM_HOST, FEME_USE_POINTER, Eindices, &Erestrict);

  // Create a 3D Q_3 Lagrange element with 4^3 Gauss quadrature points
  FemeBasisCreateTensorH1Lagrange(feme, 3, 3, 4, &Basis);

  FemeQFunctionCreateInterior(feme, 1, 1, sizeof(FemeScalar), FEME_EVAL_INTERP, FEME_EVAL_INTERP, f_mass, "ex1.c:f_mass", &qf_mass);
  FemeQFunctionCreateInterior(feme, 8, 1, 10*sizeof(FemeScalar), FEME_EVAL_GRAD, FEME_EVAL_GRAD, f_poisson3d, "ex1.c:f_poisson3d", &qf_poisson3d);
  FemeQFunctionCreateInterior(feme, 1, 3, 10*sizeof(FemeScalar), FEME_EVAL_INTERP | FEME_EVAL_GRAD, FEME_EVAL_NONE, f_buildcoeffs, "ex1.c:f_buildcoeffs", &qf_buildcoeffs);
  // We'll expect to build libraries of qfunctions, looked up by some name.  These should be cheap to create even if not used.

  FemeOperatorCreate(feme, Erestrict, Basis, qf_mass, NULL, NULL, &op_mass);
  FemeOperatorCreate(feme, Erestrict, Basis, qf_poisson3d, NULL, NULL, &op_poisson3d);
  FemeOperatorCreate(feme, Erestrict, Basis, qf_buildcoeffs, NULL, NULL, &op_buildcoeffs);

  // ... initialize xcoord

  // Apply the operator
  FemeOperatorGetQData(op_poisson3d, &qdata); // allocates if needed
  FemeOperatorApply(op_buildcoeffs, qdata, xcoord, NULL, FEME_REQUEST_IMMEDIATE);
  FemeOperatorApply(op_poisson3d, qdata, u, r, FEME_REQUEST_IMMEDIATE);

  FemeVectorDestroy(&u);
  FemeVectorDestroy(&r);
  FemeOperatorDestroy(&op_mass);
  FemeOperatorDestroy(&op_poisson3d);
  FemeQFunctionDestroy(&qf_mass);
  FemeQFunctionDestroy(&qf_poisson3d);
  FemeBasisDestroy(&Basis);
  FemeElemRestrictionDestroy(&Erestrict);
  FemeDestroy(&feme);
  return 0;
}
