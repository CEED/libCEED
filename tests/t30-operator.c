// Test operator action for mass matrix
#include <ceed.h>

static int setup(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u,
                 CeedScalar *const *v) {
  CeedScalar *w = qdata;
  for (CeedInt i=0; i<Q; i++) {
    w[i] = u[1][i]*u[4][i];
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
  CeedElemRestriction Erestrictx, Erestrictu;
  CeedBasis bx, bu;
  CeedQFunction qf_setup, qf_mass;
  CeedOperator op_setup, op_mass;
  CeedVector qdata, X, U, V;
  CeedInt nelem = 5, P = 5, Q = 8;
  CeedInt Nx = nelem+1, Nu = nelem*(P-1)+1;
  CeedInt indx[nelem*2], indu[nelem*P];
  CeedScalar x[Nx];

  CeedInit("/cpu/self", &ceed);
  for (CeedInt i=0; i<Nx; i++) x[i] = i / (Nx - 1);
  for (CeedInt i=0; i<nelem; i++) {
    indx[2*i+0] = i;
    indx[2*i+1] = i+1;
  }
  CeedElemRestrictionCreate(ceed, nelem, 2, Nx, CEED_MEM_HOST, CEED_USE_POINTER,
                            indx, &Erestrictx);

  for (CeedInt i=0; i<nelem; i++) {
    for (CeedInt j=0; j<P; j++) {
      indu[P*i+j] = i*(P-1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, nelem, P, Nu, CEED_MEM_HOST, CEED_USE_POINTER,
                            indu, &Erestrictu);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS, &bu);

  CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                              (CeedEvalMode)(CEED_EVAL_GRAD|CEED_EVAL_WEIGHT),
                              CEED_EVAL_NONE, setup, __FILE__ ":setup", &qf_setup);
  CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                              CEED_EVAL_INTERP, CEED_EVAL_INTERP,
                              mass, __FILE__ ":mass", &qf_mass);

  CeedOperatorCreate(ceed, Erestrictx, bx, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorCreate(ceed, Erestrictu, bu, qf_mass, NULL, NULL, &op_mass);

  CeedVectorCreate(ceed, Nx, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedOperatorGetQData(op_setup, &qdata);
  CeedOperatorApply(op_setup, qdata, X, NULL, CEED_REQUEST_IMMEDIATE);

  CeedVectorCreate(ceed, Nu, &U);
  CeedVectorCreate(ceed, Nu, &V);
  CeedOperatorApply(op_mass, qdata, U, V, CEED_REQUEST_IMMEDIATE);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
