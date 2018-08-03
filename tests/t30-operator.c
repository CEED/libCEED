// Test operator action for mass matrix
#include <ceed.h>
#include <stdlib.h>

static int setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
                 CeedScalar *const *out) {
  const CeedScalar *weight = in[0], *dxdX = in[1];
  CeedScalar *rho = out[0];
  for (CeedInt i=0; i<Q; i++) {
    rho[i] = weight[i] * dxdX[i];
  }
  return 0;
}

static int mass(void *ctx, CeedInt Q, const CeedScalar *const *in,
                CeedScalar *const *out) {
  const CeedScalar *rho = in[0], *u = in[1];
  CeedScalar *v = out[0];
  for (CeedInt i=0; i<Q; i++) {
    v[i] = rho[i] * u[i];
  }
  return 0;
}

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu;
  CeedBasis bx, bu;
  CeedQFunction qf_setup, qf_mass;  CeedOperator op_setup, op_mass;
  CeedVector qdata, X, U, V;
  CeedScalar *hu;
  const CeedScalar *hv;
  CeedInt nelem = 5, P = 5, Q = 8;
  CeedInt Nx = nelem+1, Nu = nelem*(P-1)+1;
  CeedInt indx[nelem*2], indu[nelem*P];
  CeedScalar x[Nx];

  CeedInit(argv[1], &ceed);
  for (CeedInt i=0; i<Nx; i++) x[i] = (CeedScalar) i / (Nx - 1);
  for (CeedInt i=0; i<nelem; i++) {
    indx[2*i+0] = i;
    indx[2*i+1] = i+1;
  }
  CeedElemRestrictionCreate(ceed, nelem, 2, Nx, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER,
                            indx, &Erestrictx);

  for (CeedInt i=0; i<nelem; i++) {
    for (CeedInt j=0; j<P; j++) {
      indu[P*i+j] = i*(P-1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, nelem, P, Nu, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER,
                            indu, &Erestrictu);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS, &bu);

  CeedQFunctionCreateInterior(ceed, 1, setup, __FILE__ ":setup", &qf_setup);
  CeedQFunctionAddInput(qf_setup, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "x", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, __FILE__ ":mass", &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);

  CeedVectorCreate(ceed, Nx, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorCreate(ceed, nelem*Q, &qdata);

  CeedOperatorSetField(op_setup, "_weight", CEED_RESTRICTION_IDENTITY, bx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "x", Erestrictx, bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", CEED_RESTRICTION_IDENTITY,
                       CEED_BASIS_COLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass, "rho", CEED_RESTRICTION_IDENTITY,
                       CEED_BASIS_COLOCATED, qdata);
  CeedOperatorSetField(op_mass, "u", Erestrictu, bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", Erestrictu, bu, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_setup, X, qdata, CEED_REQUEST_IMMEDIATE);

  CeedVectorCreate(ceed, Nu, &U);
  CeedVectorGetArray(U, CEED_MEM_HOST, &hu);
  for (CeedInt i=0; i<Nu; i++)
    hu[i] = 0.0;
  CeedVectorRestoreArray(U, &hu);
  CeedVectorCreate(ceed, Nu, &V);
  CeedOperatorApply(op_mass, U, V, CEED_REQUEST_IMMEDIATE);

  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  for (CeedInt i=0; i<Nu; i++)
    if (hv[i] != 0.0) printf("[%d] v %g != 0.0\n",i, hv[i]);
  CeedVectorRestoreArrayRead(V, &hv);

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
  CeedVectorDestroy(&qdata);
  CeedDestroy(&ceed);
  return 0;
}
