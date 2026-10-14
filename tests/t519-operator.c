/// @file
/// Verify nodal restriction scratch when tensor quadrature is smaller than nodes.
//TESTARGS(name="3D underintegrated") {ceed_resource} 3 5 4 2
//TESTARGS(name="2D underintegrated") {ceed_resource} 2 5 4 2
//TESTARGS(name="1D underintegrated") {ceed_resource} 1 5 4 2
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int Compare(CeedVector actual, CeedVector expected, CeedSize length) {
  const CeedScalar *a, *b;
  bool              correct = true;
  CeedCall(CeedVectorGetArrayRead(actual, CEED_MEM_HOST, &a));
  CeedCall(CeedVectorGetArrayRead(expected, CEED_MEM_HOST, &b));
  for (CeedSize i = 0; i < length; i++) {
    if (!isfinite(a[i]) || fabs(a[i] - b[i]) > 500 * CEED_EPSILON * fmax(1, fabs(b[i]))) {
      printf("Restriction scratch mismatch at %td: %g != %g\n", i, a[i], b[i]);
      correct = false;
      break;
    }
  }
  CeedCall(CeedVectorRestoreArrayRead(actual, &a));
  CeedCall(CeedVectorRestoreArrayRead(expected, &b));
  return correct ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc != 6) return 1;
  const CeedInt dim = atoi(argv[2]), p = atoi(argv[3]), q = atoi(argv[4]), components = atoi(argv[5]), elements = 3;
  CeedInt       nodes = 1;
  for (CeedInt d = 0; d < dim; d++) nodes *= p;
  const CeedSize length     = elements * nodes * components;
  const CeedInt  strides[3] = {1, nodes * elements, nodes};
  Ceed           ceed[2];
  CeedOperator   op[2];
  CeedVector     u[2], v[2];
  CeedCall(CeedInit(argv[1], &ceed[0]));
  CeedCall(CeedInit("/cpu/self/ref/serial", &ceed[1]));
  for (int side = 0; side < 2; side++) {
    CeedBasis           basis;
    CeedElemRestriction restriction;
    CeedQFunction       qf;
    CeedCall(CeedBasisCreateTensorH1Lagrange(ceed[side], dim, components, p, q, CEED_GAUSS, &basis));
    CeedCall(CeedElemRestrictionCreateStrided(ceed[side], elements, nodes, components, length, strides, &restriction));
    CeedCall(CeedQFunctionCreateIdentity(ceed[side], dim * components, CEED_EVAL_GRAD, CEED_EVAL_GRAD, &qf));
    CeedCall(CeedOperatorCreate(ceed[side], qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op[side]));
    CeedCall(CeedOperatorSetField(op[side], "input", restriction, basis, CEED_VECTOR_ACTIVE));
    CeedCall(CeedOperatorSetField(op[side], "output", restriction, basis, CEED_VECTOR_ACTIVE));
    CeedCall(CeedVectorCreate(ceed[side], length, &u[side]));
    CeedCall(CeedVectorCreate(ceed[side], length, &v[side]));
    CeedScalar *input;
    CeedCall(CeedVectorGetArrayWrite(u[side], CEED_MEM_HOST, &input));
    for (CeedSize i = 0; i < length; i++) input[i] = sin(0.031 * i) + 0.02 * cos(0.017 * i);
    CeedCall(CeedVectorRestoreArray(u[side], &input));
    CeedCall(CeedOperatorApply(op[side], u[side], v[side], CEED_REQUEST_IMMEDIATE));
    CeedCall(CeedBasisDestroy(&basis));
    CeedCall(CeedElemRestrictionDestroy(&restriction));
    CeedCall(CeedQFunctionDestroy(&qf));
  }
  CeedCall(Compare(v[0], v[1], length));
  for (int side = 0; side < 2; side++) CeedCall(CeedOperatorLinearAssembleDiagonal(op[side], v[side], CEED_REQUEST_IMMEDIATE));
  CeedCall(Compare(v[0], v[1], length));
  for (int side = 0; side < 2; side++) {
    CeedCall(CeedOperatorDestroy(&op[side]));
    CeedCall(CeedVectorDestroy(&u[side]));
    CeedCall(CeedVectorDestroy(&v[side]));
    CeedCall(CeedDestroy(&ceed[side]));
  }
  return 0;
}
