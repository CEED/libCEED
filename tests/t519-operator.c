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

  CeedVectorGetArrayRead(actual, CEED_MEM_HOST, &a);
  CeedVectorGetArrayRead(expected, CEED_MEM_HOST, &b);
  for (CeedSize i = 0; i < length; i++) {
    if (!isfinite(a[i]) || !isfinite(b[i]) || fabs(a[i] - b[i]) > 500 * CEED_EPSILON * fmax(1, fabs(b[i]))) {
      printf("Restriction scratch mismatch at %td: %g != %g\n", i, a[i], b[i]);
      correct = false;
      break;
    }
  }
  CeedVectorRestoreArrayRead(actual, &a);
  CeedVectorRestoreArrayRead(expected, &b);
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

  CeedInit(argv[1], &ceed[0]);
  CeedInit("/cpu/self/ref/serial", &ceed[1]);
  for (CeedInt side = 0; side < 2; side++) {
    CeedBasis           basis;
    CeedElemRestriction restriction;
    CeedQFunction       qf;

    CeedBasisCreateTensorH1Lagrange(ceed[side], dim, components, p, q, CEED_GAUSS, &basis);
    CeedElemRestrictionCreateStrided(ceed[side], elements, nodes, components, length, strides, &restriction);
    CeedQFunctionCreateIdentity(ceed[side], dim * components, CEED_EVAL_GRAD, CEED_EVAL_GRAD, &qf);
    CeedOperatorCreate(ceed[side], qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op[side]);
    CeedOperatorSetField(op[side], "input", restriction, basis, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op[side], "output", restriction, basis, CEED_VECTOR_ACTIVE);
    CeedVectorCreate(ceed[side], length, &u[side]);
    CeedVectorCreate(ceed[side], length, &v[side]);
    CeedScalar *input;

    CeedVectorGetArrayWrite(u[side], CEED_MEM_HOST, &input);
    for (CeedSize i = 0; i < length; i++) input[i] = sin(0.031 * i) + 0.02 * cos(0.017 * i);
    CeedVectorRestoreArray(u[side], &input);
    CeedOperatorApply(op[side], u[side], v[side], CEED_REQUEST_IMMEDIATE);
    CeedBasisDestroy(&basis);
    CeedElemRestrictionDestroy(&restriction);
    CeedQFunctionDestroy(&qf);
  }
  if (Compare(v[0], v[1], length)) return 1;
  for (CeedInt side = 0; side < 2; side++) CeedOperatorLinearAssembleDiagonal(op[side], v[side], CEED_REQUEST_IMMEDIATE);
  if (Compare(v[0], v[1], length)) return 1;
  for (CeedInt side = 0; side < 2; side++) {
    CeedOperatorDestroy(&op[side]);
    CeedVectorDestroy(&u[side]);
    CeedVectorDestroy(&v[side]);
    CeedDestroy(&ceed[side]);
  }
  return 0;
}
