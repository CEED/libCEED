/// @file
/// Verify nodal restriction scratch when tensor quadrature is smaller than nodes.
//TESTARGS(name="3D underintegrated") {ceed_resource} 3 5 4 2
//TESTARGS(name="2D underintegrated") {ceed_resource} 2 5 4 2
//TESTARGS(name="1D underintegrated") {ceed_resource} 1 5 4 2
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int CompareVectors(CeedVector actual, CeedVector expected) {
  const CeedScalar *actual_arr, *expected_arr;
  CeedSize          length;

  CeedVectorGetLength(actual, &length);
  CeedVectorGetArrayRead(actual, CEED_MEM_HOST, &actual_arr);
  CeedVectorGetArrayRead(expected, CEED_MEM_HOST, &expected_arr);
  for (CeedSize i = 0; i < length; i++) {
    const CeedScalar error = fabs(actual_arr[i] - expected_arr[i]) / fmax(1, fabs(expected_arr[i]));

    if (!(error <= 500 * CEED_EPSILON)) {
      printf("Restriction scratch mismatch at %td: %g != %g\n", i, actual_arr[i], expected_arr[i]);
      break;
    }
  }
  CeedVectorRestoreArrayRead(actual, &actual_arr);
  CeedVectorRestoreArrayRead(expected, &expected_arr);
  return CEED_ERROR_SUCCESS;
}

static int CreateOperator(Ceed ceed, CeedInt dim, CeedInt p, CeedInt q, CeedInt num_comp, CeedInt num_elem, CeedOperator *op, CeedVector *u,
                          CeedVector *v) {
  CeedInt             nodes = 1;
  CeedBasis           basis;
  CeedElemRestriction restriction;
  CeedQFunction       qf;

  for (CeedInt d = 0; d < dim; d++) nodes *= p;
  const CeedSize length     = num_elem * nodes * num_comp;
  const CeedInt  strides[3] = {1, nodes * num_elem, nodes};

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, p, q, CEED_GAUSS, &basis);
  CeedElemRestrictionCreateStrided(ceed, num_elem, nodes, num_comp, length, strides, &restriction);
  CeedQFunctionCreateIdentity(ceed, dim * num_comp, CEED_EVAL_GRAD, CEED_EVAL_GRAD, &qf);
  CeedOperatorCreate(ceed, qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, op);
  CeedOperatorSetField(*op, "input", restriction, basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*op, "output", restriction, basis, CEED_VECTOR_ACTIVE);
  CeedVectorCreate(ceed, length, u);
  CeedVectorCreate(ceed, length, v);
  CeedScalar *u_arr;

  CeedVectorGetArrayWrite(*u, CEED_MEM_HOST, &u_arr);
  for (CeedSize i = 0; i < length; i++) u_arr[i] = sin(0.031 * i) + 0.02 * cos(0.017 * i);
  CeedVectorRestoreArray(*u, &u_arr);
  CeedBasisDestroy(&basis);
  CeedElemRestrictionDestroy(&restriction);
  CeedQFunctionDestroy(&qf);
  return CEED_ERROR_SUCCESS;
}

int main(int argc, char **argv) {
  const CeedInt dim = atoi(argv[2]), p = atoi(argv[3]), q = atoi(argv[4]), num_comp = atoi(argv[5]), num_elem = 3;
  Ceed          ceed, ceed_ref;
  CeedOperator  op, op_ref;
  CeedVector    u, v, u_ref, v_ref;

  CeedInit(argv[1], &ceed);
  CeedInit("/cpu/self/ref/serial", &ceed_ref);
  CreateOperator(ceed, dim, p, q, num_comp, num_elem, &op, &u, &v);
  CreateOperator(ceed_ref, dim, p, q, num_comp, num_elem, &op_ref, &u_ref, &v_ref);
  CeedOperatorApply(op, u, v, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_ref, u_ref, v_ref, CEED_REQUEST_IMMEDIATE);
  CompareVectors(v, v_ref);
  CeedOperatorLinearAssembleDiagonal(op, v, CEED_REQUEST_IMMEDIATE);
  CeedOperatorLinearAssembleDiagonal(op_ref, v_ref, CEED_REQUEST_IMMEDIATE);
  CompareVectors(v, v_ref);
  CeedOperatorDestroy(&op);
  CeedOperatorDestroy(&op_ref);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&u_ref);
  CeedVectorDestroy(&v_ref);
  CeedDestroy(&ceed);
  CeedDestroy(&ceed_ref);
  return CEED_ERROR_SUCCESS;
}
