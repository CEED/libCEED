/// @file
/// Test creation, action, and destruction for identity operator
/// \test Test creation, action, and destruction for identity operator
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_u_i;
  CeedQFunction       qf_identity;
  CeedOperator        op_identity;
  CeedVector          u, v;
  CeedInt             num_elem = 15, p = 5;
  CeedInt             num_nodes = num_elem * p;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_nodes, &u);
  CeedVectorCreate(ceed, num_nodes, &v);

  // Restrictions
  CeedInt strides_u_i[3] = {1, p, p};
  CeedElemRestrictionCreateStrided(ceed, num_elem, p, 1, p * num_elem, strides_u_i, &elem_restriction_u_i);

  // QFunction
  CeedQFunctionCreateIdentity(ceed, 1, CEED_EVAL_NONE, CEED_EVAL_NONE, &qf_identity);

  // Operators
  CeedOperatorCreate(ceed, qf_identity, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_identity);
  CeedOperatorSetField(op_identity, "input", elem_restriction_u_i, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_identity, "output", elem_restriction_u_i, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedVectorSetValue(u, 3.0);
  CeedOperatorApply(op_identity, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_nodes; i++) {
      if (fabs(v_array[i] - 3.) > 100. * CEED_EPSILON) printf("[%" CeedInt_FMT "] Computed Value: %f != True Value: 3.0\n", i, v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedElemRestrictionDestroy(&elem_restriction_u_i);
  CeedQFunctionDestroy(&qf_identity);
  CeedOperatorDestroy(&op_identity);
  CeedDestroy(&ceed);
  return 0;
}
