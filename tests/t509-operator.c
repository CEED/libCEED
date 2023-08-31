/// @file
/// Test creation, action, and destruction for identity operator
/// \test Test creation, action, and destruction for identity operator
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_u, elem_restriction_u_i;
  CeedBasis           basis_u;
  CeedQFunction       qf_identity;
  CeedOperator        op_identity;
  CeedVector          u, v;
  CeedInt             num_elem = 15, p = 5, q = 8;
  CeedInt             elem_size = p, num_nodes = num_elem * (p - 1) + 1;
  CeedInt             ind_u[num_elem * p];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_nodes, &u);
  CeedVectorCreate(ceed, elem_size * num_elem, &v);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p; j++) {
      ind_u[p * i + j] = i * (p - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, elem_size, 1, 1, num_nodes, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restriction_u);

  CeedInt strides_u_i[3] = {1, p, p};
  CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, 1, elem_size * num_elem, strides_u_i, &elem_restriction_u_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, q, CEED_GAUSS, &basis_u);

  // QFunction
  CeedQFunctionCreateIdentity(ceed, 1, CEED_EVAL_NONE, CEED_EVAL_NONE, &qf_identity);

  // Operators
  CeedOperatorCreate(ceed, qf_identity, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_identity);
  // -- Note: number of quadrature points for field set by elem_size of restriction when CEED_BASIS_NONE used
  CeedOperatorSetField(op_identity, "input", elem_restriction_u, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_identity, "output", elem_restriction_u_i, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedVectorSetValue(u, 3.0);
  CeedOperatorApply(op_identity, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < elem_size * num_elem; i++) {
      if (fabs(v_array[i] - 3.) > 1e-14) printf("[%" CeedInt_FMT "] Computed Value: %f != True Value: 1.0\n", i, v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_u_i);
  CeedBasisDestroy(&basis_u);
  CeedQFunctionDestroy(&qf_identity);
  CeedOperatorDestroy(&op_identity);
  CeedDestroy(&ceed);
  return 0;
}
