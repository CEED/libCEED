/// @file
/// Test creation, action, and destruction for identity operator
/// \test Test creation, action, and destruction for identity operator
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_u, elem_restr_u_i;
  CeedBasis           basis_u;
  CeedQFunction       qf_identity;
  CeedOperator        op_identity;
  CeedVector          U, V;
  const CeedScalar   *hv;
  CeedInt             num_elem = 15, P = 5, Q = 8;
  CeedInt             elem_size = P, num_nodes = num_elem * (P - 1) + 1;
  CeedInt             ind_u[num_elem * P];

  CeedInit(argv[1], &ceed);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < P; j++) {
      ind_u[P * i + j] = i * (P - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, elem_size, 1, 1, num_nodes, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restr_u);
  CeedInt strides_u_i[3] = {1, P, P};
  CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, 1, elem_size * num_elem, strides_u_i, &elem_restr_u_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS, &basis_u);

  // QFunction
  CeedQFunctionCreateIdentity(ceed, 1, CEED_EVAL_NONE, CEED_EVAL_NONE, &qf_identity);

  // Operators
  CeedOperatorCreate(ceed, qf_identity, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_identity);
  CeedOperatorSetField(op_identity, "input", elem_restr_u, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_identity, "output", elem_restr_u_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetNumQuadraturePoints(op_identity, elem_size);

  CeedVectorCreate(ceed, num_nodes, &U);
  CeedVectorSetValue(U, 3.0);
  CeedVectorCreate(ceed, elem_size * num_elem, &V);
  CeedOperatorApply(op_identity, U, V, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  for (CeedInt i = 0; i < elem_size * num_elem; i++) {
    if (fabs(hv[i] - 3.) > 1e-14) printf("[%" CeedInt_FMT "] Computed Value: %f != True Value: 1.0\n", i, hv[i]);
  }
  CeedVectorRestoreArrayRead(V, &hv);

  CeedQFunctionDestroy(&qf_identity);
  CeedOperatorDestroy(&op_identity);
  CeedElemRestrictionDestroy(&elem_restr_u);
  CeedElemRestrictionDestroy(&elem_restr_u_i);
  CeedBasisDestroy(&basis_u);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
