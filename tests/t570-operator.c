/// @file
/// Test full assembly of an identity operator (see t509)
/// \test Test full assembly of an identity operator
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
  CeedInt             num_nodes = num_elem * (p - 1) + 1;
  CeedInt             ind_u[num_elem * p];
  CeedScalar          assembled_values[num_nodes * q * num_elem];
  CeedScalar          assembled_true[num_nodes * q * num_elem];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_nodes, &u);
  CeedVectorCreate(ceed, q * num_elem, &v);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p; j++) {
      ind_u[p * i + j] = i * (p - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p, 1, 1, num_nodes, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restriction_u);

  CeedInt strides_u_i[3] = {1, q, q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, q * num_elem, strides_u_i, &elem_restriction_u_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, q, CEED_GAUSS, &basis_u);

  // QFunction
  CeedQFunctionCreateIdentity(ceed, 1, CEED_EVAL_INTERP, CEED_EVAL_NONE, &qf_identity);

  // Operators
  CeedOperatorCreate(ceed, qf_identity, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_identity);
  CeedOperatorSetField(op_identity, "input", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_identity, "output", elem_restriction_u_i, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  // Fully assemble operator
  CeedSize   num_entries;
  CeedInt   *rows;
  CeedInt   *cols;
  CeedVector assembled;

  for (CeedInt k = 0; k < num_nodes * q * num_elem; ++k) {
    assembled_values[k] = 0.0;
    assembled_true[k]   = 0.0;
  }
  CeedOperatorLinearAssembleSymbolic(op_identity, &num_entries, &rows, &cols);
  CeedVectorCreate(ceed, num_entries, &assembled);
  CeedOperatorLinearAssemble(op_identity, assembled);
  {
    const CeedScalar *assembled_array;

    CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
    for (CeedInt k = 0; k < num_entries; ++k) {
      assembled_values[rows[k] * num_nodes + cols[k]] += assembled_array[k];
    }
    CeedVectorRestoreArrayRead(assembled, &assembled_array);
  }

  // Manually assemble operator
  CeedVectorSetValue(u, 0.0);
  for (CeedInt j = 0; j < num_nodes; j++) {
    CeedScalar       *u_array;
    const CeedScalar *v_array;

    // Set input
    CeedVectorGetArray(u, CEED_MEM_HOST, &u_array);
    u_array[j] = 1.0;
    if (j) u_array[j - 1] = 0.0;
    CeedVectorRestoreArray(u, &u_array);

    // Compute entries for column j
    CeedOperatorApply(op_identity, u, v, CEED_REQUEST_IMMEDIATE);

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < q * num_elem; i++) assembled_true[i * num_nodes + j] = v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Check output
  for (CeedInt i = 0; i < q * num_elem; i++) {
    for (CeedInt j = 0; j < num_nodes; j++) {
      if (fabs(assembled_values[i * num_nodes + j] - assembled_true[i * num_nodes + j]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT ", %" CeedInt_FMT "] Error in assembly: %f != %f\n", i, j, assembled_values[i * num_nodes + j],
               assembled_true[i * num_nodes + j]);
        // LCOV_EXCL_STOP
      }
    }
  }

  // Cleanup
  free(rows);
  free(cols);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&assembled);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_u_i);
  CeedBasisDestroy(&basis_u);
  CeedQFunctionDestroy(&qf_identity);
  CeedOperatorDestroy(&op_identity);
  CeedDestroy(&ceed);
  return 0;
}
