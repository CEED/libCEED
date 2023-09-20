/// @file
/// Test full assembly of a non-square mass matrix operator (see t553)
/// \test Test full assembly of a non-square mass matrix operator
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_q_data, elem_restriction_u_coarse, elem_restriction_u_fine;
  CeedBasis           basis_x, basis_u_coarse, basis_u_fine;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, x, u, v;
  CeedInt             num_elem = 15, p_coarse = 3, p_fine = 5, q = 8;
  CeedInt             num_dofs_x = num_elem + 1, num_dofs_u_coarse = num_elem * (p_coarse - 1) + 1, num_dofs_u_fine = num_elem * (p_fine - 1) + 1;
  CeedInt             ind_u_coarse[num_elem * p_coarse], ind_u_fine[num_elem * p_fine], ind_x[num_elem * 2];
  CeedScalar          assembled_values[num_dofs_u_coarse * num_dofs_u_fine];
  CeedScalar          assembled_true[num_dofs_u_coarse * num_dofs_u_fine];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_dofs_x, &x);
  {
    CeedScalar x_array[num_dofs_x];

    for (CeedInt i = 0; i < num_dofs_x; i++) x_array[i] = (CeedScalar)i / (num_dofs_x - 1);
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_dofs_u_coarse, &u);
  CeedVectorCreate(ceed, num_dofs_u_fine, &v);
  CeedVectorCreate(ceed, num_elem * q, &q_data);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_dofs_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p_coarse; j++) {
      ind_u_coarse[p_coarse * i + j] = i * (p_coarse - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p_coarse, 1, 1, num_dofs_u_coarse, CEED_MEM_HOST, CEED_USE_POINTER, ind_u_coarse,
                            &elem_restriction_u_coarse);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p_fine; j++) {
      ind_u_fine[p_fine * i + j] = i * (p_fine - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p_fine, 1, 1, num_dofs_u_fine, CEED_MEM_HOST, CEED_USE_POINTER, ind_u_fine, &elem_restriction_u_fine);

  CeedInt strides_q_data[3] = {1, q, q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, q * num_elem, strides_q_data, &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p_coarse, q, CEED_GAUSS, &basis_u_coarse);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p_fine, q, CEED_GAUSS, &basis_u_fine);

  // QFunctions
  CeedQFunctionCreateInteriorByName(ceed, "Mass1DBuild", &qf_setup);
  CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_mass);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);

  CeedOperatorSetField(op_setup, "weights", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "qdata", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "qdata", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u_coarse, basis_u_coarse, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u_fine, basis_u_fine, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

  // Fully assemble operator
  CeedSize   num_entries;
  CeedInt   *rows;
  CeedInt   *cols;
  CeedVector assembled;

  for (CeedInt k = 0; k < num_dofs_u_coarse * num_dofs_u_fine; ++k) {
    assembled_values[k] = 0.0;
    assembled_true[k]   = 0.0;
  }
  CeedOperatorLinearAssembleSymbolic(op_mass, &num_entries, &rows, &cols);
  CeedVectorCreate(ceed, num_entries, &assembled);
  CeedOperatorLinearAssemble(op_mass, assembled);
  {
    const CeedScalar *assembled_array;

    CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
    for (CeedInt k = 0; k < num_entries; ++k) {
      assembled_values[rows[k] * num_dofs_u_coarse + cols[k]] += assembled_array[k];
    }
    CeedVectorRestoreArrayRead(assembled, &assembled_array);
  }

  // Manually assemble operator
  CeedVectorSetValue(u, 0.0);
  for (CeedInt j = 0; j < num_dofs_u_coarse; j++) {
    CeedScalar       *u_array;
    const CeedScalar *v_array;

    // Set input
    CeedVectorGetArray(u, CEED_MEM_HOST, &u_array);
    u_array[j] = 1.0;
    if (j) u_array[j - 1] = 0.0;
    CeedVectorRestoreArray(u, &u_array);

    // Compute entries for column j
    CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_dofs_u_fine; i++) assembled_true[i * num_dofs_u_coarse + j] = v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Check output
  for (CeedInt i = 0; i < num_dofs_u_fine; i++) {
    for (CeedInt j = 0; j < num_dofs_u_coarse; j++) {
      if (fabs(assembled_values[i * num_dofs_u_coarse + j] - assembled_true[i * num_dofs_u_coarse + j]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT ", %" CeedInt_FMT "] Error in assembly: %f != %f\n", i, j, assembled_values[i * num_dofs_u_coarse + j],
               assembled_true[i * num_dofs_u_coarse + j]);
        // LCOV_EXCL_STOP
      }
    }
  }

  // Cleanup
  free(rows);
  free(cols);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&assembled);
  CeedElemRestrictionDestroy(&elem_restriction_u_coarse);
  CeedElemRestrictionDestroy(&elem_restriction_u_fine);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_u_coarse);
  CeedBasisDestroy(&basis_u_fine);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedDestroy(&ceed);
  return 0;
}
