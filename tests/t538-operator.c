/// @file
/// Test assembly of composite operator diagonal
/// \test Test assembly of composite operator diagonal
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data_mass, elem_restriction_q_data_diff;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup_mass, qf_mass, qf_setup_diff, qf_diff;
  CeedOperator        op_setup_mass, op_mass, op_setup_diff, op_diff, op_apply;
  CeedVector          q_data_mass, q_data_diff, x, assembled, u, v;
  CeedInt             num_elem = 6, p = 3, q = 4, dim = 2;
  CeedInt             n_x = 3, n_y = 2;
  CeedInt             num_dofs = (n_x * 2 + 1) * (n_y * 2 + 1), num_qpts = num_elem * q * q;
  CeedInt             ind_x[num_elem * p * p];
  CeedScalar          assembled_true[num_dofs];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs, &x);
  {
    CeedScalar x_array[dim * num_dofs];

    for (CeedInt i = 0; i < n_x * 2 + 1; i++) {
      for (CeedInt j = 0; j < n_y * 2 + 1; j++) {
        x_array[i + j * (n_x * 2 + 1) + 0 * num_dofs] = (CeedScalar)i / (2 * n_x);
        x_array[i + j * (n_x * 2 + 1) + 1 * num_dofs] = (CeedScalar)j / (2 * n_y);
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorCreate(ceed, num_dofs, &v);
  CeedVectorCreate(ceed, num_qpts, &q_data_mass);
  CeedVectorCreate(ceed, num_qpts * dim * (dim + 1) / 2, &q_data_diff);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;
    col    = i % n_x;
    row    = i / n_x;
    offset = col * (p - 1) + row * (n_x * 2 + 1) * (p - 1);
    for (CeedInt j = 0; j < p; j++) {
      for (CeedInt k = 0; k < p; k++) ind_x[p * (p * i + k) + j] = offset + k * (n_x * 2 + 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p * p, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
  CeedElemRestrictionCreate(ceed, num_elem, p * p, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_u);

  CeedInt strides_q_data_mass[3] = {1, q * q, q * q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, 1, num_qpts, strides_q_data_mass, &elem_restriction_q_data_mass);

  CeedInt strides_q_data_diff[3] = {1, q * q, q * q * dim * (dim + 1) / 2};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, dim * (dim + 1) / 2, dim * (dim + 1) / 2 * num_qpts, strides_q_data_diff,
                                   &elem_restriction_q_data_diff);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis_u);

  // QFunction - setup mass
  CeedQFunctionCreateInteriorByName(ceed, "Mass2DBuild", &qf_setup_mass);

  // Operator - setup mass
  CeedOperatorCreate(ceed, qf_setup_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_mass);
  CeedOperatorSetField(op_setup_mass, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_mass, "weights", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_mass, "qdata", elem_restriction_q_data_mass, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // QFunction - setup diffusion
  CeedQFunctionCreateInteriorByName(ceed, "Poisson2DBuild", &qf_setup_diff);

  // Operator - setup diffusion
  CeedOperatorCreate(ceed, qf_setup_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_diff);
  CeedOperatorSetField(op_setup_diff, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_diff, "weights", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_diff, "qdata", elem_restriction_q_data_diff, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operators
  CeedOperatorApply(op_setup_mass, x, q_data_mass, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_diff, x, q_data_diff, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply mass
  CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_mass);

  // Operator - apply mass
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", elem_restriction_q_data_mass, CEED_BASIS_COLLOCATED, q_data_mass);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // QFunction - apply diff
  CeedQFunctionCreateInteriorByName(ceed, "Poisson2DApply", &qf_diff);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_diff);
  CeedOperatorSetField(op_diff, "du", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff, "qdata", elem_restriction_q_data_diff, CEED_BASIS_COLLOCATED, q_data_diff);
  CeedOperatorSetField(op_diff, "dv", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Composite operator
  CeedCompositeOperatorCreate(ceed, &op_apply);
  CeedCompositeOperatorAddSub(op_apply, op_mass);
  CeedCompositeOperatorAddSub(op_apply, op_diff);

  // Assemble diagonal
  CeedVectorCreate(ceed, num_dofs, &assembled);
  CeedOperatorLinearAssembleDiagonal(op_apply, assembled, CEED_REQUEST_IMMEDIATE);

  // Manually assemble diagonal
  CeedVectorSetValue(u, 0.0);
  for (int i = 0; i < num_dofs; i++) {
    CeedScalar       *u_array;
    const CeedScalar *v_array;

    // Set input
    CeedVectorGetArray(u, CEED_MEM_HOST, &u_array);
    u_array[i] = 1.0;
    if (i) u_array[i - 1] = 0.0;
    CeedVectorRestoreArray(u, &u_array);

    // Compute diag entry for DoF i
    CeedOperatorApply(op_apply, u, v, CEED_REQUEST_IMMEDIATE);

    // Retrieve entry
    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    assembled_true[i] = v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Check output
  {
    const CeedScalar *assembled_array;

    CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
    for (int i = 0; i < num_dofs; i++) {
      if (fabs(assembled_array[i] - assembled_true[i]) > 1000. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT "] Error in assembly: %f != %f\n", i, assembled_array[i], assembled_true[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(assembled, &assembled_array);
  }

  // Cleanup
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&assembled);
  CeedVectorDestroy(&q_data_mass);
  CeedVectorDestroy(&q_data_diff);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data_mass);
  CeedElemRestrictionDestroy(&elem_restriction_q_data_diff);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_setup_diff);
  CeedQFunctionDestroy(&qf_diff);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_setup_diff);
  CeedOperatorDestroy(&op_mass);
  CeedOperatorDestroy(&op_diff);
  CeedOperatorDestroy(&op_apply);
  CeedDestroy(&ceed);
  return 0;
}
