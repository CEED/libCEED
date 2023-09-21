/// @file
/// Test assembly of operator diagonal for operator with multiple active bases
/// \test Test assembly of operator diagonal for operator with multiple active bases
#include "t539-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u_0, elem_restriction_u_1, elem_restr_qd_mass, elem_restr_qd_diff;
  CeedBasis           basis_x, basis_u_0, basis_u_1;
  CeedQFunction       qf_setup_mass, qf_setup_diff, qf_apply;
  CeedOperator        op_setup_mass, op_setup_diff, op_apply;
  CeedVector          q_data_mass, q_data_diff, x, assembled, u, v;
  CeedInt             num_elem = 6, p_0 = 2, p_1 = 3, q = 4, dim = 2, num_comp_0 = 2, num_comp_1 = 1;
  CeedInt             n_x = 3, n_y = 2;
  CeedInt             num_dofs_0 = (n_x * (p_0 - 1) + 1) * (n_y * (p_0 - 1) + 1), num_dofs_1 = (n_x * (p_1 - 1) + 1) * (n_y * (p_1 - 1) + 1);
  CeedInt             num_qpts = num_elem * q * q;
  CeedInt             ind_u_0[num_elem * p_0 * p_0], ind_u_1[num_elem * p_1 * p_1];
  CeedScalar          assembled_true[num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs_0, &x);
  {
    CeedScalar x_array[dim * num_dofs_0];

    for (CeedInt i = 0; i < n_x * (p_0 - 1) + 1; i++) {
      for (CeedInt j = 0; j < n_y * (p_0 - 1) + 1; j++) {
        x_array[i + j * (n_x * (p_0 - 1) + 1) + 0 * num_dofs_0] = (CeedScalar)i / ((p_0 - 1) * n_x);
        x_array[i + j * (n_x * (p_0 - 1) + 1) + 1 * num_dofs_0] = (CeedScalar)j / ((p_0 - 1) * n_y);
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1, &u);
  CeedVectorCreate(ceed, num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1, &v);
  CeedVectorCreate(ceed, num_qpts, &q_data_mass);
  CeedVectorCreate(ceed, num_qpts * dim * (dim + 1) / 2, &q_data_diff);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;
    col    = i % n_x;
    row    = i / n_x;
    offset = col * (p_0 - 1) + row * (n_x * (p_0 - 1) + 1) * (p_0 - 1);
    for (CeedInt j = 0; j < p_0; j++) {
      for (CeedInt k = 0; k < p_0; k++) ind_u_0[p_0 * (p_0 * i + k) + j] = offset + k * (n_x * (p_0 - 1) + 1) + j;
    }
    offset = col * (p_1 - 1) + row * (n_x * (p_1 - 1) + 1) * (p_1 - 1) + num_dofs_0 * num_comp_0;
    for (CeedInt j = 0; j < p_1; j++) {
      for (CeedInt k = 0; k < p_1; k++) ind_u_1[p_1 * (p_1 * i + k) + j] = offset + k * (n_x * (p_1 - 1) + 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p_0 * p_0, dim, num_dofs_0, dim * num_dofs_0, CEED_MEM_HOST, CEED_USE_POINTER, ind_u_0,
                            &elem_restriction_x);
  CeedElemRestrictionCreate(ceed, num_elem, p_0 * p_0, num_comp_0, num_dofs_0, num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1, CEED_MEM_HOST,
                            CEED_USE_POINTER, ind_u_0, &elem_restriction_u_0);
  CeedElemRestrictionCreate(ceed, num_elem, p_1 * p_1, num_comp_1, num_dofs_1, num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1, CEED_MEM_HOST,
                            CEED_USE_POINTER, ind_u_1, &elem_restriction_u_1);

  CeedInt strides_q_data_mass[3] = {1, q * q, q * q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, 1, num_qpts, strides_q_data_mass, &elem_restr_qd_mass);

  CeedInt strides_q_data_diff[3] = {1, q * q, dim * (dim + 1) / 2 * q * q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, dim * (dim + 1) / 2, dim * (dim + 1) / 2 * num_qpts, strides_q_data_diff,
                                   &elem_restr_qd_diff);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p_0, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_0, p_0, q, CEED_GAUSS, &basis_u_0);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_1, p_1, q, CEED_GAUSS, &basis_u_1);

  // QFunction - setup mass
  CeedQFunctionCreateInteriorByName(ceed, "Mass2DBuild", &qf_setup_mass);

  // Operator - setup mass
  CeedOperatorCreate(ceed, qf_setup_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_mass);
  CeedOperatorSetField(op_setup_mass, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_mass, "weights", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_mass, "qdata", elem_restr_qd_mass, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  // QFunction - setup diffusion
  CeedQFunctionCreateInteriorByName(ceed, "Poisson2DBuild", &qf_setup_diff);

  // Operator - setup diffusion
  CeedOperatorCreate(ceed, qf_setup_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_diff);
  CeedOperatorSetField(op_setup_diff, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_diff, "weights", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_diff, "qdata", elem_restr_qd_diff, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  // Apply Setup Operators
  CeedOperatorApply(op_setup_mass, x, q_data_mass, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_diff, x, q_data_diff, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, apply, apply_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "du_0", num_comp_0 * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_apply, "mass qdata", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply, "diff qdata", dim * (dim + 1) / 2, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply, "u_0", num_comp_0, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_apply, "u_1", num_comp_1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply, "v_0", num_comp_0, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply, "v_1", num_comp_1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply, "dv_0", num_comp_0 * dim, CEED_EVAL_GRAD);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);
  CeedOperatorSetField(op_apply, "du_0", elem_restriction_u_0, basis_u_0, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "mass qdata", elem_restr_qd_mass, CEED_BASIS_NONE, q_data_mass);
  CeedOperatorSetField(op_apply, "diff qdata", elem_restr_qd_diff, CEED_BASIS_NONE, q_data_diff);
  CeedOperatorSetField(op_apply, "u_0", elem_restriction_u_0, basis_u_0, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "u_1", elem_restriction_u_1, basis_u_1, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "v_0", elem_restriction_u_0, basis_u_0, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "v_1", elem_restriction_u_1, basis_u_1, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "dv_0", elem_restriction_u_0, basis_u_0, CEED_VECTOR_ACTIVE);

  // Assemble diagonal
  CeedVectorCreate(ceed, num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1, &assembled);
  CeedOperatorLinearAssembleDiagonal(op_apply, assembled, CEED_REQUEST_IMMEDIATE);

  // Manually assemble diagonal
  CeedVectorSetValue(u, 0.0);
  for (int i = 0; i < num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1; i++) assembled_true[i] = 0.0;
  for (int i = 0; i < num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1; i++) {
    // Set input
    CeedScalar       *u_array;
    const CeedScalar *v_array;

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
    for (int i = 0; i < num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1; i++) {
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
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&q_data_mass);
  CeedVectorDestroy(&q_data_diff);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_u_0);
  CeedElemRestrictionDestroy(&elem_restriction_u_1);
  CeedElemRestrictionDestroy(&elem_restr_qd_mass);
  CeedElemRestrictionDestroy(&elem_restr_qd_diff);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u_0);
  CeedBasisDestroy(&basis_u_1);
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_setup_diff);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_setup_diff);
  CeedOperatorDestroy(&op_apply);
  CeedDestroy(&ceed);
  return 0;
}
