/// @file
/// Test full assembly of mass and Poisson operator (see t536)
/// \test Test full assembly of mass and Poisson operator
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t320-basis.h"
#include "t535-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data_mass, elem_restriction_q_data_diff;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup_mass, qf_setup_diff, qf_apply;
  CeedOperator        op_setup_mass, op_setup_diff, op_apply;
  CeedVector          q_data_mass, q_data_diff, x, u, v;
  CeedInt             num_elem = 12, dim = 2, p = 6, q = 4;
  CeedInt             n_x = 3, n_y = 2;
  CeedInt             row, col, offset;
  CeedInt             num_dofs = (n_x * 2 + 1) * (n_y * 2 + 1), num_qpts = num_elem * q;
  CeedInt             ind_x[num_elem * p * p];
  CeedScalar          assembled_values[num_dofs * num_dofs];
  CeedScalar          assembled_true[num_dofs * num_dofs];
  CeedScalar          q_ref[dim * q], q_weight[q];
  CeedScalar          interp[p * q], grad[dim * p * q];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs, &x);
  {
    CeedScalar x_array[dim * num_dofs];

    for (CeedInt i = 0; i < num_dofs; i++) {
      x_array[i]            = (1. / (n_x * 2)) * (CeedScalar)(i % (n_x * 2 + 1));
      x_array[i + num_dofs] = (1. / (n_y * 2)) * (CeedScalar)(i / (n_x * 2 + 1));
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorCreate(ceed, num_dofs, &v);
  CeedVectorCreate(ceed, num_qpts, &q_data_mass);
  CeedVectorCreate(ceed, num_qpts * dim * (dim + 1) / 2, &q_data_diff);

  // Restrictions
  for (CeedInt i = 0; i < num_elem / 2; i++) {
    col    = i % n_x;
    row    = i / n_x;
    offset = col * 2 + row * (n_x * 2 + 1) * 2;

    ind_x[i * 2 * p + 0] = 2 + offset;
    ind_x[i * 2 * p + 1] = 9 + offset;
    ind_x[i * 2 * p + 2] = 16 + offset;
    ind_x[i * 2 * p + 3] = 1 + offset;
    ind_x[i * 2 * p + 4] = 8 + offset;
    ind_x[i * 2 * p + 5] = 0 + offset;

    ind_x[i * 2 * p + 6]  = 14 + offset;
    ind_x[i * 2 * p + 7]  = 7 + offset;
    ind_x[i * 2 * p + 8]  = 0 + offset;
    ind_x[i * 2 * p + 9]  = 15 + offset;
    ind_x[i * 2 * p + 10] = 8 + offset;
    ind_x[i * 2 * p + 11] = 16 + offset;
  }
  CeedElemRestrictionCreate(ceed, num_elem, p, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
  CeedElemRestrictionCreate(ceed, num_elem, p, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_u);

  CeedInt strides_q_data_mass[3] = {1, q, q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, num_qpts, strides_q_data_mass, &elem_restriction_q_data_mass);

  CeedInt strides_q_data_diff[3] = {1, q, q * dim * (dim + 1) / 2};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, dim * (dim + 1) / 2, dim * (dim + 1) / 2 * num_qpts, strides_q_data_diff,
                                   &elem_restriction_q_data_diff);

  // Bases
  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, dim, p, q, interp, grad, q_ref, q_weight, &basis_x);

  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, 1, p, q, interp, grad, q_ref, q_weight, &basis_u);

  // QFunction - setup mass
  CeedQFunctionCreateInterior(ceed, 1, setup_mass, setup_mass_loc, &qf_setup_mass);
  CeedQFunctionAddInput(qf_setup_mass, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_mass, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_mass, "q data", 1, CEED_EVAL_NONE);

  // Operator - setup mass
  CeedOperatorCreate(ceed, qf_setup_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_mass);
  CeedOperatorSetField(op_setup_mass, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_mass, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_mass, "q data", elem_restriction_q_data_mass, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // QFunction - setup diff
  CeedQFunctionCreateInterior(ceed, 1, setup_diff, setup_diff_loc, &qf_setup_diff);
  CeedQFunctionAddInput(qf_setup_diff, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_diff, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_diff, "q data", dim * (dim + 1) / 2, CEED_EVAL_NONE);

  // Operator - setup diff
  CeedOperatorCreate(ceed, qf_setup_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_diff);
  CeedOperatorSetField(op_setup_diff, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_diff, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_diff, "q data", elem_restriction_q_data_diff, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operators
  CeedOperatorApply(op_setup_mass, x, q_data_mass, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_diff, x, q_data_diff, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, apply, apply_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_apply, "mass q data", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply, "diff q data", dim * (dim + 1) / 2, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);
  CeedOperatorSetField(op_apply, "du", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "mass q data", elem_restriction_q_data_mass, CEED_BASIS_COLLOCATED, q_data_mass);
  CeedOperatorSetField(op_apply, "diff q data", elem_restriction_q_data_diff, CEED_BASIS_COLLOCATED, q_data_diff);
  CeedOperatorSetField(op_apply, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "dv", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Fully assemble operator
  CeedSize   num_entries;
  CeedInt   *rows;
  CeedInt   *cols;
  CeedVector assembled;

  for (CeedInt k = 0; k < num_dofs * num_dofs; ++k) {
    assembled_values[k] = 0.0;
    assembled_true[k]   = 0.0;
  }
  CeedOperatorLinearAssembleSymbolic(op_apply, &num_entries, &rows, &cols);
  CeedVectorCreate(ceed, num_entries, &assembled);
  CeedOperatorLinearAssemble(op_apply, assembled);
  {
    const CeedScalar *assembled_array;
    CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
    for (CeedInt k = 0; k < num_entries; ++k) assembled_values[rows[k] * num_dofs + cols[k]] += assembled_array[k];
    CeedVectorRestoreArrayRead(assembled, &assembled_array);
  }

  // Manually assemble operator
  CeedVectorSetValue(u, 0.0);
  for (CeedInt i = 0; i < num_dofs; i++) {
    CeedScalar       *u_array;
    const CeedScalar *v_array;

    // Set input
    CeedVectorGetArray(u, CEED_MEM_HOST, &u_array);
    u_array[i] = 1.0;
    if (i) u_array[i - 1] = 0.0;
    CeedVectorRestoreArray(u, &u_array);

    // Compute entries for column i
    CeedOperatorApply(op_apply, u, v, CEED_REQUEST_IMMEDIATE);

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt k = 0; k < num_dofs; k++) assembled_true[i * num_dofs + k] = v_array[k];
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Check output
  for (CeedInt i = 0; i < num_dofs; i++) {
    for (CeedInt j = 0; j < num_dofs; j++) {
      if (fabs(assembled_values[j * num_dofs + i] - assembled_true[j * num_dofs + i]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT ", %" CeedInt_FMT "] Error in assembly: %f != %f\n", i, j, assembled_values[j * num_dofs + i],
               assembled_true[j * num_dofs + i]);
        // LCOV_EXCL_STOP
      }
    }
  }

  // Cleanup
  free(rows);
  free(cols);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&q_data_mass);
  CeedVectorDestroy(&q_data_diff);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&assembled);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data_mass);
  CeedElemRestrictionDestroy(&elem_restriction_q_data_diff);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_setup_diff);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_setup_diff);
  CeedOperatorDestroy(&op_apply);
  CeedDestroy(&ceed);
  return 0;
}
