/// @file
/// Test assembly of mass and Poisson operator diagonal
/// \test Test assembly of mass and Poisson operator diagonal
#include "t535-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_qd_mass_i, elem_restr_qd_diff_i;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup_mass, qf_setup_diff, qf_apply;
  CeedOperator        op_setup_mass, op_setup_diff, op_apply;
  CeedVector          q_data_mass, q_data_diff, X, A, U, V;
  CeedInt             num_elem = 6, P = 3, Q = 4, dim = 2;
  CeedInt             n_x = 3, n_y = 2;
  CeedInt             num_dofs = (n_x * 2 + 1) * (n_y * 2 + 1), num_qpts = num_elem * Q * Q;
  CeedInt             ind_x[num_elem * P * P];
  CeedScalar          x[dim * num_dofs], assembled_true[num_dofs];
  CeedScalar         *u;
  const CeedScalar   *a, *v;

  CeedInit(argv[1], &ceed);

  // DoF Coordinates
  for (CeedInt i = 0; i < n_x * 2 + 1; i++) {
    for (CeedInt j = 0; j < n_y * 2 + 1; j++) {
      x[i + j * (n_x * 2 + 1) + 0 * num_dofs] = (CeedScalar)i / (2 * n_x);
      x[i + j * (n_x * 2 + 1) + 1 * num_dofs] = (CeedScalar)j / (2 * n_y);
    }
  }
  CeedVectorCreate(ceed, dim * num_dofs, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vectors
  CeedVectorCreate(ceed, num_qpts, &q_data_mass);
  CeedVectorCreate(ceed, num_qpts * dim * (dim + 1) / 2, &q_data_diff);

  // Element Setup
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;
    col    = i % n_x;
    row    = i / n_x;
    offset = col * (P - 1) + row * (n_x * 2 + 1) * (P - 1);
    for (CeedInt j = 0; j < P; j++) {
      for (CeedInt k = 0; k < P; k++) ind_x[P * (P * i + k) + j] = offset + k * (n_x * 2 + 1) + j;
    }
  }

  // Restrictions
  CeedElemRestrictionCreate(ceed, num_elem, P * P, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_x);

  CeedElemRestrictionCreate(ceed, num_elem, P * P, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_u);
  CeedInt strides_qd_mass[3] = {1, Q * Q, Q * Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * Q, 1, num_qpts, strides_qd_mass, &elem_restr_qd_mass_i);

  CeedInt strides_qd_diff[3] = {1, Q * Q, Q * Q * dim * (dim + 1) / 2};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * Q, dim * (dim + 1) / 2, dim * (dim + 1) / 2 * num_qpts, strides_qd_diff,
                                   &elem_restr_qd_diff_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &basis_u);

  // QFunction - setup mass
  CeedQFunctionCreateInterior(ceed, 1, setup_mass, setup_mass_loc, &qf_setup_mass);
  CeedQFunctionAddInput(qf_setup_mass, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_mass, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_mass, "qdata", 1, CEED_EVAL_NONE);

  // Operator - setup mass
  CeedOperatorCreate(ceed, qf_setup_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_mass);
  CeedOperatorSetField(op_setup_mass, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_mass, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_mass, "qdata", elem_restr_qd_mass_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // QFunction - setup diff
  CeedQFunctionCreateInterior(ceed, 1, setup_diff, setup_diff_loc, &qf_setup_diff);
  CeedQFunctionAddInput(qf_setup_diff, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_diff, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_diff, "qdata", dim * (dim + 1) / 2, CEED_EVAL_NONE);

  // Operator - setup diff
  CeedOperatorCreate(ceed, qf_setup_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_diff);
  CeedOperatorSetField(op_setup_diff, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_diff, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_diff, "qdata", elem_restr_qd_diff_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operators
  CeedOperatorApply(op_setup_mass, X, q_data_mass, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_diff, X, q_data_diff, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, apply, apply_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_apply, "mass qdata", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply, "diff qdata", dim * (dim + 1) / 2, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);
  CeedOperatorSetField(op_apply, "du", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "mass qdata", elem_restr_qd_mass_i, CEED_BASIS_COLLOCATED, q_data_mass);
  CeedOperatorSetField(op_apply, "diff qdata", elem_restr_qd_diff_i, CEED_BASIS_COLLOCATED, q_data_diff);
  CeedOperatorSetField(op_apply, "u", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "v", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "dv", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  // Assemble diagonal
  CeedVectorCreate(ceed, num_dofs, &A);
  CeedOperatorLinearAssembleDiagonal(op_apply, A, CEED_REQUEST_IMMEDIATE);

  // Manually assemble diagonal
  CeedVectorCreate(ceed, num_dofs, &U);
  CeedVectorSetValue(U, 0.0);
  CeedVectorCreate(ceed, num_dofs, &V);
  for (int i = 0; i < num_dofs; i++) {
    // Set input
    CeedVectorGetArray(U, CEED_MEM_HOST, &u);
    u[i] = 1.0;
    if (i) u[i - 1] = 0.0;
    CeedVectorRestoreArray(U, &u);

    // Compute diag entry for DoF i
    CeedOperatorApply(op_apply, U, V, CEED_REQUEST_IMMEDIATE);

    // Retrieve entry
    CeedVectorGetArrayRead(V, CEED_MEM_HOST, &v);
    assembled_true[i] = v[i];
    CeedVectorRestoreArrayRead(V, &v);
  }

  // Check output
  CeedVectorGetArrayRead(A, CEED_MEM_HOST, &a);
  for (int i = 0; i < num_dofs; i++) {
    if (fabs(a[i] - assembled_true[i]) > 100. * CEED_EPSILON) printf("[%" CeedInt_FMT "] Error in assembly: %f != %f\n", i, a[i], assembled_true[i]);
  }
  CeedVectorRestoreArrayRead(A, &a);

  // Cleanup
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_setup_diff);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_setup_diff);
  CeedOperatorDestroy(&op_apply);
  CeedElemRestrictionDestroy(&elem_restr_u);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_qd_mass_i);
  CeedElemRestrictionDestroy(&elem_restr_qd_diff_i);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&A);
  CeedVectorDestroy(&q_data_mass);
  CeedVectorDestroy(&q_data_diff);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
