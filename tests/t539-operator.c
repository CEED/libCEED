/// @file
/// Test assembly of operator diagonal for operator with multiple active bases
/// \test Test assembly of operator diagonal for operator with multiple active bases
#include "t539-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x, elem_restr_u_0, elem_restr_u_1, elem_restr_qd_mass, elem_restr_qd_diff;
  CeedBasis           basis_x, basis_u_0, basis_u_1;
  CeedQFunction       qf_setup_mass, qf_setup_diff, qf_apply;
  CeedOperator        op_setup_mass, op_setup_diff, op_apply;
  CeedVector          q_data_mass, q_data_diff, X, A, U, V;
  CeedInt             num_elem = 6, P_0 = 2, P_1 = 3, Q = 4, dim = 2, num_comp_0 = 2, num_comp_1 = 1;
  CeedInt             nx = 3, ny = 2;
  CeedInt             num_dofs_0 = (nx * (P_0 - 1) + 1) * (ny * (P_0 - 1) + 1), num_dofs_1 = (nx * (P_1 - 1) + 1) * (ny * (P_1 - 1) + 1);
  CeedInt             num_qpts = num_elem * Q * Q;
  CeedInt             ind_u_0[num_elem * P_0 * P_0], ind_u_1[num_elem * P_1 * P_1];
  CeedScalar          x[dim * num_dofs_0], assembled_true[num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1];
  CeedScalar         *u;
  const CeedScalar   *a, *v;

  CeedInit(argv[1], &ceed);

  // DoF Coordinates
  for (CeedInt i = 0; i < nx * 2 + 1; i++) {
    for (CeedInt j = 0; j < ny * 2 + 1; j++) {
      x[i + j * (nx * 2 + 1) + 0 * num_dofs_0] = (CeedScalar)i / (2 * nx);
      x[i + j * (nx * 2 + 1) + 1 * num_dofs_0] = (CeedScalar)j / (2 * ny);
    }
  }
  CeedVectorCreate(ceed, dim * num_dofs_0, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vectors
  CeedVectorCreate(ceed, num_qpts, &q_data_mass);
  CeedVectorCreate(ceed, num_qpts * dim * (dim + 1) / 2, &q_data_diff);

  // Element Setup
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;
    col    = i % nx;
    row    = i / nx;
    offset = col * (P_0 - 1) + row * (nx * (P_0 - 1) + 1) * (P_0 - 1);
    for (CeedInt j = 0; j < P_0; j++) {
      for (CeedInt k = 0; k < P_0; k++) ind_u_0[P_0 * (P_0 * i + k) + j] = offset + k * (nx * (P_0 - 1) + 1) + j;
    }
    offset = col * (P_1 - 1) + row * (nx * (P_1 - 1) + 1) * (P_1 - 1) + num_dofs_0 * num_comp_0;
    for (CeedInt j = 0; j < P_1; j++) {
      for (CeedInt k = 0; k < P_1; k++) ind_u_1[P_1 * (P_1 * i + k) + j] = offset + k * (nx * (P_1 - 1) + 1) + j;
    }
  }

  // Restrictions
  CeedElemRestrictionCreate(ceed, num_elem, P_0 * P_0, dim, num_dofs_0, dim * num_dofs_0, CEED_MEM_HOST, CEED_USE_POINTER, ind_u_0, &elem_restr_x);
  CeedElemRestrictionCreate(ceed, num_elem, P_0 * P_0, num_comp_0, num_dofs_0, num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1, CEED_MEM_HOST,
                            CEED_USE_POINTER, ind_u_0, &elem_restr_u_0);
  CeedElemRestrictionCreate(ceed, num_elem, P_1 * P_1, num_comp_1, num_dofs_1, num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1, CEED_MEM_HOST,
                            CEED_USE_POINTER, ind_u_1, &elem_restr_u_1);
  CeedInt strides_qd_mass[3] = {1, Q * Q, Q * Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * Q, 1, num_qpts, strides_qd_mass, &elem_restr_qd_mass);
  CeedInt strides_qd_diff[3] = {1, Q * Q, dim * (dim + 1) / 2 * Q * Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * Q, dim * (dim + 1) / 2, dim * (dim + 1) / 2 * num_qpts, strides_qd_diff, &elem_restr_qd_diff);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P_0, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_0, P_0, Q, CEED_GAUSS, &basis_u_0);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_1, P_1, Q, CEED_GAUSS, &basis_u_1);

  // QFunction - setup mass
  CeedQFunctionCreateInteriorByName(ceed, "Mass2DBuild", &qf_setup_mass);

  // Operator - setup mass
  CeedOperatorCreate(ceed, qf_setup_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_mass);
  CeedOperatorSetField(op_setup_mass, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_mass, "weights", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_mass, "qdata", elem_restr_qd_mass, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // QFunction - setup diffusion
  CeedQFunctionCreateInteriorByName(ceed, "Poisson2DBuild", &qf_setup_diff);

  // Operator - setup diffusion
  CeedOperatorCreate(ceed, qf_setup_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_diff);
  CeedOperatorSetField(op_setup_diff, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_diff, "weights", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_diff, "qdata", elem_restr_qd_diff, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operators
  CeedOperatorApply(op_setup_mass, X, q_data_mass, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_diff, X, q_data_diff, CEED_REQUEST_IMMEDIATE);

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
  CeedOperatorSetField(op_apply, "du_0", elem_restr_u_0, basis_u_0, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "mass qdata", elem_restr_qd_mass, CEED_BASIS_COLLOCATED, q_data_mass);
  CeedOperatorSetField(op_apply, "diff qdata", elem_restr_qd_diff, CEED_BASIS_COLLOCATED, q_data_diff);
  CeedOperatorSetField(op_apply, "u_0", elem_restr_u_0, basis_u_0, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "u_1", elem_restr_u_1, basis_u_1, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "v_0", elem_restr_u_0, basis_u_0, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "v_1", elem_restr_u_1, basis_u_1, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "dv_0", elem_restr_u_0, basis_u_0, CEED_VECTOR_ACTIVE);

  // Assemble diagonal
  CeedVectorCreate(ceed, num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1, &A);
  CeedOperatorLinearAssembleDiagonal(op_apply, A, CEED_REQUEST_IMMEDIATE);

  // Manually assemble diagonal
  CeedVectorCreate(ceed, num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1, &U);
  CeedVectorSetValue(U, 0.0);
  CeedVectorCreate(ceed, num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1, &V);
  for (int i = 0; i < num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1; i++) assembled_true[i] = 0.0;
  for (int i = 0; i < num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1; i++) {
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
  for (int i = 0; i < num_comp_0 * num_dofs_0 + num_comp_1 * num_dofs_1; i++) {
    if (fabs(a[i] - assembled_true[i]) > 1000. * CEED_EPSILON) printf("[%" CeedInt_FMT "] Error in assembly: %f != %f\n", i, a[i], assembled_true[i]);
  }
  CeedVectorRestoreArrayRead(A, &a);

  // Cleanup
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_setup_diff);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_setup_diff);
  CeedOperatorDestroy(&op_apply);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_u_0);
  CeedElemRestrictionDestroy(&elem_restr_u_1);
  CeedElemRestrictionDestroy(&elem_restr_qd_mass);
  CeedElemRestrictionDestroy(&elem_restr_qd_diff);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u_0);
  CeedBasisDestroy(&basis_u_1);
  CeedVectorDestroy(&q_data_mass);
  CeedVectorDestroy(&q_data_diff);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&A);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
