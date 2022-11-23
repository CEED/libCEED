/// @file
/// Test assembly of Poisson operator QFunction
/// \test Test assembly of Poisson operator QFunction
#include "t531-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_qd_i, elem_restr_lin_i;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_diff, qf_diff_lin;
  CeedOperator        op_setup, op_diff, op_diff_lin;
  CeedVector          q_data, X, A, u, v;
  CeedInt             num_elem = 6, P = 3, Q = 4, dim = 2;
  CeedInt             nx = 3, ny = 2;
  CeedInt             num_dofs = (nx * 2 + 1) * (ny * 2 + 1), num_qpts = num_elem * Q * Q;
  CeedInt             ind_x[num_elem * P * P];
  CeedScalar          x[dim * num_dofs];

  CeedInit(argv[1], &ceed);

  // DoF Coordinates
  for (CeedInt i = 0; i < nx * 2 + 1; i++) {
    for (CeedInt j = 0; j < ny * 2 + 1; j++) {
      x[i + j * (nx * 2 + 1) + 0 * num_dofs] = (CeedScalar)i / (2 * nx);
      x[i + j * (nx * 2 + 1) + 1 * num_dofs] = (CeedScalar)j / (2 * ny);
    }
  }
  CeedVectorCreate(ceed, dim * num_dofs, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vector
  CeedVectorCreate(ceed, num_qpts * dim * (dim + 1) / 2, &q_data);

  // Element Setup
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;
    col    = i % nx;
    row    = i / nx;
    offset = col * (P - 1) + row * (nx * 2 + 1) * (P - 1);
    for (CeedInt j = 0; j < P; j++) {
      for (CeedInt k = 0; k < P; k++) ind_x[P * (P * i + k) + j] = offset + k * (nx * 2 + 1) + j;
    }
  }

  // Restrictions
  CeedElemRestrictionCreate(ceed, num_elem, P * P, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_x);

  CeedElemRestrictionCreate(ceed, num_elem, P * P, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_u);
  CeedInt strides_qd[3] = {1, Q * Q, Q * Q * dim * (dim + 1) / 2};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * Q, dim * (dim + 1) / 2, dim * (dim + 1) / 2 * num_qpts, strides_qd, &elem_restr_qd_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &basis_u);

  // QFunction - setup
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", dim * (dim + 1) / 2, CEED_EVAL_NONE);

  // Operator - setup
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, X, q_data, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, diff, diff_loc, &qf_diff);
  CeedQFunctionAddInput(qf_diff, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_diff, "qdata", dim * (dim + 1) / 2, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_diff, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_diff);
  CeedOperatorSetField(op_diff, "du", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff, "qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_diff, "dv", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply original Poisson Operator
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorSetValue(u, 1.0);
  CeedVectorCreate(ceed, num_dofs, &v);
  CeedVectorSetValue(v, 0.0);
  CeedOperatorApply(op_diff, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  const CeedScalar *vv;
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &vv);
  for (CeedInt i = 0; i < num_dofs; i++) {
    if (fabs(vv[i]) > 100. * CEED_EPSILON) printf("Error: Operator computed v[%" CeedInt_FMT "] = %f != 0.0\n", i, vv[i]);
  }
  CeedVectorRestoreArrayRead(v, &vv);

  // Assemble QFunction
  CeedOperatorSetQFunctionAssemblyReuse(op_diff, true);
  CeedOperatorLinearAssembleQFunction(op_diff, &A, &elem_restr_lin_i, CEED_REQUEST_IMMEDIATE);
  // Second call will be no-op since SetQFunctionUpdated was not called
  CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(op_diff, false);
  CeedOperatorLinearAssembleQFunction(op_diff, &A, &elem_restr_lin_i, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply assembled
  CeedQFunctionCreateInterior(ceed, 1, diff_lin, diff_lin_loc, &qf_diff_lin);
  CeedQFunctionAddInput(qf_diff_lin, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_diff_lin, "qdata", dim * dim, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_diff_lin, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply assembled
  CeedOperatorCreate(ceed, qf_diff_lin, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_diff_lin);
  CeedOperatorSetField(op_diff_lin, "du", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff_lin, "qdata", elem_restr_lin_i, CEED_BASIS_COLLOCATED, A);
  CeedOperatorSetField(op_diff_lin, "dv", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply new Poisson Operator
  CeedVectorSetValue(v, 0.0);
  CeedOperatorApply(op_diff_lin, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &vv);
  for (CeedInt i = 0; i < num_dofs; i++) {
    if (fabs(vv[i]) > 100. * CEED_EPSILON) printf("Error: Linearized operator computed v[i] = %f != 0.0\n", vv[i]);
  }
  CeedVectorRestoreArrayRead(v, &vv);

  // Cleanup
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_diff);
  CeedQFunctionDestroy(&qf_diff_lin);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_diff);
  CeedOperatorDestroy(&op_diff_lin);
  CeedElemRestrictionDestroy(&elem_restr_u);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_qd_i);
  CeedElemRestrictionDestroy(&elem_restr_lin_i);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&A);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedDestroy(&ceed);
  return 0;
}
