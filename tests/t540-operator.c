/// @file
/// Test creation and use of FDM element inverse
/// \test Test creation and use of FDM element inverse
#include "t540-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x_i, elem_restr_u_i, elem_restr_qd_i;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup_mass, qf_apply;
  CeedOperator        op_setup_mass, op_apply, op_inv;
  CeedVector          q_data_mass, X, U, V;
  CeedInt             num_elem = 1, P = 4, Q = 5, dim = 2;
  CeedInt             num_dofs = P * P, num_qpts = num_elem * Q * Q;
  CeedScalar          x[dim * num_elem * (2 * 2)];
  const CeedScalar   *u;

  CeedInit(argv[1], &ceed);

  // DoF Coordinates
  for (CeedInt i = 0; i < 2; i++) {
    for (CeedInt j = 0; j < 2; j++) {
      x[i + j * 2 + 0 * 4] = i;
      x[i + j * 2 + 1 * 4] = j;
    }
  }
  CeedVectorCreate(ceed, dim * num_elem * (2 * 2), &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vector
  CeedVectorCreate(ceed, num_qpts, &q_data_mass);

  // Element Setup

  // Restrictions
  CeedInt strides_x[3] = {1, 2 * 2, 2 * 2 * dim};
  CeedElemRestrictionCreateStrided(ceed, num_elem, 2 * 2, dim, dim * num_elem * 2 * 2, strides_x, &elem_restr_x_i);

  CeedInt strides_u[3] = {1, P * P, P * P};
  CeedElemRestrictionCreateStrided(ceed, num_elem, P * P, 1, num_dofs, strides_u, &elem_restr_u_i);

  CeedInt strides_qd[3] = {1, Q * Q, Q * Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * Q, 1, num_qpts, strides_qd, &elem_restr_qd_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &basis_u);

  // QFunction - setup mass
  CeedQFunctionCreateInterior(ceed, 1, setup_mass, setup_mass_loc, &qf_setup_mass);
  CeedQFunctionAddInput(qf_setup_mass, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_mass, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_mass, "qdata", 1, CEED_EVAL_NONE);

  // Operator - setup mass
  CeedOperatorCreate(ceed, qf_setup_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_mass);
  CeedOperatorSetField(op_setup_mass, "dx", elem_restr_x_i, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_mass, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_mass, "qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup_mass, X, q_data_mass, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, apply, apply_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_apply, "mass qdata", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", 1, CEED_EVAL_INTERP);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);
  CeedOperatorSetField(op_apply, "u", elem_restr_u_i, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "mass qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data_mass);
  CeedOperatorSetField(op_apply, "v", elem_restr_u_i, basis_u, CEED_VECTOR_ACTIVE);

  // Apply original operator
  CeedVectorCreate(ceed, num_dofs, &U);
  CeedVectorSetValue(U, 1.0);
  CeedVectorCreate(ceed, num_dofs, &V);
  CeedVectorSetValue(V, 0.0);
  CeedOperatorApply(op_apply, U, V, CEED_REQUEST_IMMEDIATE);

  // Create FDM element inverse
  CeedOperatorCreateFDMElementInverse(op_apply, &op_inv, CEED_REQUEST_IMMEDIATE);

  // Apply FDM element inverse
  CeedOperatorApply(op_inv, V, U, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u);
  for (int i = 0; i < num_dofs; i++) {
    if (fabs(u[i] - 1.0) > 500. * CEED_EPSILON) printf("[%" CeedInt_FMT "] Error in inverse: %e - 1.0 = %e\n", i, u[i], u[i] - 1.);
  }
  CeedVectorRestoreArrayRead(U, &u);

  // Cleanup
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_apply);
  CeedOperatorDestroy(&op_inv);
  CeedElemRestrictionDestroy(&elem_restr_u_i);
  CeedElemRestrictionDestroy(&elem_restr_x_i);
  CeedElemRestrictionDestroy(&elem_restr_qd_i);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&q_data_mass);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
