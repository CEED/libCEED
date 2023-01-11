/// @file
/// Test creation reuse of the same QFunction for multiple operators
/// \test Test creation reuse of the same QFunction for multiple operators
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t502-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_qd_i_small, elem_restr_qd_i_large;
  CeedBasis           basis_x_small, basis_x_large, basis_u_small, basis_u_large;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup_small, op_mass_small, op_setup_large, op_mass_large;
  CeedVector          q_data_small, q_data_large, X, U, V;
  CeedScalar         *hu;
  const CeedScalar   *hv;
  CeedInt             num_elem = 15, P = 5, Q = 8, scale = 3;
  CeedInt             num_nodes_x = num_elem + 1, num_nodes_u = num_elem * (P - 1) + 1;
  CeedInt             ind_x[num_elem * 2], ind_u[num_elem * P];
  CeedScalar          x[num_nodes_x];
  CeedScalar          sum_1, sum_2;

  CeedInit(argv[1], &ceed);
  for (CeedInt i = 0; i < num_nodes_x; i++) x[i] = (CeedScalar)i / (num_nodes_x - 1);
  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  // Restrictions
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_nodes_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_x);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < P; j++) {
      ind_u[P * i + j] = 2 * (i * (P - 1) + j);
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, P, 2, 1, 2 * num_nodes_u, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restr_u);
  CeedInt strides_qd_small[3] = {1, Q, Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q, 1, Q * num_elem, strides_qd_small, &elem_restr_qd_i_small);
  CeedInt strides_qd_large[3] = {1, Q * scale, Q * scale};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * scale, 1, Q * num_elem * scale, strides_qd_large, &elem_restr_qd_i_large);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &basis_x_small);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 2, P, Q, CEED_GAUSS, &basis_u_small);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q * scale, CEED_GAUSS, &basis_x_large);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 2, P, Q * scale, CEED_GAUSS, &basis_u_large);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "x", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 2, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 2, CEED_EVAL_INTERP);

  // Input vector
  CeedVectorCreate(ceed, num_nodes_x, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // 'Small' Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_small);
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_small);

  CeedVectorCreate(ceed, num_elem * Q, &q_data_small);

  CeedOperatorSetField(op_setup_small, "weight", CEED_ELEMRESTRICTION_NONE, basis_x_small, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_small, "x", elem_restr_x, basis_x_small, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_small, "rho", elem_restr_qd_i_small, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass_small, "rho", elem_restr_qd_i_small, CEED_BASIS_COLLOCATED, q_data_small);
  CeedOperatorSetField(op_mass_small, "u", elem_restr_u, basis_u_small, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_small, "v", elem_restr_u, basis_u_small, CEED_VECTOR_ACTIVE);

  // 'Large' operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_large);
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_large);

  CeedVectorCreate(ceed, num_elem * Q * scale, &q_data_large);

  CeedOperatorSetField(op_setup_large, "weight", CEED_ELEMRESTRICTION_NONE, basis_x_large, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_large, "x", elem_restr_x, basis_x_large, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_large, "rho", elem_restr_qd_i_large, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass_large, "rho", elem_restr_qd_i_large, CEED_BASIS_COLLOCATED, q_data_large);
  CeedOperatorSetField(op_mass_large, "u", elem_restr_u, basis_u_large, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_large, "v", elem_restr_u, basis_u_large, CEED_VECTOR_ACTIVE);

  // Setup
  CeedOperatorApply(op_setup_small, X, q_data_small, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_large, X, q_data_large, CEED_REQUEST_IMMEDIATE);

  CeedVectorCreate(ceed, 2 * num_nodes_u, &U);
  CeedVectorGetArrayWrite(U, CEED_MEM_HOST, &hu);
  for (int i = 0; i < num_nodes_u; i++) {
    hu[2 * i]     = 1.0;
    hu[2 * i + 1] = 2.0;
  }
  CeedVectorRestoreArray(U, &hu);
  CeedVectorCreate(ceed, 2 * num_nodes_u, &V);

  // 'Small' operator
  CeedOperatorApply(op_mass_small, U, V, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  sum_1 = 0.;
  sum_2 = 0.;
  for (CeedInt i = 0; i < num_nodes_u; i++) {
    sum_1 += hv[2 * i];
    sum_2 += hv[2 * i + 1];
  }
  if (fabs(sum_1 - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum_1);
  if (fabs(sum_2 - 2.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 2.0\n", sum_2);
  CeedVectorRestoreArrayRead(V, &hv);

  // 'Large' operator
  CeedOperatorApply(op_mass_large, U, V, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  sum_1 = 0.;
  sum_2 = 0.;
  for (CeedInt i = 0; i < num_nodes_u; i++) {
    sum_1 += hv[2 * i];
    sum_2 += hv[2 * i + 1];
  }
  if (fabs(sum_1 - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum_1);
  if (fabs(sum_2 - 2.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 2.0\n", sum_2);
  CeedVectorRestoreArrayRead(V, &hv);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup_small);
  CeedOperatorDestroy(&op_mass_small);
  CeedOperatorDestroy(&op_setup_large);
  CeedOperatorDestroy(&op_mass_large);
  CeedElemRestrictionDestroy(&elem_restr_u);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_qd_i_small);
  CeedElemRestrictionDestroy(&elem_restr_qd_i_large);
  CeedBasisDestroy(&basis_u_small);
  CeedBasisDestroy(&basis_x_small);
  CeedBasisDestroy(&basis_u_large);
  CeedBasisDestroy(&basis_x_large);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&q_data_small);
  CeedVectorDestroy(&q_data_large);
  CeedDestroy(&ceed);
  return 0;
}
