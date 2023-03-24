/// @file
/// Test creation reuse of the same QFunction for multiple operators
/// \test Test creation reuse of the same QFunction for multiple operators
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t502-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data_small, elem_restriction_q_data_large;
  CeedBasis           basis_x_small, basis_x_large, basis_u_small, basis_u_large;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup_small, op_mass_small, op_setup_large, op_mass_large;
  CeedVector          q_data_small, q_data_large, x, u, v;
  CeedInt             num_elem = 15, p = 5, q = 8, scale = 3;
  CeedInt             num_nodes_x = num_elem + 1, num_nodes_u = num_elem * (p - 1) + 1;
  CeedInt             ind_x[num_elem * 2], ind_u[num_elem * p];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_nodes_x, &x);
  {
    CeedScalar x_array[num_nodes_x];

    for (CeedInt i = 0; i < num_nodes_x; i++) x_array[i] = (CeedScalar)i / (num_nodes_x - 1);
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, 2 * num_nodes_u, &u);
  CeedVectorCreate(ceed, 2 * num_nodes_u, &v);
  CeedVectorCreate(ceed, num_elem * q, &q_data_small);
  CeedVectorCreate(ceed, num_elem * q * scale, &q_data_large);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_nodes_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p; j++) {
      ind_u[p * i + j] = 2 * (i * (p - 1) + j);
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p, 2, 1, 2 * num_nodes_u, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restriction_u);

  CeedInt strides_q_data_small[3] = {1, q, q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, q * num_elem, strides_q_data_small, &elem_restriction_q_data_small);

  CeedInt strides_q_data_large[3] = {1, q * scale, q * scale};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * scale, 1, q * num_elem * scale, strides_q_data_large, &elem_restriction_q_data_large);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x_small);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 2, p, q, CEED_GAUSS, &basis_u_small);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q * scale, CEED_GAUSS, &basis_x_large);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 2, p, q * scale, CEED_GAUSS, &basis_u_large);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "x", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 2, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 2, CEED_EVAL_INTERP);

  // 'Small' Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_small);
  CeedOperatorSetField(op_setup_small, "weight", CEED_ELEMRESTRICTION_NONE, basis_x_small, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_small, "x", elem_restriction_x, basis_x_small, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_small, "rho", elem_restriction_q_data_small, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_small);
  CeedOperatorSetField(op_mass_small, "rho", elem_restriction_q_data_small, CEED_BASIS_COLLOCATED, q_data_small);
  CeedOperatorSetField(op_mass_small, "u", elem_restriction_u, basis_u_small, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_small, "v", elem_restriction_u, basis_u_small, CEED_VECTOR_ACTIVE);

  // 'Large' operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_large);
  CeedOperatorSetField(op_setup_large, "weight", CEED_ELEMRESTRICTION_NONE, basis_x_large, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_large, "x", elem_restriction_x, basis_x_large, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_large, "rho", elem_restriction_q_data_large, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_large);
  CeedOperatorSetField(op_mass_large, "rho", elem_restriction_q_data_large, CEED_BASIS_COLLOCATED, q_data_large);
  CeedOperatorSetField(op_mass_large, "u", elem_restriction_u, basis_u_large, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_large, "v", elem_restriction_u, basis_u_large, CEED_VECTOR_ACTIVE);

  // Setup
  CeedOperatorApply(op_setup_small, x, q_data_small, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_large, x, q_data_large, CEED_REQUEST_IMMEDIATE);

  {
    CeedScalar *u_array;

    CeedVectorGetArrayWrite(u, CEED_MEM_HOST, &u_array);
    for (int i = 0; i < num_nodes_u; i++) {
      u_array[2 * i]     = 1.0;
      u_array[2 * i + 1] = 2.0;
    }
    CeedVectorRestoreArray(u, &u_array);
  }

  // 'Small' operator
  CeedOperatorApply(op_mass_small, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        sum_1 = 0., sum_2 = 0.;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_nodes_u; i++) {
      sum_1 += v_array[2 * i];
      sum_2 += v_array[2 * i + 1];
    }
    CeedVectorRestoreArrayRead(v, &v_array);
    if (fabs(sum_1 - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum_1);
    if (fabs(sum_2 - 2.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 2.0\n", sum_2);
  }

  // 'Large' operator
  CeedOperatorApply(op_mass_large, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        sum_1 = 0., sum_2 = 0.;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_nodes_u; i++) {
      sum_1 += v_array[2 * i];
      sum_2 += v_array[2 * i + 1];
    }
    CeedVectorRestoreArrayRead(v, &v_array);

    if (fabs(sum_1 - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum_1);
    if (fabs(sum_2 - 2.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 2.0\n", sum_2);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&q_data_small);
  CeedVectorDestroy(&q_data_large);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data_small);
  CeedElemRestrictionDestroy(&elem_restriction_q_data_large);
  CeedBasisDestroy(&basis_u_small);
  CeedBasisDestroy(&basis_x_small);
  CeedBasisDestroy(&basis_u_large);
  CeedBasisDestroy(&basis_x_large);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup_small);
  CeedOperatorDestroy(&op_mass_small);
  CeedOperatorDestroy(&op_setup_large);
  CeedOperatorDestroy(&op_mass_large);
  CeedDestroy(&ceed);
  return 0;
}
