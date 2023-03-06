/// @file
/// Test VLA macro for operator
/// \test VLA marco for operator
#include "t507-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, x, u, v;
  CeedInt             num_elem = 15, p = 5, q = 8;
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
  CeedVectorCreate(ceed, num_elem * q, &q_data);

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

  CeedInt strides_q_data[3] = {1, q, q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, q * num_elem, strides_q_data, &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 2, p, q, CEED_GAUSS, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1 * 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 2, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 2, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", elem_restriction_q_data, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "rho", elem_restriction_q_data, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

  {
    CeedScalar *u_array;

    CeedVectorGetArrayWrite(u, CEED_MEM_HOST, &u_array);
    for (int i = 0; i < num_nodes_u; i++) {
      u_array[2 * i]     = 1.0;
      u_array[2 * i + 1] = 2.0;
    }
    CeedVectorRestoreArray(u, &u_array);
  }

  CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

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
  CeedVectorDestroy(&q_data);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedDestroy(&ceed);
  return 0;
}
