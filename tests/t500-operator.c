/// @file
/// Test creation, action, and destruction for mass matrix operator
/// \test Test creation, action, and destruction for mass matrix operator
#include "t500-operator.h"

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
  CeedScalar          x_array[num_nodes_x];

  //! [Ceed Init]
  CeedInit(argv[1], &ceed);
  //! [Ceed Init]
  for (CeedInt i = 0; i < num_nodes_x; i++) x_array[i] = (CeedScalar)i / (num_nodes_x - 1);
  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  //! [ElemRestr Create]
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_nodes_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
  //! [ElemRestr Create]

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p; j++) {
      ind_u[p * i + j] = i * (p - 1) + j;
    }
  }
  //! [ElemRestrU Create]
  CeedElemRestrictionCreate(ceed, num_elem, p, 1, 1, num_nodes_u, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restriction_u);
  CeedInt strides_q_data[3] = {1, q, q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, q * num_elem, strides_q_data, &elem_restriction_q_data);
  //! [ElemRestrU Create]

  //! [Basis Create]
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, q, CEED_GAUSS, &basis_u);
  //! [Basis Create]

  //! [QFunction Create]
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);
  //! [QFunction Create]

  //! [Setup Create]
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  //! [Setup Create]

  //! [Operator Create]
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  //! [Operator Create]

  CeedVectorCreate(ceed, num_nodes_x, &x);
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, x_array);
  CeedVectorCreate(ceed, num_elem * q, &q_data);

  //! [Setup Set]
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", elem_restriction_q_data, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  //! [Setup Set]

  //! [Operator Set]
  CeedOperatorSetField(op_mass, "rho", elem_restriction_q_data, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  //! [Operator Set]

  //! [Setup Apply]
  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);
  //! [Setup Apply]

  CeedVectorCreate(ceed, num_nodes_u, &u);
  CeedVectorSetValue(u, 0.0);
  CeedVectorCreate(ceed, num_nodes_u, &v);
  //! [Operator Apply]
  CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);
  //! [Operator Apply]

  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_nodes_u; i++) {
      if (fabs(v_array[i]) > 1e-14) printf("[%" CeedInt_FMT "] v %g != 0.0\n", i, v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&q_data);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedDestroy(&ceed);
  return 0;
}
