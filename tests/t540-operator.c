/// @file
/// Test creation and use of FDM element inverse
/// \test Test creation and use of FDM element inverse
#include "t540-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup_mass, qf_apply;
  CeedOperator        op_setup_mass, op_apply, op_inverse;
  CeedVector          q_data_mass, x, u, v;
  CeedInt             num_elem = 1, p = 4, q = 5, dim = 2;
  CeedInt             num_dofs = p * p, num_qpts = num_elem * q * q;

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_elem * (2 * 2), &x);
  {
    CeedScalar x_array[dim * num_elem * (2 * 2)];

    for (CeedInt i = 0; i < 2; i++) {
      for (CeedInt j = 0; j < 2; j++) {
        x_array[i + j * 2 + 0 * 4] = i;
        x_array[i + j * 2 + 1 * 4] = j;
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorCreate(ceed, num_dofs, &v);
  CeedVectorCreate(ceed, num_qpts, &q_data_mass);

  // Restrictions
  CeedInt strides_x[3] = {1, 2 * 2, 2 * 2 * dim};
  CeedElemRestrictionCreateStrided(ceed, num_elem, 2 * 2, dim, dim * num_elem * 2 * 2, strides_x, &elem_restriction_x);

  CeedInt strides_u[3] = {1, p * p, p * p};
  CeedElemRestrictionCreateStrided(ceed, num_elem, p * p, 1, num_dofs, strides_u, &elem_restriction_u);

  CeedInt strides_q_data[3] = {1, q * q, q * q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, 1, num_qpts, strides_q_data, &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis_u);

  // QFunction - setup mass
  CeedQFunctionCreateInterior(ceed, 1, setup_mass, setup_mass_loc, &qf_setup_mass);
  CeedQFunctionAddInput(qf_setup_mass, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_mass, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_mass, "q data", 1, CEED_EVAL_NONE);

  // Operator - setup mass
  CeedOperatorCreate(ceed, qf_setup_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_mass);
  CeedOperatorSetField(op_setup_mass, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_mass, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_mass, "q data", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup_mass, x, q_data_mass, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, apply, apply_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_apply, "mass q data", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", 1, CEED_EVAL_INTERP);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);
  CeedOperatorSetField(op_apply, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "mass q data", elem_restriction_q_data, CEED_BASIS_NONE, q_data_mass);
  CeedOperatorSetField(op_apply, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply original operator
  CeedVectorSetValue(u, 1.0);
  CeedOperatorApply(op_apply, u, v, CEED_REQUEST_IMMEDIATE);

  // Create FDM element inverse
  CeedOperatorCreateFDMElementInverse(op_apply, &op_inverse, CEED_REQUEST_IMMEDIATE);

  // Apply FDM element inverse
  CeedOperatorApply(op_inverse, v, u, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *u_array;

    CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
    for (int i = 0; i < num_dofs; i++) {
      if (fabs(u_array[i] - 1.0) > 500. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT "] Error in inverse: %e - 1.0 = %e\n", i, u_array[i], u_array[i] - 1.);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(u, &u_array);
  }

  // Cleanup
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&q_data_mass);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_apply);
  CeedOperatorDestroy(&op_inverse);
  CeedDestroy(&ceed);
  return 0;
}
