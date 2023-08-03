/// @file
/// Test creation, action, and destruction for mass matrix operator using a trivial oriented element restriction (see t510)
/// \test Test creation, action, and destruction for mass matrix operator using a trivial oriented element restriction
#include "t510-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t320-basis.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, x, u, v;
  CeedInt             num_elem = 12, dim = 2, p = 6, q = 4;
  CeedInt             nx = 3, ny = 2;
  CeedInt             row, col, offset;
  CeedInt             num_dofs = (nx * 2 + 1) * (ny * 2 + 1), num_qpts = num_elem * q;
  CeedInt             ind_x[num_elem * p];
  bool                orients_u[num_elem * p];
  CeedScalar          q_ref[dim * q], q_weight[q];
  CeedScalar          interp[p * q], grad[dim * p * q];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, dim * num_dofs, &x);
  {
    CeedScalar x_array[dim * num_dofs];

    for (CeedInt i = 0; i < num_dofs; i++) {
      x_array[i]            = (1. / (nx * 2)) * (CeedScalar)(i % (nx * 2 + 1));
      x_array[i + num_dofs] = (1. / (ny * 2)) * (CeedScalar)(i / (nx * 2 + 1));
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_qpts, &q_data);
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorCreate(ceed, num_dofs, &v);

  // Restrictions
  for (CeedInt i = 0; i < num_elem / 2; i++) {
    col    = i % nx;
    row    = i / nx;
    offset = col * 2 + row * (nx * 2 + 1) * 2;

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

    for (CeedInt j = 0; j < 12; j++) {
      orients_u[i * 2 * p + j] = false;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
  CeedElemRestrictionCreateOriented(ceed, num_elem, p, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, orients_u, &elem_restriction_u);

  CeedInt strides_q_data[3] = {1, q, q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, num_qpts, strides_q_data, &elem_restriction_q_data);

  // Bases
  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, dim, p, q, interp, grad, q_ref, q_weight, &basis_x);

  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, 1, p, q, interp, grad, q_ref, q_weight, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

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

  CeedVectorSetValue(u, 0.0);
  CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_dofs; i++) {
      if (fabs(v_array[i]) > 1e-14) printf("[%" CeedInt_FMT "] v %g != 0.0\n", i, v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
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
