/// @file
/// Test creation, action, and destruction for mass matrix operator
/// \test Test creation, action, and destruction for mass matrix operator
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t320-basis.h"
#include "t510-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_qd_i;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, X, U, V;
  const CeedScalar   *hv;
  CeedInt             num_elem = 12, dim = 2, P = 6, Q = 4;
  CeedInt             nx = 3, ny = 2;
  CeedInt             row, col, offset;
  CeedInt             num_dofs = (nx * 2 + 1) * (ny * 2 + 1), num_qpts = num_elem * Q;
  CeedInt             indx[num_elem * P];
  CeedScalar          x[dim * num_dofs];
  CeedScalar          q_ref[dim * Q], q_weight[Q];
  CeedScalar          interp[P * Q], grad[dim * P * Q];
  CeedScalar          sum;

  CeedInit(argv[1], &ceed);

  for (CeedInt i = 0; i < num_dofs; i++) {
    x[i]            = (1. / (nx * 2)) * (CeedScalar)(i % (nx * 2 + 1));
    x[i + num_dofs] = (1. / (ny * 2)) * (CeedScalar)(i / (nx * 2 + 1));
  }
  for (CeedInt i = 0; i < num_elem / 2; i++) {
    col    = i % nx;
    row    = i / nx;
    offset = col * 2 + row * (nx * 2 + 1) * 2;

    indx[i * 2 * P + 0] = 2 + offset;
    indx[i * 2 * P + 1] = 9 + offset;
    indx[i * 2 * P + 2] = 16 + offset;
    indx[i * 2 * P + 3] = 1 + offset;
    indx[i * 2 * P + 4] = 8 + offset;
    indx[i * 2 * P + 5] = 0 + offset;

    indx[i * 2 * P + 6]  = 14 + offset;
    indx[i * 2 * P + 7]  = 7 + offset;
    indx[i * 2 * P + 8]  = 0 + offset;
    indx[i * 2 * P + 9]  = 15 + offset;
    indx[i * 2 * P + 10] = 8 + offset;
    indx[i * 2 * P + 11] = 16 + offset;
  }

  // Restrictions
  CeedElemRestrictionCreate(ceed, num_elem, P, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, indx, &elem_restr_x);

  CeedElemRestrictionCreate(ceed, num_elem, P, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, indx, &elem_restr_u);
  CeedInt strides_qd[3] = {1, Q, Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q, 1, num_qpts, strides_qd, &elem_restr_qd_i);

  // Bases
  buildmats(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, dim, P, Q, interp, grad, q_ref, q_weight, &basis_x);

  buildmats(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, 1, P, Q, interp, grad, q_ref, q_weight, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "x", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);

  CeedVectorCreate(ceed, dim * num_dofs, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorCreate(ceed, num_qpts, &q_data);

  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "x", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass, "rho", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_setup, X, q_data, CEED_REQUEST_IMMEDIATE);

  CeedVectorCreate(ceed, num_dofs, &U);
  CeedVectorSetValue(U, 1.0);
  CeedVectorCreate(ceed, num_dofs, &V);

  CeedOperatorApply(op_mass, U, V, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i = 0; i < num_dofs; i++) sum += hv[i];
  if (fabs(sum - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum);
  CeedVectorRestoreArrayRead(V, &hv);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedElemRestrictionDestroy(&elem_restr_u);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_qd_i);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&q_data);
  CeedDestroy(&ceed);
  return 0;
}
