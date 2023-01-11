/// @file
/// Test assembly of mass matrix operator QFunction
/// \test Test assembly of mass matrix operator QFunction
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t510-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_qd_i, elem_restr_lin_i;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, X, A, u, v;
  const CeedScalar   *a, *q;
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
  CeedVectorCreate(ceed, num_qpts, &q_data);

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
  CeedInt strides_qd[3] = {1, Q * Q, Q * Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * Q, 1, num_qpts, strides_qd, &elem_restr_qd_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &basis_u);

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
  CeedOperatorSetField(op_setup, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "rho", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, X, q_data, CEED_REQUEST_IMMEDIATE);

  // Assemble QFunction
  CeedOperatorSetQFunctionAssemblyReuse(op_mass, true);
  CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(op_mass, true);
  CeedOperatorLinearAssembleQFunction(op_mass, &A, &elem_restr_lin_i, CEED_REQUEST_IMMEDIATE);
  // Second call will be no-op since SetQFunctionUpdated was not called
  CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(op_mass, false);
  CeedOperatorLinearAssembleQFunction(op_mass, &A, &elem_restr_lin_i, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(A, CEED_MEM_HOST, &a);
  CeedVectorGetArrayRead(q_data, CEED_MEM_HOST, &q);
  for (CeedInt i = 0; i < num_qpts; i++)
    if (fabs(q[i] - a[i]) > 1e-9) printf("Error: A[%" CeedInt_FMT "] = %f != %f\n", i, a[i], q[i]);
  CeedVectorRestoreArrayRead(A, &a);
  CeedVectorRestoreArrayRead(q_data, &q);

  // Apply original Mass Operator
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorSetValue(u, 1.0);
  CeedVectorCreate(ceed, num_dofs, &v);
  CeedVectorSetValue(v, 0.0);
  CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedScalar        area = 0.0;
  const CeedScalar *vv;
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &vv);
  for (CeedInt i = 0; i < num_dofs; i++) area += vv[i];
  CeedVectorRestoreArrayRead(v, &vv);
  if (fabs(area - 1.0) > 100. * CEED_EPSILON) printf("Error: True operator computed area = %f != 1.0\n", area);

  // Switch to new q_data
  CeedVectorGetArrayRead(A, CEED_MEM_HOST, &a);
  CeedVectorSetArray(q_data, CEED_MEM_HOST, CEED_COPY_VALUES, (CeedScalar *)a);
  CeedVectorRestoreArrayRead(A, &a);

  // Apply new Mass Operator
  CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  area = 0.0;
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &vv);
  for (CeedInt i = 0; i < num_dofs; i++) area += vv[i];
  CeedVectorRestoreArrayRead(v, &vv);
  if (fabs(area - 1.0) > 1000. * CEED_EPSILON) printf("Error: Linearized operator computed area = %f != 1.0\n", area);

  // Cleanup
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
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
