/// @file
/// Test assembly of mass matrix operator point block diagonal
/// \test Test assembly of mass matrix operator point block diagonal
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t537-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, x, assembled, u, v;
  CeedInt             num_elem = 6, p = 3, q = 4, dim = 2, num_comp = 2;
  CeedInt             nx = 3, ny = 2;
  CeedInt             num_dofs = (nx * 2 + 1) * (ny * 2 + 1), num_qpts = num_elem * q * q;
  CeedInt             ind_x[num_elem * p * p];
  CeedScalar          assembled_true[num_comp * num_comp * num_dofs];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs, &x);
  {
    CeedScalar x_array[dim * num_dofs];

    for (CeedInt i = 0; i < nx * 2 + 1; i++) {
      for (CeedInt j = 0; j < ny * 2 + 1; j++) {
        x_array[i + j * (nx * 2 + 1) + 0 * num_dofs] = (CeedScalar)i / (2 * nx);
        x_array[i + j * (nx * 2 + 1) + 1 * num_dofs] = (CeedScalar)j / (2 * ny);
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_qpts, &q_data);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;
    col    = i % nx;
    row    = i / nx;
    offset = col * (p - 1) + row * (nx * 2 + 1) * (p - 1);
    for (CeedInt j = 0; j < p; j++) {
      for (CeedInt k = 0; k < p; k++) ind_x[p * (p * i + k) + j] = offset + k * (nx * 2 + 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p * p, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
  CeedElemRestrictionCreate(ceed, num_elem, p * p, num_comp, num_dofs, num_comp * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x,
                            &elem_restriction_u);

  CeedInt strides_q_data[3] = {1, q * q, q * q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, 1, num_qpts, strides_q_data, &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, p, q, CEED_GAUSS, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", num_comp, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", num_comp, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", elem_restriction_q_data, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "rho", elem_restriction_q_data, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

  // Assemble diagonal
  CeedVectorCreate(ceed, num_comp * num_comp * num_dofs, &assembled);
  CeedOperatorLinearAssemblePointBlockDiagonal(op_mass, assembled, CEED_REQUEST_IMMEDIATE);

  // Manually assemble diagonal
  CeedVectorCreate(ceed, num_comp * num_dofs, &u);
  CeedVectorSetValue(u, 0.0);
  CeedVectorCreate(ceed, num_comp * num_dofs, &v);
  for (int i = 0; i < num_comp * num_comp * num_dofs; i++) assembled_true[i] = 0.0;
  CeedInt ind_old = -1;
  for (int i = 0; i < num_dofs; i++) {
    for (int j = 0; j < num_comp; j++) {
      CeedScalar       *u_array;
      const CeedScalar *v_array;

      // Set input
      CeedVectorGetArray(u, CEED_MEM_HOST, &u_array);
      CeedInt ind  = i + j * num_dofs;
      u_array[ind] = 1.0;
      if (ind > 0) u_array[ind_old] = 0.0;
      ind_old = ind;
      CeedVectorRestoreArray(u, &u_array);

      // Compute effect of DoF i, comp j
      CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

      // Retrieve entry
      CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
      for (int k = 0; k < num_comp; k++) assembled_true[i * num_comp * num_comp + k * num_comp + j] += v_array[i + k * num_dofs];
      CeedVectorRestoreArrayRead(v, &v_array);
    }
  }

  // Check output
  {
    const CeedScalar *assembled_array;

    CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
    for (int i = 0; i < num_comp * num_comp * num_dofs; i++) {
      if (fabs(assembled_array[i] - assembled_true[i]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT "] Error in assembly: %f != %f\n", i, assembled_array[i], assembled_true[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(assembled, &assembled_array);
  }

  // Cleanup
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&assembled);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
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
