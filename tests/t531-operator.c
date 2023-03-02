/// @file
/// Test assembly of Poisson operator QFunction
/// \test Test assembly of Poisson operator QFunction
#include "t531-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data, elem_restriction_assembled;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_diff, qf_diff_assembled;
  CeedOperator        op_setup, op_diff, op_diff_assembled;
  CeedVector          q_data, x, assembled, u, v;
  CeedInt             num_elem = 6, p = 3, q = 4, dim = 2;
  CeedInt             nx = 3, ny = 2;
  CeedInt             num_dofs = (nx * 2 + 1) * (ny * 2 + 1), num_qpts = num_elem * q * q;
  CeedInt             ind_x[num_elem * p * p];

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
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorCreate(ceed, num_dofs, &v);
  CeedVectorCreate(ceed, num_qpts * dim * (dim + 1) / 2, &q_data);

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
  CeedElemRestrictionCreate(ceed, num_elem, p * p, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_u);

  CeedInt strides_q_data[3] = {1, q * q, q * q * dim * (dim + 1) / 2};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, dim * (dim + 1) / 2, dim * (dim + 1) / 2 * num_qpts, strides_q_data,
                                   &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis_u);

  // QFunction - setup
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "q data", dim * (dim + 1) / 2, CEED_EVAL_NONE);

  // Operator - setup
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "q data", elem_restriction_q_data, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, diff, diff_loc, &qf_diff);
  CeedQFunctionAddInput(qf_diff, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_diff, "q data", dim * (dim + 1) / 2, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_diff, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_diff);
  CeedOperatorSetField(op_diff, "du", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff, "q data", elem_restriction_q_data, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_diff, "dv", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply original Poisson Operator
  CeedVectorSetValue(u, 1.0);
  CeedOperatorApply(op_diff, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_dofs; i++) {
      if (fabs(v_array[i]) > 100. * CEED_EPSILON) printf("Error: Operator computed v[%" CeedInt_FMT "] = %f != 0.0\n", i, v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Assemble QFunction
  CeedOperatorSetQFunctionAssemblyReuse(op_diff, true);
  CeedOperatorLinearAssembleQFunction(op_diff, &assembled, &elem_restriction_assembled, CEED_REQUEST_IMMEDIATE);
  // Second call will be no-op since SetQFunctionUpdated was not called
  CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(op_diff, false);
  CeedOperatorLinearAssembleQFunction(op_diff, &assembled, &elem_restriction_assembled, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply assembled
  CeedQFunctionCreateInterior(ceed, 1, diff_lin, diff_lin_loc, &qf_diff_assembled);
  CeedQFunctionAddInput(qf_diff_assembled, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_diff_assembled, "q data", dim * dim, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_diff_assembled, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply assembled
  CeedOperatorCreate(ceed, qf_diff_assembled, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_diff_assembled);
  CeedOperatorSetField(op_diff_assembled, "du", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff_assembled, "q data", elem_restriction_assembled, CEED_BASIS_COLLOCATED, assembled);
  CeedOperatorSetField(op_diff_assembled, "dv", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply new Poisson Operator
  CeedVectorSetValue(v, 0.0);
  CeedOperatorApply(op_diff_assembled, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_dofs; i++) {
      if (fabs(v_array[i]) > 100. * CEED_EPSILON) printf("Error: Linearized operator computed v[i] = %f != 0.0\n", v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
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
  CeedElemRestrictionDestroy(&elem_restriction_assembled);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_diff);
  CeedQFunctionDestroy(&qf_diff_assembled);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_diff);
  CeedOperatorDestroy(&op_diff_assembled);
  CeedDestroy(&ceed);
  return 0;
}
