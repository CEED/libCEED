/// @file
/// Test assembly of mass and Poisson operator QFunction
/// \test Test assembly of mass and Poisson operator QFunction
#include "t532-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data_mass, elem_restriction_q_data_diff, elem_restriction_assembled;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup_mass, qf_setup_diff, qf_apply, qf_apply_assembled;
  CeedOperator        op_setup_mass, op_setup_diff, op_apply, op_apply_assembled;
  CeedVector          q_data_mass, q_data_diff, x, assembled, u, v;
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
  CeedVectorCreate(ceed, num_qpts, &q_data_mass);
  CeedVectorCreate(ceed, num_qpts * dim * (dim + 1) / 2, &q_data_diff);

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

  CeedInt strides_q_data_mass[3] = {1, q * q, q * q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, 1, num_qpts, strides_q_data_mass, &elem_restriction_q_data_mass);

  CeedInt strides_q_data_diff[3] = {1, q * q, q * q * dim * (dim + 1) / 2};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, dim * (dim + 1) / 2, dim * (dim + 1) / 2 * num_qpts, strides_q_data_diff,
                                   &elem_restriction_q_data_diff);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p, q, CEED_GAUSS, &basis_x);
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
  CeedOperatorSetField(op_setup_mass, "q data", elem_restriction_q_data_mass, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // QFunction - setup diff
  CeedQFunctionCreateInterior(ceed, 1, setup_diff, setup_diff_loc, &qf_setup_diff);
  CeedQFunctionAddInput(qf_setup_diff, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_diff, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_diff, "q data", dim * (dim + 1) / 2, CEED_EVAL_NONE);

  // Operator - setup diff
  CeedOperatorCreate(ceed, qf_setup_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_diff);
  CeedOperatorSetField(op_setup_diff, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_diff, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_diff, "q data", elem_restriction_q_data_diff, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operators
  CeedOperatorApply(op_setup_mass, x, q_data_mass, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_diff, x, q_data_diff, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, apply, apply_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_apply, "mass q data", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply, "diff q data", dim * (dim + 1) / 2, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);
  CeedOperatorSetField(op_apply, "du", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "mass q data", elem_restriction_q_data_mass, CEED_BASIS_COLLOCATED, q_data_mass);
  CeedOperatorSetField(op_apply, "diff q data", elem_restriction_q_data_diff, CEED_BASIS_COLLOCATED, q_data_diff);
  CeedOperatorSetField(op_apply, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "dv", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply original operator
  CeedVectorSetValue(u, 1.0);
  CeedOperatorApply(op_apply, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        area = 0.0;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_dofs; i++) area += v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
    if (fabs(area - 1.0) > 100. * CEED_EPSILON) printf("Error: True operator computed area = %f != 1.0\n", area);
  }

  // Assemble QFunction
  CeedOperatorSetQFunctionAssemblyReuse(op_apply, true);
  CeedOperatorLinearAssembleQFunction(op_apply, &assembled, &elem_restriction_assembled, CEED_REQUEST_IMMEDIATE);
  // Second call will be no-op since SetQFunctionUpdated was not called
  CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(op_apply, false);
  CeedOperatorLinearAssembleQFunction(op_apply, &assembled, &elem_restriction_assembled, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply assembled
  CeedQFunctionCreateInterior(ceed, 1, apply_lin, apply_lin_loc, &qf_apply_assembled);
  CeedQFunctionAddInput(qf_apply_assembled, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_apply_assembled, "q data", (dim + 1) * (dim + 1), CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply_assembled, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply_assembled, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply_assembled, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply assembled
  CeedOperatorCreate(ceed, qf_apply_assembled, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply_assembled);
  CeedOperatorSetField(op_apply_assembled, "du", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply_assembled, "q data", elem_restriction_assembled, CEED_BASIS_COLLOCATED, assembled);
  CeedOperatorSetField(op_apply_assembled, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply_assembled, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply_assembled, "dv", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply assembled QFunction operator
  CeedVectorSetValue(v, 0.0);
  CeedOperatorApply(op_apply_assembled, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        area = 0.0;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_dofs; i++) area += v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
    if (fabs(area - 1.0) > 100. * CEED_EPSILON) printf("Error: Assembled operator computed area = %f != 1.0\n", area);
  }

  // Cleanup
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&assembled);
  CeedVectorDestroy(&q_data_mass);
  CeedVectorDestroy(&q_data_diff);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data_mass);
  CeedElemRestrictionDestroy(&elem_restriction_q_data_diff);
  CeedElemRestrictionDestroy(&elem_restriction_assembled);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_setup_diff);
  CeedQFunctionDestroy(&qf_apply);
  CeedQFunctionDestroy(&qf_apply_assembled);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_setup_diff);
  CeedOperatorDestroy(&op_apply);
  CeedOperatorDestroy(&op_apply_assembled);
  CeedDestroy(&ceed);
  return 0;
}
