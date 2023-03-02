/// @file
/// Test assembly of mass matrix operator QFunction
/// \test Test assembly of mass matrix operator QFunction
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t510-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data, elem_restriction_assembled;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, x, qf_assembled, u, v;
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
  CeedElemRestrictionCreate(ceed, num_elem, p * p, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_u);

  CeedInt strides_q_data[3] = {1, q * q, q * q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, 1, num_qpts, strides_q_data, &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis_u);

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

  // Apply Setup Operator
  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

  // Assemble QFunction
  CeedOperatorSetQFunctionAssemblyReuse(op_mass, true);
  CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(op_mass, true);
  CeedOperatorLinearAssembleQFunction(op_mass, &qf_assembled, &elem_restriction_assembled, CEED_REQUEST_IMMEDIATE);
  // Second call will be no-op since SetQFunctionUpdated was not called
  CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(op_mass, false);
  CeedOperatorLinearAssembleQFunction(op_mass, &qf_assembled, &elem_restriction_assembled, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *assembled_array, *q_data_array;

    CeedVectorGetArrayRead(qf_assembled, CEED_MEM_HOST, &assembled_array);
    CeedVectorGetArrayRead(q_data, CEED_MEM_HOST, &q_data_array);
    for (CeedInt i = 0; i < num_qpts; i++)
      if (fabs(q_data_array[i] - assembled_array[i]) > 1e-9) {
        // LCOV_EXCL_START
        printf("Error: qf_assembled[%" CeedInt_FMT "] = %f != %f\n", i, assembled_array[i], q_data_array[i]);
        // LCOV_EXCL_STOP
      }
    CeedVectorRestoreArrayRead(qf_assembled, &assembled_array);
    CeedVectorRestoreArrayRead(q_data, &q_data_array);
  }

  // Apply original Mass Operator
  CeedVectorSetValue(u, 1.0);
  CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        area = 0.0;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_dofs; i++) area += v_array[i];
    if (fabs(area - 1.0) > 100. * CEED_EPSILON) printf("Error: True operator computed area = %f != 1.0\n", area);
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Switch to new q_data
  {
    const CeedScalar *assembled_array;

    CeedVectorGetArrayRead(qf_assembled, CEED_MEM_HOST, &assembled_array);
    CeedVectorSetArray(q_data, CEED_MEM_HOST, CEED_COPY_VALUES, (CeedScalar *)assembled_array);
    CeedVectorRestoreArrayRead(qf_assembled, &assembled_array);
  }

  // Apply new Mass Operator
  CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        area = 0.0;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_dofs; i++) area += v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
    if (fabs(area - 1.0) > 1000. * CEED_EPSILON) printf("Error: Linearized operator computed area = %f != 1.0\n", area);
  }

  // Cleanup
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&qf_assembled);
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
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedDestroy(&ceed);
  return 0;
}
