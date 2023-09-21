/// @file
/// Test full assembly of mass matrix operator with oriented and curl-oriented element restrictions (see t560)
/// \test Test full assembly of mass matrix operator with oriented and curl-oriented element restrictions
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t510-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_q_data;
  CeedElemRestriction elem_restriction_u, oriented_elem_restriction_u, curl_oriented_elem_restriction_u;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass, op_mass_oriented, op_mass_curl_oriented;
  CeedVector          q_data, x;
  CeedInt             p = 3, q = 4, dim = 2;
  CeedInt             n_x = 3, n_y = 2;
  CeedInt             num_elem = n_x * n_y;
  CeedInt             num_dofs = (n_x * 2 + 1) * (n_y * 2 + 1), num_qpts = num_elem * q * q;
  CeedInt             ind_x[num_elem * p * p];
  bool                orients_u[num_elem * p * p];
  CeedInt8            curl_orients_u[3 * num_elem * p * p];
  CeedScalar          assembled_values[num_dofs * num_dofs];
  CeedScalar          assembled_values_oriented[num_dofs * num_dofs];
  CeedScalar          assembled_values_curl_oriented[num_dofs * num_dofs];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs, &x);
  {
    CeedScalar x_array[dim * num_dofs];

    for (CeedInt i = 0; i < n_x * 2 + 1; i++) {
      for (CeedInt j = 0; j < n_y * 2 + 1; j++) {
        x_array[i + j * (n_x * 2 + 1) + 0 * num_dofs] = (CeedScalar)i / (2 * n_x);
        x_array[i + j * (n_x * 2 + 1) + 1 * num_dofs] = (CeedScalar)j / (2 * n_y);
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_qpts, &q_data);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;
    col    = i % n_x;
    row    = i / n_x;
    offset = col * (p - 1) + row * (n_x * 2 + 1) * (p - 1);
    for (CeedInt j = 0; j < p; j++) {
      for (CeedInt k = 0; k < p; k++) {
        ind_x[p * (p * i + k) + j]                    = offset + k * (n_x * 2 + 1) + j;
        orients_u[p * (p * i + k) + j]                = false;
        curl_orients_u[3 * (p * (p * i + k) + j) + 0] = 0;
        curl_orients_u[3 * (p * (p * i + k) + j) + 1] = 1;
        curl_orients_u[3 * (p * (p * i + k) + j) + 2] = 0;
      }
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p * p, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
  CeedElemRestrictionCreate(ceed, num_elem, p * p, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_u);
  CeedElemRestrictionCreateOriented(ceed, num_elem, p * p, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, orients_u,
                                    &oriented_elem_restriction_u);
  CeedElemRestrictionCreateCurlOriented(ceed, num_elem, p * p, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, curl_orients_u,
                                        &curl_oriented_elem_restriction_u);

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
  CeedOperatorSetField(op_setup, "rho", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_oriented);
  CeedOperatorSetField(op_mass_oriented, "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass_oriented, "u", oriented_elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_oriented, "v", oriented_elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_curl_oriented);
  CeedOperatorSetField(op_mass_curl_oriented, "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass_curl_oriented, "u", curl_oriented_elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_curl_oriented, "v", curl_oriented_elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

  // Fully assemble operators
  CeedSize   num_entries, num_entries_oriented, num_entries_curl_oriented;
  CeedInt   *rows, *rows_oriented, *rows_curl_oriented;
  CeedInt   *cols, *cols_oriented, *cols_curl_oriented;
  CeedVector assembled, assembled_oriented, assembled_curl_oriented;

  for (CeedInt k = 0; k < num_dofs * num_dofs; ++k) {
    assembled_values[k]               = 0.0;
    assembled_values_oriented[k]      = 0.0;
    assembled_values_curl_oriented[k] = 0.0;
  }
  CeedOperatorLinearAssembleSymbolic(op_mass, &num_entries, &rows, &cols);
  CeedOperatorLinearAssembleSymbolic(op_mass_oriented, &num_entries_oriented, &rows_oriented, &cols_oriented);
  CeedOperatorLinearAssembleSymbolic(op_mass_curl_oriented, &num_entries_curl_oriented, &rows_curl_oriented, &cols_curl_oriented);
  CeedVectorCreate(ceed, num_entries, &assembled);
  CeedVectorCreate(ceed, num_entries_oriented, &assembled_oriented);
  CeedVectorCreate(ceed, num_entries_curl_oriented, &assembled_curl_oriented);
  CeedOperatorLinearAssemble(op_mass, assembled);
  CeedOperatorLinearAssemble(op_mass_oriented, assembled_oriented);
  CeedOperatorLinearAssemble(op_mass_curl_oriented, assembled_curl_oriented);
  {
    const CeedScalar *assembled_array;

    CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
    for (CeedInt k = 0; k < num_entries; ++k) {
      assembled_values[rows[k] * num_dofs + cols[k]] += assembled_array[k];
    }
    CeedVectorRestoreArrayRead(assembled, &assembled_array);
  }
  {
    const CeedScalar *assembled_array;

    CeedVectorGetArrayRead(assembled_oriented, CEED_MEM_HOST, &assembled_array);
    for (CeedInt k = 0; k < num_entries_oriented; ++k) {
      assembled_values_oriented[rows_oriented[k] * num_dofs + cols_oriented[k]] += assembled_array[k];
    }
    CeedVectorRestoreArrayRead(assembled_oriented, &assembled_array);
  }
  {
    const CeedScalar *assembled_array;

    CeedVectorGetArrayRead(assembled_curl_oriented, CEED_MEM_HOST, &assembled_array);
    for (CeedInt k = 0; k < num_entries_curl_oriented; ++k) {
      assembled_values_curl_oriented[rows_curl_oriented[k] * num_dofs + cols_curl_oriented[k]] += assembled_array[k];
    }
    CeedVectorRestoreArrayRead(assembled_curl_oriented, &assembled_array);
  }

  // Check output
  for (CeedInt i = 0; i < num_dofs; i++) {
    for (CeedInt j = 0; j < num_dofs; j++) {
      if (fabs(assembled_values_oriented[j * num_dofs + i] - assembled_values[j * num_dofs + i]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT ", %" CeedInt_FMT "] Error in oriented assembly: %f != %f\n", i, j, assembled_values_oriented[j * num_dofs + i],
               assembled_values[j * num_dofs + i]);
        // LCOV_EXCL_STOP
      }
      if (fabs(assembled_values_curl_oriented[j * num_dofs + i] - assembled_values[j * num_dofs + i]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT ", %" CeedInt_FMT "] Error in curl-oriented assembly: %f != %f\n", i, j,
               assembled_values_curl_oriented[j * num_dofs + i], assembled_values[j * num_dofs + i]);
        // LCOV_EXCL_STOP
      }
    }
  }

  // Cleanup
  free(rows);
  free(cols);
  free(rows_oriented);
  free(cols_oriented);
  free(rows_curl_oriented);
  free(cols_curl_oriented);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&assembled);
  CeedVectorDestroy(&assembled_oriented);
  CeedVectorDestroy(&assembled_curl_oriented);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&oriented_elem_restriction_u);
  CeedElemRestrictionDestroy(&curl_oriented_elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedOperatorDestroy(&op_mass_oriented);
  CeedOperatorDestroy(&op_mass_curl_oriented);
  CeedDestroy(&ceed);
  return 0;
}
