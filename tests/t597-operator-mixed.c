/// @file
/// Test full assembly of Poisson operator AtPoints with mixed precision
/// \test Test full assembly of Poisson operator AtPoints with mixed precision
#include <ceed.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t597-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  for (CeedInt num_comp = 1; num_comp <= 3; num_comp++) {
    CeedElemRestriction elem_restriction_x, elem_restriction_x_points, elem_restriction_u, elem_restriction_q_data;
    CeedBasis           basis_x, basis_u;
    CeedQFunction       qf_setup, qf_diff;
    CeedOperator        op_setup, op_diff;
    CeedVector          q_data, x, x_points, u, v;
    CeedInt             p = 3, q = 4, dim = 2;
    CeedInt             n_x = 3, n_y = 2;
    CeedInt             num_elem = n_x * n_y;
    CeedInt             num_dofs = (n_x * 2 + 1) * (n_y * 2 + 1), num_points_per_elem = 4, num_points = num_elem * num_points_per_elem;
    CeedInt             ind_x[num_elem * p * p];
    CeedScalar          assembled_values[num_comp * num_comp * num_dofs * num_dofs];
    CeedScalar          assembled_true[num_comp * num_comp * num_dofs * num_dofs];

    // Points
    CeedVectorCreate(ceed, dim * num_points, &x_points);
    {
      CeedScalar x_array[dim * num_points];

      for (CeedInt e = 0; e < num_elem; e++) {
        for (CeedInt d = 0; d < dim; d++) {
          x_array[num_points_per_elem * (e * dim + d) + 0] = 0.25;
          x_array[num_points_per_elem * (e * dim + d) + 1] = d == 0 ? -0.25 : 0.25;
          x_array[num_points_per_elem * (e * dim + d) + 2] = d == 0 ? 0.25 : -0.25;
          x_array[num_points_per_elem * (e * dim + d) + 3] = 0.25;
        }
      }
      CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    }
    {
      CeedInt ind_x[num_elem + 1 + num_points];

      for (CeedInt i = 0; i <= num_elem; i++) ind_x[i] = num_elem + 1 + i * num_points_per_elem;
      for (CeedInt i = 0; i < num_points; i++) ind_x[num_elem + 1 + i] = i;
      CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, dim, num_points * dim, CEED_MEM_HOST, CEED_COPY_VALUES, ind_x,
                                        &elem_restriction_x_points);
      CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, dim * (dim + 1) / 2, num_points * dim * (dim + 1) / 2, CEED_MEM_HOST,
                                        CEED_COPY_VALUES, ind_x, &elem_restriction_q_data);
    }

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
    CeedVectorCreate(ceed, num_comp * num_dofs, &u);
    CeedVectorCreate(ceed, num_comp * num_dofs, &v);
    CeedVectorCreate(ceed, num_points * dim * (dim + 1) / 2, &q_data);

    // Restrictions
    for (CeedInt i = 0; i < num_elem; i++) {
      CeedInt col, row, offset;
      col    = i % n_x;
      row    = i / n_x;
      offset = col * (p - 1) + row * (n_x * 2 + 1) * (p - 1);
      for (CeedInt j = 0; j < p; j++) {
        for (CeedInt k = 0; k < p; k++) ind_x[p * (p * i + k) + j] = offset + k * (n_x * 2 + 1) + j;
      }
    }
    CeedElemRestrictionCreate(ceed, num_elem, p * p, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
    CeedElemRestrictionCreate(ceed, num_elem, p * p, num_comp, num_dofs, num_comp * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x,
                              &elem_restriction_u);

    // Bases
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p, q, CEED_GAUSS, &basis_x);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, p, q, CEED_GAUSS, &basis_u);

    // QFunction - setup
    CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
    CeedQFunctionAddInput(qf_setup, "dx", dim * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddOutput(qf_setup, "q data", dim * (dim + 1) / 2, CEED_EVAL_NONE);

    // Operator - setup
    CeedOperatorCreateAtPoints(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
    CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_setup, "q data", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
    CeedOperatorSetMixedPrecision(op_setup);
    CeedOperatorAtPointsSetPoints(op_setup, elem_restriction_x_points, x_points);

    // Apply Setup Operator
    CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

    // QFunction - apply
    CeedQFunctionCreateInterior(ceed, 1, diff, diff_loc, &qf_diff);
    CeedQFunctionAddInput(qf_diff, "du", num_comp * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_diff, "q data", dim * (dim + 1) / 2, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_diff, "dv", num_comp * dim, CEED_EVAL_GRAD);
    {
      CeedQFunctionContext qf_context;

      CeedQFunctionContextCreate(ceed, &qf_context);
      CeedQFunctionContextSetData(qf_context, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(CeedInt), &num_comp);
      CeedQFunctionSetContext(qf_diff, qf_context);
      CeedQFunctionContextDestroy(&qf_context);
    }

    // Operator - apply
    CeedOperatorCreateAtPoints(ceed, qf_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_diff);
    CeedOperatorSetField(op_diff, "du", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_diff, "q data", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
    CeedOperatorSetField(op_diff, "dv", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorSetMixedPrecision(op_diff);
    CeedOperatorAtPointsSetPoints(op_diff, elem_restriction_x_points, x_points);

    // Fully assemble operator
    CeedSize   num_entries;
    CeedInt   *rows;
    CeedInt   *cols;
    CeedVector assembled;

    for (CeedInt k = 0; k < num_comp * num_comp * num_dofs * num_dofs; ++k) {
      assembled_values[k] = 0.0;
      assembled_true[k]   = 0.0;
    }
    CeedOperatorLinearAssembleSymbolic(op_diff, &num_entries, &rows, &cols);
    CeedVectorCreate(ceed, num_entries, &assembled);
    CeedOperatorLinearAssemble(op_diff, assembled);
    {
      const CeedScalar *assembled_array;

      CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
      for (CeedInt k = 0; k < num_entries; k++) assembled_values[rows[k] * num_comp * num_dofs + cols[k]] += assembled_array[k];
      CeedVectorRestoreArrayRead(assembled, &assembled_array);
    }

    // Manually assemble operator
    CeedVectorSetValue(u, 0.0);
    for (CeedInt j = 0; j < num_comp * num_dofs; j++) {
      CeedScalar       *u_array;
      const CeedScalar *v_array;

      // Set input
      CeedVectorGetArray(u, CEED_MEM_HOST, &u_array);
      u_array[j] = 1.0;
      if (j) u_array[j - 1] = 0.0;
      CeedVectorRestoreArray(u, &u_array);

      // Compute entries for column j
      CeedOperatorApply(op_diff, u, v, CEED_REQUEST_IMMEDIATE);

      CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
      for (CeedInt i = 0; i < num_comp * num_dofs; i++) assembled_true[i * num_comp * num_dofs + j] = v_array[i];
      CeedVectorRestoreArrayRead(v, &v_array);
    }

    // Check output
    for (CeedInt i = 0; i < num_comp * num_dofs; i++) {
      for (CeedInt j = 0; j < num_comp * num_dofs; j++) {
        if (fabs(assembled_values[i * num_comp * num_dofs + j] - assembled_true[i * num_comp * num_dofs + j]) > FLT_EPSILON) {
          // LCOV_EXCL_START
          printf("[%" CeedInt_FMT ", %" CeedInt_FMT "] Error in assembly: %f != %f\n", i, j, assembled_values[i * num_comp * num_dofs + j],
                 assembled_true[i * num_comp * num_dofs + j]);
          // LCOV_EXCL_STOP
        }
      }
    }

    // Cleanup
    free(rows);
    free(cols);
    CeedVectorDestroy(&x);
    CeedVectorDestroy(&x_points);
    CeedVectorDestroy(&q_data);
    CeedVectorDestroy(&u);
    CeedVectorDestroy(&v);
    CeedVectorDestroy(&assembled);
    CeedElemRestrictionDestroy(&elem_restriction_u);
    CeedElemRestrictionDestroy(&elem_restriction_x);
    CeedElemRestrictionDestroy(&elem_restriction_x_points);
    CeedElemRestrictionDestroy(&elem_restriction_q_data);
    CeedBasisDestroy(&basis_u);
    CeedBasisDestroy(&basis_x);
    CeedQFunctionDestroy(&qf_setup);
    CeedQFunctionDestroy(&qf_diff);
    CeedOperatorDestroy(&op_setup);
    CeedOperatorDestroy(&op_diff);
  }
  CeedDestroy(&ceed);
  return 0;
}
