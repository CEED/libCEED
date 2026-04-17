/// @file
/// Test full assembly of multi-basis asymmetric mass-like operator
/// \test Test full assembly of multi-basis asymmetric mass-like operator
#include "t571-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_p, elem_restriction_q_data;
  CeedBasis           basis_x, basis_u, basis_p;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, x, u, v;
  CeedInt             p_u = 3, p_p = 2, q = 3, dim = 2, num_comp_u = 3, num_comp_p = 2;
  CeedInt             n_x = 3, n_y = 2;
  CeedInt             num_elem   = n_x * n_y;
  CeedInt             num_dofs_u = (n_x * (p_u - 1) + 1) * (n_y * (p_u - 1) + 1), num_dofs_p = (n_x * (p_p - 1) + 1) * (n_y * (p_p - 1) + 1),
          num_qpts = num_elem * q * q;
  CeedInt    ind_x[num_elem * p_u * p_u], ind_p[num_elem * p_p * p_p];
  CeedInt    l_size = num_comp_u * num_dofs_u + num_comp_p * num_dofs_p;
  CeedScalar assembled_values[l_size * l_size];
  CeedScalar assembled_true[l_size * l_size];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs_u, &x);
  {
    CeedScalar x_array[dim * num_dofs_u];

    for (CeedInt i = 0; i < n_x * (p_u - 1) + 1; i++) {
      for (CeedInt j = 0; j < n_y * (p_u - 1) + 1; j++) {
        x_array[i + j * (n_x * 2 + 1) + 0 * num_dofs_u] = (CeedScalar)i / (n_x * (p_u - 1));
        x_array[i + j * (n_x * 2 + 1) + 1 * num_dofs_u] = (CeedScalar)j / (n_y * (p_u - 1));
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, l_size, &u);
  CeedVectorCreate(ceed, l_size, &v);
  CeedVectorCreate(ceed, num_qpts, &q_data);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;

    col    = i % n_x;
    row    = i / n_x;
    offset = col * (p_u - 1) + row * (n_x * (p_u - 1) + 1) * (p_u - 1);
    for (CeedInt j = 0; j < p_u; j++) {
      for (CeedInt k = 0; k < p_u; k++) ind_x[p_u * (p_u * i + k) + j] = offset + k * p_u + j;
    }
  }
  const CeedInt offset_p = num_comp_u * num_dofs_u;

  for (CeedInt k = 0; k < num_elem; k++) {
    CeedInt col, row, offset;

    col    = k % n_x;
    row    = k / n_x;
    offset = col * (p_p - 1) + row * (n_x * (p_p - 1) + 1) * (p_p - 1);
    // Data for node i, component j, element k can be found in the L-vector at index offsets[i + k*elem_size] + j*comp_stride.
    for (CeedInt j = 0; j < p_p; j++) {
      for (CeedInt i = 0; i < p_p; i++) ind_p[k * p_p * p_p + (p_p * i + j)] = offset + i * p_p + j + offset_p;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p_u * p_u, dim, num_dofs_u, dim * num_dofs_u, CEED_MEM_HOST, CEED_USE_POINTER, ind_x,
                            &elem_restriction_x);
  CeedElemRestrictionCreate(ceed, num_elem, p_u * p_u, num_comp_u, num_dofs_u, l_size, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_u);
  CeedElemRestrictionCreate(ceed, num_elem, p_p * p_p, num_comp_p, num_dofs_p, l_size, CEED_MEM_HOST, CEED_USE_POINTER, ind_p, &elem_restriction_p);

  CeedInt strides_q_data[3] = {1, q * q * num_elem, q * q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, 1, num_qpts * 1, strides_q_data, &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p_u, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_u, p_u, q, CEED_GAUSS, &basis_u);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_p, p_p, q, CEED_GAUSS, &basis_p);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "qdata", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, multi_basis, multi_basis_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "qdata", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "du", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "dp", num_comp_p, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "dv", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "dq", num_comp_p, CEED_EVAL_INTERP);

  Context_t            context = {num_comp_u, num_comp_p};
  CeedQFunctionContext ctx;

  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(Context_t), &context);
  CeedQFunctionSetContext(qf_mass, ctx);
  CeedQFunctionContextDestroy(&ctx);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "qdata", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "qdata", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass, "du", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "dp", elem_restriction_p, basis_p, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "dv", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "dq", elem_restriction_p, basis_p, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

  // Fuly assemble operator
  CeedSize   num_entries;
  CeedInt   *rows;
  CeedInt   *cols;
  CeedVector assembled;

  for (CeedInt k = 0; k < l_size * l_size; k++) {
    assembled_values[k] = 0.0;
    assembled_true[k]   = 0.0;
  }
  CeedOperatorLinearAssembleSymbolic(op_mass, &num_entries, &rows, &cols);
  CeedVectorCreate(ceed, num_entries, &assembled);
  CeedOperatorLinearAssemble(op_mass, assembled);
  {
    const CeedScalar *assembled_array;

    CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
    for (CeedInt k = 0; k < num_entries; k++) assembled_values[rows[k] * l_size + cols[k]] += assembled_array[k];
    CeedVectorRestoreArrayRead(assembled, &assembled_array);
  }

  // Manually assemble operator
  CeedInt old_index = -1;

  CeedVectorSetValue(u, 0.0);
  for (CeedInt j = 0; j < l_size; j++) {
    CeedScalar       *u_array;
    const CeedScalar *v_array;

    // Set input
    CeedVectorGetArray(u, CEED_MEM_HOST, &u_array);
    u_array[j] = 1.0;
    if (j > 0) u_array[old_index] = 0.0;
    old_index = j;
    CeedVectorRestoreArray(u, &u_array);

    // Compute effect of DoF ind
    CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < l_size; i++) assembled_true[i * l_size + j] = v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Check output
  for (CeedInt i = 0; i < l_size; i++) {
    for (CeedInt j = 0; j < l_size; j++) {
      const CeedScalar assembled_value      = assembled_values[i * l_size + j];
      const CeedScalar assembled_true_value = assembled_true[i * l_size + j];
      const CeedScalar error                = fabs(assembled_value - assembled_true_value) / (fmax(fabs(assembled_true_value), 1.0));

      if (!(error < 100. * CEED_EPSILON)) {
        // LCOV_EXCL_START
        const CeedInt node_out = (i < num_comp_u * num_dofs_u) ? i % num_dofs_u : (i - num_comp_u * num_dofs_u) % num_dofs_p;
        const CeedInt comp_out = (i < num_comp_u * num_dofs_u) ? i / num_dofs_u : (i - num_comp_u * num_dofs_u) / num_dofs_p;
        const CeedInt node_in  = (j < num_comp_u * num_dofs_u) ? j % num_dofs_u : (j - num_comp_u * num_dofs_u) % num_dofs_p;
        const CeedInt comp_in  = (j < num_comp_u * num_dofs_u) ? j / num_dofs_u : (j - num_comp_u * num_dofs_u) / num_dofs_p;

        printf("[(%s, %" CeedInt_FMT ", %" CeedInt_FMT "), (%s, %" CeedInt_FMT ", %" CeedInt_FMT ")] Error in assembly: %g != %g (error: %g)\n",
               (i < num_comp_u * num_dofs_u ? "u" : "p"), node_out, comp_out, (j < num_comp_u * num_dofs_u ? "u" : "p"), node_in, comp_in,
               assembled_value, assembled_true_value, error);
        // LCOV_EXCL_STOP
      }
    }
  }

  // Cleanup
  free(rows);
  free(cols);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&assembled);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_p);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_p);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedDestroy(&ceed);
  return 0;
}
