/// @file
/// Test assembly of non-symmetric mass matrix operator (multi-component) (see t537)
/// \test Test assembly of non-symmetric mass matrix operator (multi-component)
#include "t566-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, x, u, v;
  CeedInt             p = 3, q = 3, dim = 2, num_comp = 2;
  CeedInt             n_x = 1, n_y = 1;
  CeedInt             num_elem = n_x * n_y;
  CeedInt             num_dofs = (n_x * (p - 1) + 1) * (n_y * (p - 1) + 1), num_qpts = num_elem * q * q;
  CeedInt             ind_x[num_elem * p * p];
  CeedScalar          assembled_values[num_comp * num_comp * num_dofs * num_dofs];
  CeedScalar          assembled_true[num_comp * num_comp * num_dofs * num_dofs];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs, &x);
  {
    CeedScalar x_array[dim * num_dofs];

    for (CeedInt i = 0; i < n_x * (p - 1) + 1; i++) {
      for (CeedInt j = 0; j < n_y * (p - 1) + 1; j++) {
        x_array[i + j * (n_x * 2 + 1) + 0 * num_dofs] = (CeedScalar)i / (n_x * (p - 1));
        x_array[i + j * (n_x * 2 + 1) + 1 * num_dofs] = (CeedScalar)j / (n_y * (p - 1));
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_comp * num_dofs, &u);
  CeedVectorCreate(ceed, num_comp * num_dofs, &v);
  CeedVectorCreate(ceed, num_qpts, &q_data);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;

    col    = i % n_x;
    row    = i / n_x;
    offset = col * (p - 1) + row * (n_x * (p - 1) + 1) * (p - 1);
    for (CeedInt j = 0; j < p; j++) {
      for (CeedInt k = 0; k < p; k++) ind_x[p * (p * i + k) + j] = offset + k * p + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p * p, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
  CeedElemRestrictionCreate(ceed, num_elem, p * p, num_comp, num_dofs, num_comp * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x,
                            &elem_restriction_u);

  CeedInt strides_q_data[3] = {1, q * q * num_elem, q * q};
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

  // Fuly assemble operator
  CeedSize   num_entries;
  CeedInt   *rows;
  CeedInt   *cols;
  CeedVector assembled;

  for (CeedInt k = 0; k < num_comp * num_comp * num_dofs * num_dofs; k++) {
    assembled_values[k] = 0.0;
    assembled_true[k]   = 0.0;
  }
  CeedOperatorLinearAssembleSymbolic(op_mass, &num_entries, &rows, &cols);
  CeedVectorCreate(ceed, num_entries, &assembled);
  CeedOperatorLinearAssemble(op_mass, assembled);
  {
    const CeedScalar *assembled_array;

    CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
    for (CeedInt k = 0; k < num_entries; k++) assembled_values[rows[k] * num_comp * num_dofs + cols[k]] += assembled_array[k];
    CeedVectorRestoreArrayRead(assembled, &assembled_array);
  }

  // Manually assemble operator
  CeedInt old_index = -1;

  CeedVectorSetValue(u, 0.0);
  for (CeedInt comp_in = 0; comp_in < num_comp; comp_in++) {
    for (CeedInt node_in = 0; node_in < num_dofs; node_in++) {
      CeedScalar       *u_array;
      const CeedScalar *v_array;

      // Set input
      CeedVectorGetArray(u, CEED_MEM_HOST, &u_array);
      CeedInt ind  = node_in + comp_in * num_dofs;
      u_array[ind] = 1.0;
      if (ind > 0) u_array[old_index] = 0.0;
      old_index = ind;
      CeedVectorRestoreArray(u, &u_array);

      // Compute effect of DoF j
      CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

      CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
      for (CeedInt k = 0; k < num_dofs * num_comp; k++) assembled_true[k * num_dofs * num_comp + ind] = v_array[k];
      CeedVectorRestoreArrayRead(v, &v_array);
    }
  }

  // Check output
  for (CeedInt node_in = 0; node_in < num_dofs; node_in++) {
    for (CeedInt comp_in = 0; comp_in < num_comp; comp_in++) {
      for (CeedInt node_out = 0; node_out < num_dofs; node_out++) {
        for (CeedInt comp_out = 0; comp_out < num_comp; comp_out++) {
          const CeedInt    index                = (node_out + comp_out * num_dofs) * num_comp + node_in + comp_in * num_dofs;
          const CeedScalar assembled_value      = assembled_values[index];
          const CeedScalar assembled_true_value = assembled_true[index];
          if (fabs(assembled_value - assembled_true_value) > 100. * CEED_EPSILON) {
            // LCOV_EXCL_START
            printf("[(%" CeedInt_FMT ", %" CeedInt_FMT "), (%" CeedInt_FMT ", %" CeedInt_FMT ")] Error in assembly: %f != %f\n", node_out, comp_out,
                   node_in, comp_in, assembled_value, assembled_true_value);
            // LCOV_EXCL_STOP
          }
        }
      }
    }
  }

  // Cleanup
  free(rows);
  free(cols);
  CeedVectorDestroy(&assembled);
  CeedVectorDestroy(&x);
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
