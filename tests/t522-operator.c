/// @file
/// Test creation, action, and destruction for diffusion matrix operator
/// \test Test creation, action, and destruction for diffusion matrix operator
#include "t522-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t320-basis.h"

/* The mesh comprises of two rows of 3 quadrilaterals followed by one row
     of 6 triangles:
   _ _ _
  |_|_|_|
  |_|_|_|
  |/|/|/|

*/

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x_tet, elem_restriction_u_tet, elem_restriction_q_data_tet, elem_restriction_x_hex, elem_restriction_u_hex,
      elem_restriction_q_data_hex;
  CeedBasis     basis_x_tet, basis_u_tet, basis_x_hex, basis_u_hex;
  CeedQFunction qf_setup_tet, qf_diff_tet, qf_setup_hex, qf_diff_hex;
  CeedOperator  op_setup_tet, op_diff_tet, op_setup_hex, op_diff_hex, op_setup, op_diff;
  CeedVector    q_data_tet, q_data_hex, x, u, v;
  CeedInt       num_elem_tet = 6, p_tet = 6, q_tet = 4, num_elem_hex = 6, p_hex = 3, q_hex = 4, dim = 2;
  CeedInt       n_x = 3, n_y = 3, n_x_tet = 3, n_y_tet = 1, n_x_hex = 3;
  CeedInt       row, col, offset;
  CeedInt       num_dofs = (n_x * 2 + 1) * (n_y * 2 + 1), num_qpts_tet = num_elem_tet * q_tet, num_qpts_hex = num_elem_hex * q_hex * q_hex;
  CeedInt       ind_x_tet[num_elem_tet * p_tet], ind_x_hex[num_elem_hex * p_hex * p_hex];
  CeedScalar    q_ref[dim * q_tet], q_weight[q_tet];
  CeedScalar    interp[p_tet * q_tet], grad[dim * p_tet * q_tet];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs, &x);
  {
    CeedScalar x_array[dim * num_dofs];

    for (CeedInt i = 0; i < n_y * 2 + 1; i++) {
      for (CeedInt j = 0; j < n_x * 2 + 1; j++) {
        x_array[i + j * (n_y * 2 + 1) + 0 * num_dofs] = (CeedScalar)i / (2 * n_y);
        x_array[i + j * (n_y * 2 + 1) + 1 * num_dofs] = (CeedScalar)j / (2 * n_x);
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorCreate(ceed, num_dofs, &v);
  CeedVectorCreate(ceed, num_qpts_tet * dim * (dim + 1) / 2, &q_data_tet);
  CeedVectorCreate(ceed, num_qpts_hex * dim * (dim + 1) / 2, &q_data_hex);

  // Tet Elements
  // -- Restrictions
  for (CeedInt i = 0; i < num_elem_tet / 2; i++) {
    col    = i % n_x_tet;
    row    = i / n_x_tet;
    offset = col * 2 + row * (n_x_tet * 2 + 1) * 2;

    ind_x_tet[i * 2 * p_tet + 0] = 2 + offset;
    ind_x_tet[i * 2 * p_tet + 1] = 9 + offset;
    ind_x_tet[i * 2 * p_tet + 2] = 16 + offset;
    ind_x_tet[i * 2 * p_tet + 3] = 1 + offset;
    ind_x_tet[i * 2 * p_tet + 4] = 8 + offset;
    ind_x_tet[i * 2 * p_tet + 5] = 0 + offset;

    ind_x_tet[i * 2 * p_tet + 6]  = 14 + offset;
    ind_x_tet[i * 2 * p_tet + 7]  = 7 + offset;
    ind_x_tet[i * 2 * p_tet + 8]  = 0 + offset;
    ind_x_tet[i * 2 * p_tet + 9]  = 15 + offset;
    ind_x_tet[i * 2 * p_tet + 10] = 8 + offset;
    ind_x_tet[i * 2 * p_tet + 11] = 16 + offset;
  }
  CeedElemRestrictionCreate(ceed, num_elem_tet, p_tet, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x_tet,
                            &elem_restriction_x_tet);
  CeedElemRestrictionCreate(ceed, num_elem_tet, p_tet, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x_tet, &elem_restriction_u_tet);

  CeedInt strides_q_data_tet[3] = {1, q_tet, q_tet * dim * (dim + 1) / 2};
  CeedElemRestrictionCreateStrided(ceed, num_elem_tet, q_tet, dim * (dim + 1) / 2, dim * (dim + 1) / 2 * num_qpts_tet, strides_q_data_tet,
                                   &elem_restriction_q_data_tet);

  // -- Bases
  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, dim, p_tet, q_tet, interp, grad, q_ref, q_weight, &basis_x_tet);

  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, 1, p_tet, q_tet, interp, grad, q_ref, q_weight, &basis_u_tet);

  // -- QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup_tet);
  CeedQFunctionAddInput(qf_setup_tet, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup_tet, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup_tet, "rho", dim * (dim + 1) / 2, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, diff, diff_loc, &qf_diff_tet);
  CeedQFunctionAddInput(qf_diff_tet, "rho", dim * (dim + 1) / 2, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_diff_tet, "u", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_diff_tet, "v", dim, CEED_EVAL_GRAD);

  // -- Operators
  // ---- Setup Tet
  CeedOperatorCreate(ceed, qf_setup_tet, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_tet);
  CeedOperatorSetField(op_setup_tet, "weight", CEED_ELEMRESTRICTION_NONE, basis_x_tet, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_tet, "dx", elem_restriction_x_tet, basis_x_tet, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_tet, "rho", elem_restriction_q_data_tet, CEED_BASIS_COLLOCATED, q_data_tet);
  // ---- Diff Tet
  CeedOperatorCreate(ceed, qf_diff_tet, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_diff_tet);
  CeedOperatorSetField(op_diff_tet, "rho", elem_restriction_q_data_tet, CEED_BASIS_COLLOCATED, q_data_tet);
  CeedOperatorSetField(op_diff_tet, "u", elem_restriction_u_tet, basis_u_tet, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff_tet, "v", elem_restriction_u_tet, basis_u_tet, CEED_VECTOR_ACTIVE);

  // Hex Elements
  // -- Restrictions
  for (CeedInt i = 0; i < num_elem_hex; i++) {
    col    = i % n_x_hex;
    row    = i / n_x_hex;
    offset = (n_x_tet * 2 + 1) * (n_y_tet * 2) * (1 + row) + col * 2;
    for (CeedInt j = 0; j < p_hex; j++) {
      for (CeedInt k = 0; k < p_hex; k++) ind_x_hex[p_hex * (p_hex * i + k) + j] = offset + k * (n_x_hex * 2 + 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem_hex, p_hex * p_hex, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x_hex,
                            &elem_restriction_x_hex);
  CeedElemRestrictionCreate(ceed, num_elem_hex, p_hex * p_hex, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x_hex, &elem_restriction_u_hex);

  CeedInt strides_q_data_hex[3] = {1, q_hex * q_hex, q_hex * q_hex * dim * (dim + 1) / 2};
  CeedElemRestrictionCreateStrided(ceed, num_elem_hex, q_hex * q_hex, dim * (dim + 1) / 2, dim * (dim + 1) / 2 * num_qpts_hex, strides_q_data_hex,
                                   &elem_restriction_q_data_hex);

  // -- Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p_hex, q_hex, CEED_GAUSS, &basis_x_hex);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p_hex, q_hex, CEED_GAUSS, &basis_u_hex);

  // -- QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup_hex);
  CeedQFunctionAddInput(qf_setup_hex, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup_hex, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup_hex, "rho", dim * (dim + 1) / 2, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, diff, diff_loc, &qf_diff_hex);
  CeedQFunctionAddInput(qf_diff_hex, "rho", dim * (dim + 1) / 2, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_diff_hex, "u", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_diff_hex, "v", dim, CEED_EVAL_GRAD);

  // -- Operators
  CeedOperatorCreate(ceed, qf_setup_hex, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_hex);
  CeedOperatorSetField(op_setup_hex, "weight", CEED_ELEMRESTRICTION_NONE, basis_x_hex, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_hex, "dx", elem_restriction_x_hex, basis_x_hex, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_hex, "rho", elem_restriction_q_data_hex, CEED_BASIS_COLLOCATED, q_data_hex);

  CeedOperatorCreate(ceed, qf_diff_hex, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_diff_hex);
  CeedOperatorSetField(op_diff_hex, "rho", elem_restriction_q_data_hex, CEED_BASIS_COLLOCATED, q_data_hex);
  CeedOperatorSetField(op_diff_hex, "u", elem_restriction_u_hex, basis_u_hex, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff_hex, "v", elem_restriction_u_hex, basis_u_hex, CEED_VECTOR_ACTIVE);

  // Composite Operators
  CeedCompositeOperatorCreate(ceed, &op_setup);
  CeedCompositeOperatorAddSub(op_setup, op_setup_tet);
  CeedCompositeOperatorAddSub(op_setup, op_setup_hex);

  CeedCompositeOperatorCreate(ceed, &op_diff);
  CeedCompositeOperatorAddSub(op_diff, op_diff_tet);
  CeedCompositeOperatorAddSub(op_diff, op_diff_hex);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, x, CEED_VECTOR_NONE, CEED_REQUEST_IMMEDIATE);

  // Apply diff Operator
  CeedVectorSetValue(u, 1.0);
  CeedOperatorApply(op_diff, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_dofs; i++) {
      if (fabs(v_array[i]) > 100. * CEED_EPSILON) printf("Computed: %f != True: 0.0\n", v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Cleanup
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&q_data_tet);
  CeedVectorDestroy(&q_data_hex);
  CeedElemRestrictionDestroy(&elem_restriction_u_tet);
  CeedElemRestrictionDestroy(&elem_restriction_x_tet);
  CeedElemRestrictionDestroy(&elem_restriction_q_data_tet);
  CeedElemRestrictionDestroy(&elem_restriction_u_hex);
  CeedElemRestrictionDestroy(&elem_restriction_x_hex);
  CeedElemRestrictionDestroy(&elem_restriction_q_data_hex);
  CeedBasisDestroy(&basis_u_tet);
  CeedBasisDestroy(&basis_x_tet);
  CeedBasisDestroy(&basis_u_hex);
  CeedBasisDestroy(&basis_x_hex);
  CeedQFunctionDestroy(&qf_setup_tet);
  CeedQFunctionDestroy(&qf_diff_tet);
  CeedOperatorDestroy(&op_setup_tet);
  CeedOperatorDestroy(&op_diff_tet);
  CeedQFunctionDestroy(&qf_setup_hex);
  CeedQFunctionDestroy(&qf_diff_hex);
  CeedOperatorDestroy(&op_setup_hex);
  CeedOperatorDestroy(&op_diff_hex);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_diff);
  CeedDestroy(&ceed);
  return 0;
}
