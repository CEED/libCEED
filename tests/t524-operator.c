/// @file
/// Test CeedOperatorApplyAdd for composite operator
/// \test CeedOperatorApplyAdd for composite operator
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t320-basis.h"
#include "t510-operator.h"

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
  CeedQFunction qf_setup_tet, qf_mass_tet, qf_setup_hex, qf_mass_hex;
  CeedOperator  op_setup_tet, op_mass_tet, op_setup_hex, op_mass_hex, op_setup, op_mass;
  CeedVector    q_data_tet, q_data_hex, x, u, v;
  CeedInt       num_elem_tet = 6, p_tet = 6, q_tet = 4, num_elem_hex = 6, p_hex = 3, q_hex = 4, dim = 2;
  CeedInt       nx = 3, ny = 3, nx_tet = 3, ny_tet = 1, nx_hex = 3;
  CeedInt       row, col, offset;
  CeedInt       num_dofs = (nx * 2 + 1) * (ny * 2 + 1), num_qpts_tet = num_elem_tet * q_tet, num_qpts_hex = num_elem_hex * q_hex * q_hex;
  CeedInt       ind_x_tet[num_elem_tet * p_tet], ind_x_hex[num_elem_hex * p_hex * p_hex];
  CeedScalar    q_ref[dim * q_tet], q_weight[q_tet];
  CeedScalar    interp[p_tet * q_tet], grad[dim * p_tet * q_tet];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs, &x);
  {
    CeedScalar x_array[dim * num_dofs];
    for (CeedInt i = 0; i < ny * 2 + 1; i++) {
      for (CeedInt j = 0; j < nx * 2 + 1; j++) {
        x_array[i + j * (ny * 2 + 1) + 0 * num_dofs] = (CeedScalar)i / (2 * ny);
        x_array[i + j * (ny * 2 + 1) + 1 * num_dofs] = (CeedScalar)j / (2 * nx);
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorCreate(ceed, num_dofs, &v);
  CeedVectorCreate(ceed, num_qpts_tet, &q_data_tet);
  CeedVectorCreate(ceed, num_qpts_hex, &q_data_hex);

  // Tet Elements
  // -- Restrictions
  for (CeedInt i = 0; i < num_elem_tet / 2; i++) {
    col    = i % nx_tet;
    row    = i / nx_tet;
    offset = col * 2 + row * (nx_tet * 2 + 1) * 2;

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

  CeedInt strides_q_data_tet[3] = {1, q_tet, q_tet};
  CeedElemRestrictionCreateStrided(ceed, num_elem_tet, q_tet, 1, num_qpts_tet, strides_q_data_tet, &elem_restriction_q_data_tet);

  // -- Bases
  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, dim, p_tet, q_tet, interp, grad, q_ref, q_weight, &basis_x_tet);

  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, 1, p_tet, q_tet, interp, grad, q_ref, q_weight, &basis_u_tet);

  // -- QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup_tet);
  CeedQFunctionAddInput(qf_setup_tet, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup_tet, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup_tet, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass_tet);
  CeedQFunctionAddInput(qf_mass_tet, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass_tet, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass_tet, "v", 1, CEED_EVAL_INTERP);

  // -- Operators
  // ---- Setup _tet
  CeedOperatorCreate(ceed, qf_setup_tet, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_tet);
  CeedOperatorSetField(op_setup_tet, "weight", CEED_ELEMRESTRICTION_NONE, basis_x_tet, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_tet, "dx", elem_restriction_x_tet, basis_x_tet, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_tet, "rho", elem_restriction_q_data_tet, CEED_BASIS_COLLOCATED, q_data_tet);
  // ---- Mass _tet
  CeedOperatorCreate(ceed, qf_mass_tet, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_tet);
  CeedOperatorSetField(op_mass_tet, "rho", elem_restriction_q_data_tet, CEED_BASIS_COLLOCATED, q_data_tet);
  CeedOperatorSetField(op_mass_tet, "u", elem_restriction_u_tet, basis_u_tet, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_tet, "v", elem_restriction_u_tet, basis_u_tet, CEED_VECTOR_ACTIVE);

  // Hex Elements
  // -- Restrictions
  for (CeedInt i = 0; i < num_elem_hex; i++) {
    col    = i % nx_hex;
    row    = i / nx_hex;
    offset = (nx_tet * 2 + 1) * (ny_tet * 2) * (1 + row) + col * 2;
    for (CeedInt j = 0; j < p_hex; j++) {
      for (CeedInt k = 0; k < p_hex; k++) ind_x_hex[p_hex * (p_hex * i + k) + j] = offset + k * (nx_hex * 2 + 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem_hex, p_hex * p_hex, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x_hex,
                            &elem_restriction_x_hex);
  CeedElemRestrictionCreate(ceed, num_elem_hex, p_hex * p_hex, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x_hex, &elem_restriction_u_hex);

  CeedInt strides_q_data_hex[3] = {1, q_hex * q_hex, q_hex * q_hex};
  CeedElemRestrictionCreateStrided(ceed, num_elem_hex, q_hex * q_hex, 1, num_qpts_hex, strides_q_data_hex, &elem_restriction_q_data_hex);

  // -- Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p_hex, q_hex, CEED_GAUSS, &basis_x_hex);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p_hex, q_hex, CEED_GAUSS, &basis_u_hex);

  // -- QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup_hex);
  CeedQFunctionAddInput(qf_setup_hex, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup_hex, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup_hex, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass_hex);
  CeedQFunctionAddInput(qf_mass_hex, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass_hex, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass_hex, "v", 1, CEED_EVAL_INTERP);

  // -- Operators
  CeedOperatorCreate(ceed, qf_setup_hex, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_hex);
  CeedOperatorSetField(op_setup_hex, "weight", CEED_ELEMRESTRICTION_NONE, basis_x_hex, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_hex, "dx", elem_restriction_x_hex, basis_x_hex, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_hex, "rho", elem_restriction_q_data_hex, CEED_BASIS_COLLOCATED, q_data_hex);

  CeedOperatorCreate(ceed, qf_mass_hex, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_hex);
  CeedOperatorSetField(op_mass_hex, "rho", elem_restriction_q_data_hex, CEED_BASIS_COLLOCATED, q_data_hex);
  CeedOperatorSetField(op_mass_hex, "u", elem_restriction_u_hex, basis_u_hex, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_hex, "v", elem_restriction_u_hex, basis_u_hex, CEED_VECTOR_ACTIVE);

  // Composite Operators
  CeedCompositeOperatorCreate(ceed, &op_setup);
  CeedCompositeOperatorAddSub(op_setup, op_setup_tet);
  CeedCompositeOperatorAddSub(op_setup, op_setup_hex);

  CeedCompositeOperatorCreate(ceed, &op_mass);
  CeedCompositeOperatorAddSub(op_mass, op_mass_tet);
  CeedCompositeOperatorAddSub(op_mass, op_mass_hex);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, x, CEED_VECTOR_NONE, CEED_REQUEST_IMMEDIATE);

  // Apply Mass Operator
  CeedVectorSetValue(u, 1.0);
  CeedVectorSetValue(v, 0.0);
  CeedOperatorApplyAdd(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        sum = 0.;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_dofs; i++) sum += v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
    if (fabs(sum - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum);
  }

  // Apply Add
  CeedVectorSetValue(v, 1.0);
  CeedOperatorApplyAdd(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        sum = -num_dofs;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_dofs; i++) sum += v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
    if (fabs(sum - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum);
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
  CeedQFunctionDestroy(&qf_mass_tet);
  CeedOperatorDestroy(&op_setup_tet);
  CeedOperatorDestroy(&op_mass_tet);
  CeedQFunctionDestroy(&qf_setup_hex);
  CeedQFunctionDestroy(&qf_mass_hex);
  CeedOperatorDestroy(&op_setup_hex);
  CeedOperatorDestroy(&op_mass_hex);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedDestroy(&ceed);
  return 0;
}
