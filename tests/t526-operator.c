/// @file
/// Test FLOP estimation for composite mass matrix operator
/// \test Test FLOP estimation for composite mass matrix operator
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t320-basis.h"

/* The mesh comprises of two rows of 3 quadralaterals followed by one row
     of 6 triangles:
   _ _ _
  |_|_|_|
  |_|_|_|
  |/|/|/|

*/

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedSize            flop_estimate;
  CeedElemRestriction elem_restr_x_tet, elem_restr_u_tet, elem_restr_qd_i_tet, elem_restr_x_hex, elem_restr_u_hex, elem_restr_qd_i_hex;
  CeedBasis           basis_x_tet, basis_u_tet, basis_x_hex, basis_u_hex;
  CeedQFunction       qf_mass;
  CeedOperator        op_mass_tet, op_mass_hex, op_mass;
  CeedVector          q_data_tet, q_data_hex;
  CeedInt             num_elem_tet = 6, P_tet = 6, Q_tet = 4, num_elem_hex = 6, P_hex = 3, Q_hex = 4, dim = 2;
  CeedInt             n_x = 3, n_y = 3, n_x_tet = 3, n_y_tet = 1, n_x_hex = 3;
  CeedInt             row, col, offset;
  CeedInt             num_dofs = (n_x * 2 + 1) * (n_y * 2 + 1), num_qpts_tet = num_elem_tet * Q_tet, num_qpts_hex = num_elem_hex * Q_hex * Q_hex;
  CeedInt             ind_x_tet[num_elem_tet * P_tet], ind_x_hex[num_elem_hex * P_hex * P_hex];
  CeedScalar          q_ref[dim * Q_tet], q_weight[Q_tet];
  CeedScalar          interp[P_tet * Q_tet], grad[dim * P_tet * Q_tet];

  CeedInit(argv[1], &ceed);

  // Qdata Vectors
  CeedVectorCreate(ceed, num_qpts_tet, &q_data_tet);
  CeedVectorCreate(ceed, num_qpts_hex, &q_data_hex);

  // Set up Tet Elements
  for (CeedInt i = 0; i < num_elem_tet / 2; i++) {
    col    = i % n_x_tet;
    row    = i / n_x_tet;
    offset = col * 2 + row * (n_x_tet * 2 + 1) * 2;

    ind_x_tet[i * 2 * P_tet + 0] = 2 + offset;
    ind_x_tet[i * 2 * P_tet + 1] = 9 + offset;
    ind_x_tet[i * 2 * P_tet + 2] = 16 + offset;
    ind_x_tet[i * 2 * P_tet + 3] = 1 + offset;
    ind_x_tet[i * 2 * P_tet + 4] = 8 + offset;
    ind_x_tet[i * 2 * P_tet + 5] = 0 + offset;

    ind_x_tet[i * 2 * P_tet + 6]  = 14 + offset;
    ind_x_tet[i * 2 * P_tet + 7]  = 7 + offset;
    ind_x_tet[i * 2 * P_tet + 8]  = 0 + offset;
    ind_x_tet[i * 2 * P_tet + 9]  = 15 + offset;
    ind_x_tet[i * 2 * P_tet + 10] = 8 + offset;
    ind_x_tet[i * 2 * P_tet + 11] = 16 + offset;
  }

  // -- Restrictions
  CeedElemRestrictionCreate(ceed, num_elem_tet, P_tet, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x_tet, &elem_restr_x_tet);

  CeedElemRestrictionCreate(ceed, num_elem_tet, P_tet, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x_tet, &elem_restr_u_tet);
  CeedInt strides_qd_tet[3] = {1, Q_tet, Q_tet};
  CeedElemRestrictionCreateStrided(ceed, num_elem_tet, Q_tet, 1, num_qpts_tet, strides_qd_tet, &elem_restr_qd_i_tet);

  // -- Bases
  buildmats(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, dim, P_tet, Q_tet, interp, grad, q_ref, q_weight, &basis_x_tet);

  buildmats(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, 1, P_tet, Q_tet, interp, grad, q_ref, q_weight, &basis_u_tet);

  // -- QFunction
  CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_mass);

  // -- Operators
  // ---- Mass Tet
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_tet);
  CeedOperatorSetField(op_mass_tet, "u", elem_restr_u_tet, basis_u_tet, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_tet, "qdata", elem_restr_qd_i_tet, CEED_BASIS_COLLOCATED, q_data_tet);
  CeedOperatorSetField(op_mass_tet, "v", elem_restr_u_tet, basis_u_tet, CEED_VECTOR_ACTIVE);

  // Set up Hex Elements
  for (CeedInt i = 0; i < num_elem_hex; i++) {
    col    = i % n_x_hex;
    row    = i / n_x_hex;
    offset = (n_x_tet * 2 + 1) * (n_y_tet * 2) * (1 + row) + col * 2;
    for (CeedInt j = 0; j < P_hex; j++) {
      for (CeedInt k = 0; k < P_hex; k++) ind_x_hex[P_hex * (P_hex * i + k) + j] = offset + k * (n_x_hex * 2 + 1) + j;
    }
  }

  // -- Restrictions
  CeedElemRestrictionCreate(ceed, num_elem_hex, P_hex * P_hex, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x_hex,
                            &elem_restr_x_hex);

  CeedElemRestrictionCreate(ceed, num_elem_hex, P_hex * P_hex, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x_hex, &elem_restr_u_hex);
  CeedInt strides_qd_hex[3] = {1, Q_hex * Q_hex, Q_hex * Q_hex};
  CeedElemRestrictionCreateStrided(ceed, num_elem_hex, Q_hex * Q_hex, 1, num_qpts_hex, strides_qd_hex, &elem_restr_qd_i_hex);

  // -- Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P_hex, Q_hex, CEED_GAUSS, &basis_x_hex);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P_hex, Q_hex, CEED_GAUSS, &basis_u_hex);

  // -- Operators
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_hex);
  CeedOperatorSetField(op_mass_hex, "u", elem_restr_u_hex, basis_u_hex, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_hex, "qdata", elem_restr_qd_i_hex, CEED_BASIS_COLLOCATED, q_data_hex);
  CeedOperatorSetField(op_mass_hex, "v", elem_restr_u_hex, basis_u_hex, CEED_VECTOR_ACTIVE);

  // Set up Composite Operator
  // -- Create
  CeedCompositeOperatorCreate(ceed, &op_mass);
  // -- Add SubOperators
  CeedCompositeOperatorAddSub(op_mass, op_mass_tet);
  CeedCompositeOperatorAddSub(op_mass, op_mass_hex);

  // Estimate FLOPs
  CeedQFunctionSetUserFlopsEstimate(qf_mass, 1);
  CeedOperatorGetFlopsEstimate(op_mass, &flop_estimate);

  // Check output
  if (flop_estimate != 3042) printf("Incorrect FLOP estimate computed, %ld != 3042\n", flop_estimate);

  // Cleanup
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_mass_tet);
  CeedOperatorDestroy(&op_mass_hex);
  CeedOperatorDestroy(&op_mass);
  CeedElemRestrictionDestroy(&elem_restr_u_tet);
  CeedElemRestrictionDestroy(&elem_restr_x_tet);
  CeedElemRestrictionDestroy(&elem_restr_qd_i_tet);
  CeedElemRestrictionDestroy(&elem_restr_u_hex);
  CeedElemRestrictionDestroy(&elem_restr_x_hex);
  CeedElemRestrictionDestroy(&elem_restr_qd_i_hex);
  CeedBasisDestroy(&basis_u_tet);
  CeedBasisDestroy(&basis_x_tet);
  CeedBasisDestroy(&basis_u_hex);
  CeedBasisDestroy(&basis_x_hex);
  CeedVectorDestroy(&q_data_tet);
  CeedVectorDestroy(&q_data_hex);
  CeedDestroy(&ceed);
  return 0;
}
