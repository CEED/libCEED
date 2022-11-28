/// @file
/// Test creation, action, and destruction for mass matrix composite operator with multigrid level, tensor basis and interpolation basis generation
/// \test Test creation, action, and destruction for mass matrix composite operator with multigrid level, tensor basis and interpolation basis
/// generation
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t502-operator.h"

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedOperator      op_mass_c, op_mass_f, op_prolong, op_restrict;
  CeedVector        X, U_c, U_f, V_c, V_f, p_mult_f;
  const CeedScalar *hv;
  CeedInt           num_elem = 15, num_elem_sub = 5, num_sub_ops = 3, P_c = 3, P_f = 5, Q = 8, num_comp = 2;
  CeedInt           num_dofs_x = num_elem + 1, num_dofs_u_c = num_elem * (P_c - 1) + 1, num_dofs_u_f = num_elem * (P_f - 1) + 1;
  CeedInt           ind_u_c[num_elem_sub * P_c], ind_u_f[num_elem_sub * P_f], ind_x[num_elem_sub * 2];
  CeedScalar        x[num_dofs_x];
  CeedScalar        sum;

  CeedInit(argv[1], &ceed);

  // Composite operators
  CeedCompositeOperatorCreate(ceed, &op_mass_c);
  CeedCompositeOperatorCreate(ceed, &op_mass_f);
  CeedCompositeOperatorCreate(ceed, &op_prolong);
  CeedCompositeOperatorCreate(ceed, &op_restrict);

  // Coordinates
  for (CeedInt i = 0; i < num_dofs_x; i++) x[i] = (CeedScalar)i / (num_dofs_x - 1);
  CeedVectorCreate(ceed, num_dofs_x, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Setup suboperators
  for (CeedInt i = 0; i < num_sub_ops; i++) {
    CeedVector          q_data;
    CeedElemRestriction elem_restr_x, elem_restr_qd_i, elem_restr_u_c, elem_restr_u_f;
    CeedBasis           basis_x, basis_u_c, basis_u_f;
    CeedQFunction       qf_setup, qf_mass;
    CeedOperator        sub_op_setup, sub_op_mass_c, sub_op_mass_f, sub_op_prolong, sub_op_restrict;

    // -- QData
    CeedVectorCreate(ceed, num_elem * Q, &q_data);

    // -- Restrictions
    CeedInt offset = num_elem_sub * i;
    for (CeedInt j = 0; j < num_elem_sub; j++) {
      ind_x[2 * j + 0] = offset + j;
      ind_x[2 * j + 1] = offset + j + 1;
    }
    CeedElemRestrictionCreate(ceed, num_elem_sub, 2, 1, 1, num_dofs_x, CEED_MEM_HOST, CEED_COPY_VALUES, ind_x, &elem_restr_x);

    offset = num_elem_sub * i * (P_c - 1);
    for (CeedInt j = 0; j < num_elem_sub; j++) {
      for (CeedInt k = 0; k < P_c; k++) {
        ind_u_c[P_c * j + k] = offset + j * (P_c - 1) + k;
      }
    }
    CeedElemRestrictionCreate(ceed, num_elem_sub, P_c, num_comp, num_dofs_u_c, num_comp * num_dofs_u_c, CEED_MEM_HOST, CEED_COPY_VALUES, ind_u_c,
                              &elem_restr_u_c);

    offset = num_elem_sub * i * (P_f - 1);
    for (CeedInt j = 0; j < num_elem_sub; j++) {
      for (CeedInt k = 0; k < P_f; k++) {
        ind_u_f[P_f * j + k] = offset + j * (P_f - 1) + k;
      }
    }
    CeedElemRestrictionCreate(ceed, num_elem_sub, P_f, num_comp, num_dofs_u_f, num_comp * num_dofs_u_f, CEED_MEM_HOST, CEED_COPY_VALUES, ind_u_f,
                              &elem_restr_u_f);

    CeedInt strides_qd[3] = {1, Q, Q};
    CeedElemRestrictionCreateStrided(ceed, num_elem_sub, Q, 1, Q * num_elem, strides_qd, &elem_restr_qd_i);

    // -- Bases
    CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &basis_x);
    CeedBasisCreateTensorH1Lagrange(ceed, 1, num_comp, P_c, Q, CEED_GAUSS, &basis_u_c);
    CeedBasisCreateTensorH1Lagrange(ceed, 1, num_comp, P_f, Q, CEED_GAUSS, &basis_u_f);

    // -- QFunctions
    CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
    CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddInput(qf_setup, "dx", 1 * 1, CEED_EVAL_GRAD);
    CeedQFunctionAddOutput(qf_setup, "qdata", 1, CEED_EVAL_NONE);

    CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
    CeedQFunctionAddInput(qf_mass, "qdata", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_mass, "u", num_comp, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_mass, "v", num_comp, CEED_EVAL_INTERP);

    // -- Operators
    CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &sub_op_setup);
    CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &sub_op_mass_f);

    CeedOperatorSetField(sub_op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(sub_op_setup, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(sub_op_setup, "qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

    CeedOperatorSetField(sub_op_mass_f, "qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
    CeedOperatorSetField(sub_op_mass_f, "u", elem_restr_u_f, basis_u_f, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(sub_op_mass_f, "v", elem_restr_u_f, basis_u_f, CEED_VECTOR_ACTIVE);

    // -- Create qdata
    CeedOperatorApply(sub_op_setup, X, q_data, CEED_REQUEST_IMMEDIATE);

    // -- Create multigrid level
    CeedVectorCreate(ceed, num_comp * num_dofs_u_f, &p_mult_f);
    CeedVectorSetValue(p_mult_f, 1.0);
    CeedOperatorMultigridLevelCreate(sub_op_mass_f, p_mult_f, elem_restr_u_c, basis_u_c, &sub_op_mass_c, &sub_op_prolong, &sub_op_restrict);

    // -- Composite operators
    CeedCompositeOperatorAddSub(op_mass_c, sub_op_mass_c);
    CeedCompositeOperatorAddSub(op_mass_f, sub_op_mass_f);
    CeedCompositeOperatorAddSub(op_prolong, sub_op_prolong);
    CeedCompositeOperatorAddSub(op_restrict, sub_op_restrict);

    // -- Cleanup
    CeedVectorDestroy(&p_mult_f);
    CeedVectorDestroy(&q_data);
    CeedElemRestrictionDestroy(&elem_restr_u_c);
    CeedElemRestrictionDestroy(&elem_restr_u_f);
    CeedElemRestrictionDestroy(&elem_restr_x);
    CeedElemRestrictionDestroy(&elem_restr_qd_i);
    CeedBasisDestroy(&basis_u_c);
    CeedBasisDestroy(&basis_u_f);
    CeedBasisDestroy(&basis_x);
    CeedQFunctionDestroy(&qf_setup);
    CeedQFunctionDestroy(&qf_mass);
    CeedOperatorDestroy(&sub_op_setup);
    CeedOperatorDestroy(&sub_op_mass_c);
    CeedOperatorDestroy(&sub_op_mass_f);
    CeedOperatorDestroy(&sub_op_prolong);
    CeedOperatorDestroy(&sub_op_restrict);
  }

  // Scale for suboperator multiplicity
  CeedCompositeOperatorScaleMultigridLevel(op_prolong, op_restrict);

  // Coarse problem
  CeedVectorCreate(ceed, num_comp * num_dofs_u_c, &U_c);
  CeedVectorSetValue(U_c, 1.0);
  CeedVectorCreate(ceed, num_comp * num_dofs_u_c, &V_c);
  CeedOperatorApply(op_mass_c, U_c, V_c, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V_c, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i = 0; i < num_comp * num_dofs_u_c; i++) {
    sum += hv[i];
  }
  if (fabs(sum - 2.) > 1000. * CEED_EPSILON) printf("Computed Area Coarse Grid: %f != True Area: 2.0\n", sum);
  CeedVectorRestoreArrayRead(V_c, &hv);

  // Prolong coarse u
  CeedVectorCreate(ceed, num_comp * num_dofs_u_f, &U_f);
  CeedOperatorApply(op_prolong, U_c, U_f, CEED_REQUEST_IMMEDIATE);

  // Fine problem
  CeedVectorCreate(ceed, num_comp * num_dofs_u_f, &V_f);
  CeedOperatorApply(op_mass_f, U_f, V_f, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V_f, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i = 0; i < num_comp * num_dofs_u_f; i++) {
    sum += hv[i];
  }
  if (fabs(sum - 2.) > 1000. * CEED_EPSILON) printf("Computed Area Fine Grid: %f != True Area: 2.0\n", sum);
  CeedVectorRestoreArrayRead(V_f, &hv);

  // Restrict state to coarse grid
  CeedOperatorApply(op_restrict, V_f, V_c, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V_c, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i = 0; i < num_comp * num_dofs_u_c; i++) {
    sum += hv[i];
  }
  if (fabs(sum - 2.) > 1000. * CEED_EPSILON) printf("Computed Area Coarse Grid: %f != True Area: 2.0\n", sum);
  CeedVectorRestoreArrayRead(V_c, &hv);

  // Cleanup
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U_c);
  CeedVectorDestroy(&U_f);
  CeedVectorDestroy(&V_c);
  CeedVectorDestroy(&V_f);
  CeedOperatorDestroy(&op_mass_c);
  CeedOperatorDestroy(&op_mass_f);
  CeedOperatorDestroy(&op_prolong);
  CeedOperatorDestroy(&op_restrict);
  CeedDestroy(&ceed);
  return 0;
}
