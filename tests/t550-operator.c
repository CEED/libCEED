/// @file
/// Test creation, action, and destruction for mass matrix operator with multigrid level, tensor basis and interpolation basis generation
/// \test Test creation, action, and destruction for mass matrix operator with multigrid level, tensor basis and interpolation basis generation
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t502-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x, elem_restr_qd_i, elem_restr_u_c, elem_restr_u_f;
  CeedBasis           basis_x, basis_u_c, basis_u_f;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass_c, op_mass_f, op_prolong, op_restrict;
  CeedVector          q_data, X, U_c, U_f, V_c, V_f, p_mult_f;
  const CeedScalar   *hv;
  CeedInt             num_elem = 15, P_c = 3, P_f = 5, Q = 8, num_comp = 2;
  CeedInt             num_dofs_x = num_elem + 1, num_dofs_u_c = num_elem * (P_c - 1) + 1, num_dofs_u_f = num_elem * (P_f - 1) + 1;
  CeedInt             ind_u_c[num_elem * P_c], ind_u_f[num_elem * P_f], ind_x[num_elem * 2];
  CeedScalar          x[num_dofs_x];
  CeedScalar          sum;

  CeedInit(argv[1], &ceed);

  for (CeedInt i = 0; i < num_dofs_x; i++) x[i] = (CeedScalar)i / (num_dofs_x - 1);
  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  // Restrictions
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_dofs_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_x);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < P_c; j++) {
      ind_u_c[P_c * i + j] = i * (P_c - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, P_c, num_comp, num_dofs_u_c, num_comp * num_dofs_u_c, CEED_MEM_HOST, CEED_USE_POINTER, ind_u_c,
                            &elem_restr_u_c);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < P_f; j++) {
      ind_u_f[P_f * i + j] = i * (P_f - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, P_f, num_comp, num_dofs_u_f, num_comp * num_dofs_u_f, CEED_MEM_HOST, CEED_USE_POINTER, ind_u_f,
                            &elem_restr_u_f);

  CeedInt strides_qd[3] = {1, Q, Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q, 1, Q * num_elem, strides_qd, &elem_restr_qd_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, num_comp, P_c, Q, CEED_GAUSS, &basis_u_c);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, num_comp, P_f, Q, CEED_GAUSS, &basis_u_f);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1 * 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "qdata", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "qdata", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", num_comp, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", num_comp, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_f);

  CeedVectorCreate(ceed, num_dofs_x, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorCreate(ceed, num_elem * Q, &q_data);

  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass_f, "qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_mass_f, "u", elem_restr_u_f, basis_u_f, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_f, "v", elem_restr_u_f, basis_u_f, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_setup, X, q_data, CEED_REQUEST_IMMEDIATE);

  // Create multigrid level
  CeedVectorCreate(ceed, num_comp * num_dofs_u_f, &p_mult_f);
  CeedVectorSetValue(p_mult_f, 1.0);
  CeedOperatorMultigridLevelCreate(op_mass_f, p_mult_f, elem_restr_u_c, basis_u_c, &op_mass_c, &op_prolong, &op_restrict);

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
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass_c);
  CeedOperatorDestroy(&op_mass_f);
  CeedOperatorDestroy(&op_prolong);
  CeedOperatorDestroy(&op_restrict);
  CeedElemRestrictionDestroy(&elem_restr_u_c);
  CeedElemRestrictionDestroy(&elem_restr_u_f);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_qd_i);
  CeedBasisDestroy(&basis_u_c);
  CeedBasisDestroy(&basis_u_f);
  CeedBasisDestroy(&basis_x);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U_c);
  CeedVectorDestroy(&U_f);
  CeedVectorDestroy(&V_c);
  CeedVectorDestroy(&V_f);
  CeedVectorDestroy(&p_mult_f);
  CeedVectorDestroy(&q_data);
  CeedDestroy(&ceed);
  return 0;
}
