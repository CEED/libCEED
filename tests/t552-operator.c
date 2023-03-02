/// @file
/// Test creation, action, and destruction for mass matrix operator with multigrid level, tensor basis
/// \test Test creation, action, and destruction for mass matrix operator with multigrid level, tensor basis
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t502-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_q_data, elem_restriction_u_coarse, elem_restr_coarse_fine;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass_coarse, op_mass_fine, op_prolong, op_restrict;
  CeedVector          q_data, x, u_coarse, u_fine, v_coarse, v_fine, p_mult_fine;
  CeedInt             num_elem = 15, p_coarse = 3, p_fine = 5, q = 8, num_comp = 2;
  CeedInt             num_dofs_x = num_elem + 1, num_dofs_u_coarse = num_elem * (p_coarse - 1) + 1, num_dofs_u_fine = num_elem * (p_fine - 1) + 1;
  CeedInt             ind_u_coarse[num_elem * p_coarse], ind_u_fine[num_elem * p_fine], ind_x[num_elem * 2];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_dofs_x, &x);
  {
    CeedScalar x_array[num_dofs_x];

    for (CeedInt i = 0; i < num_dofs_x; i++) x_array[i] = (CeedScalar)i / (num_dofs_x - 1);
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_comp * num_dofs_u_fine, &p_mult_fine);
  CeedVectorCreate(ceed, num_comp * num_dofs_u_coarse, &u_coarse);
  CeedVectorCreate(ceed, num_comp * num_dofs_u_coarse, &v_coarse);
  CeedVectorCreate(ceed, num_comp * num_dofs_u_fine, &u_fine);
  CeedVectorCreate(ceed, num_comp * num_dofs_u_fine, &v_fine);
  CeedVectorCreate(ceed, num_elem * q, &q_data);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_dofs_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p_coarse; j++) {
      ind_u_coarse[p_coarse * i + j] = i * (p_coarse - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p_coarse, num_comp, num_dofs_u_coarse, num_comp * num_dofs_u_coarse, CEED_MEM_HOST, CEED_USE_POINTER,
                            ind_u_coarse, &elem_restriction_u_coarse);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p_fine; j++) {
      ind_u_fine[p_fine * i + j] = i * (p_fine - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p_fine, num_comp, num_dofs_u_fine, num_comp * num_dofs_u_fine, CEED_MEM_HOST, CEED_USE_POINTER,
                            ind_u_fine, &elem_restr_coarse_fine);

  CeedInt strides_q_data[3] = {1, q, q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, q * num_elem, strides_q_data, &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, num_comp, p_fine, q, CEED_GAUSS, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1 * 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "q data", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "q data", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", num_comp, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", num_comp, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "q data", elem_restriction_q_data, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_fine);
  CeedOperatorSetField(op_mass_fine, "q data", elem_restriction_q_data, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_mass_fine, "u", elem_restr_coarse_fine, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_fine, "v", elem_restr_coarse_fine, basis_u, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

  // Create multigrid level
  CeedVectorSetValue(p_mult_fine, 1.0);
  CeedBasis basis_u_coarse, basis_coarse_to_fine;
  CeedBasisCreateTensorH1Lagrange(ceed, 1, num_comp, p_coarse, q, CEED_GAUSS, &basis_u_coarse);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, num_comp, p_coarse, p_fine, CEED_GAUSS_LOBATTO, &basis_coarse_to_fine);
  const CeedScalar *interp_coarse_to_fine;
  CeedBasisGetInterp1D(basis_coarse_to_fine, &interp_coarse_to_fine);
  CeedOperatorMultigridLevelCreateTensorH1(op_mass_fine, p_mult_fine, elem_restriction_u_coarse, basis_u_coarse, interp_coarse_to_fine,
                                           &op_mass_coarse, &op_prolong, &op_restrict);

  // Coarse problem
  CeedVectorSetValue(u_coarse, 1.0);
  CeedOperatorApply(op_mass_coarse, u_coarse, v_coarse, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        sum = 0.;

    CeedVectorGetArrayRead(v_coarse, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_comp * num_dofs_u_coarse; i++) {
      sum += v_array[i];
    }
    CeedVectorRestoreArrayRead(v_coarse, &v_array);
    if (fabs(sum - 2.) > 1000. * CEED_EPSILON) printf("Computed Area Coarse Grid: %f != True Area: 2.0\n", sum);
  }

  // Prolong coarse u
  CeedOperatorApply(op_prolong, u_coarse, u_fine, CEED_REQUEST_IMMEDIATE);

  // Fine problem
  CeedOperatorApply(op_mass_fine, u_fine, v_fine, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        sum = 0.;
    CeedVectorGetArrayRead(v_fine, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_comp * num_dofs_u_fine; i++) {
      sum += v_array[i];
    }
    CeedVectorRestoreArrayRead(v_fine, &v_array);
    if (fabs(sum - 2.) > 1000. * CEED_EPSILON) printf("Computed Area Fine Grid: %f != True Area: 2.0\n", sum);
  }

  // Restrict state to coarse grid
  CeedOperatorApply(op_restrict, v_fine, v_coarse, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        sum = 0.;
    CeedVectorGetArrayRead(v_coarse, CEED_MEM_HOST, &v_array);
    sum = 0.;
    for (CeedInt i = 0; i < num_comp * num_dofs_u_coarse; i++) {
      sum += v_array[i];
    }
    CeedVectorRestoreArrayRead(v_coarse, &v_array);
    if (fabs(sum - 2.) > 1000. * CEED_EPSILON) printf("Computed Area Coarse Grid: %f != True Area: 2.0\n", sum);
  }

  // Cleanup
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&u_coarse);
  CeedVectorDestroy(&u_fine);
  CeedVectorDestroy(&v_coarse);
  CeedVectorDestroy(&v_fine);
  CeedVectorDestroy(&p_mult_fine);
  CeedVectorDestroy(&q_data);
  CeedElemRestrictionDestroy(&elem_restriction_u_coarse);
  CeedElemRestrictionDestroy(&elem_restr_coarse_fine);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass_coarse);
  CeedOperatorDestroy(&op_mass_fine);
  CeedOperatorDestroy(&op_prolong);
  CeedOperatorDestroy(&op_restrict);
  CeedDestroy(&ceed);
  return 0;
}
