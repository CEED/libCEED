/// @file
/// Test creation, action, and destruction for mass matrix composite operator with multigrid level, tensor basis and interpolation basis generation
/// \test Test creation, action, and destruction for mass matrix composite operator with multigrid level, tensor basis and interpolation basis
/// generation
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t502-operator.h"

int main(int argc, char **argv) {
  Ceed         ceed;
  CeedOperator op_mass_coarse, op_mass_fine, op_prolong, op_restrict;
  CeedVector   x, u_coarse, u_fine, v_coarse, v_fine, p_mult_fine;
  CeedInt      num_elem = 15, num_elem_sub = 5, num_sub_ops = 3, p_coarse = 3, p_fine = 5, q = 8, num_comp = 2;
  CeedInt      num_dofs_x = num_elem + 1, num_dofs_u_coarse = num_elem * (p_coarse - 1) + 1, num_dofs_u_fine = num_elem * (p_fine - 1) + 1;
  CeedInt      ind_u_coarse[num_elem_sub * p_coarse], ind_u_fine[num_elem_sub * p_fine], ind_x[num_elem_sub * 2];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, num_dofs_x, &x);
  {
    CeedScalar x_array[num_dofs_x];

    for (CeedInt i = 0; i < num_dofs_x; i++) x_array[i] = (CeedScalar)i / (num_dofs_x - 1);
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_comp * num_dofs_u_coarse, &u_coarse);
  CeedVectorCreate(ceed, num_comp * num_dofs_u_coarse, &v_coarse);
  CeedVectorCreate(ceed, num_comp * num_dofs_u_fine, &u_fine);
  CeedVectorCreate(ceed, num_comp * num_dofs_u_fine, &v_fine);

  // Composite operators
  CeedCompositeOperatorCreate(ceed, &op_mass_coarse);
  CeedCompositeOperatorCreate(ceed, &op_mass_fine);
  CeedCompositeOperatorCreate(ceed, &op_prolong);
  CeedCompositeOperatorCreate(ceed, &op_restrict);

  // Setup fine suboperators
  for (CeedInt i = 0; i < num_sub_ops; i++) {
    CeedVector          q_data;
    CeedElemRestriction elem_restriction_x, elem_restriction_q_data, elem_restriction_u_fine;
    CeedBasis           basis_x, basis_u_fine;
    CeedQFunction       qf_setup, qf_mass;
    CeedOperator        sub_op_setup, sub_op_mass_fine;

    // -- QData
    CeedVectorCreate(ceed, num_elem * q, &q_data);

    // -- Restrictions
    CeedInt offset = num_elem_sub * i;
    for (CeedInt j = 0; j < num_elem_sub; j++) {
      ind_x[2 * j + 0] = offset + j;
      ind_x[2 * j + 1] = offset + j + 1;
    }
    CeedElemRestrictionCreate(ceed, num_elem_sub, 2, 1, 1, num_dofs_x, CEED_MEM_HOST, CEED_COPY_VALUES, ind_x, &elem_restriction_x);

    offset = num_elem_sub * i * (p_fine - 1);
    for (CeedInt j = 0; j < num_elem_sub; j++) {
      for (CeedInt k = 0; k < p_fine; k++) {
        ind_u_fine[p_fine * j + k] = offset + j * (p_fine - 1) + k;
      }
    }
    CeedElemRestrictionCreate(ceed, num_elem_sub, p_fine, num_comp, num_dofs_u_fine, num_comp * num_dofs_u_fine, CEED_MEM_HOST, CEED_COPY_VALUES,
                              ind_u_fine, &elem_restriction_u_fine);

    CeedInt strides_q_data[3] = {1, q, q};
    CeedElemRestrictionCreateStrided(ceed, num_elem_sub, q, 1, q * num_elem, strides_q_data, &elem_restriction_q_data);

    // -- Bases
    CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x);
    CeedBasisCreateTensorH1Lagrange(ceed, 1, num_comp, p_fine, q, CEED_GAUSS, &basis_u_fine);

    // -- QFunctions
    CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
    CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddInput(qf_setup, "dx", 1 * 1, CEED_EVAL_GRAD);
    CeedQFunctionAddOutput(qf_setup, "q data", 1, CEED_EVAL_NONE);

    CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
    CeedQFunctionAddInput(qf_mass, "q data", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_mass, "u", num_comp, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_mass, "v", num_comp, CEED_EVAL_INTERP);

    // -- Operators
    CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &sub_op_setup);
    CeedOperatorSetField(sub_op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(sub_op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(sub_op_setup, "q data", elem_restriction_q_data, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

    CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &sub_op_mass_fine);
    CeedOperatorSetField(sub_op_mass_fine, "q data", elem_restriction_q_data, CEED_BASIS_COLLOCATED, q_data);
    CeedOperatorSetField(sub_op_mass_fine, "u", elem_restriction_u_fine, basis_u_fine, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(sub_op_mass_fine, "v", elem_restriction_u_fine, basis_u_fine, CEED_VECTOR_ACTIVE);

    // -- Create qdata
    CeedOperatorApply(sub_op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

    // -- Composite operators
    CeedCompositeOperatorAddSub(op_mass_fine, sub_op_mass_fine);

    // -- Cleanup
    CeedVectorDestroy(&q_data);
    CeedElemRestrictionDestroy(&elem_restriction_u_fine);
    CeedElemRestrictionDestroy(&elem_restriction_x);
    CeedElemRestrictionDestroy(&elem_restriction_q_data);
    CeedBasisDestroy(&basis_u_fine);
    CeedBasisDestroy(&basis_x);
    CeedQFunctionDestroy(&qf_setup);
    CeedQFunctionDestroy(&qf_mass);
    CeedOperatorDestroy(&sub_op_setup);
    CeedOperatorDestroy(&sub_op_mass_fine);
  }

  // Scale for suboperator multiplicity
  CeedVectorCreate(ceed, num_comp * num_dofs_u_fine, &p_mult_fine);
  CeedCompositeOperatorGetMultiplicity(op_mass_fine, 0, NULL, p_mult_fine);

  // Setup coarse and prolong/restriction suboperators
  for (CeedInt i = 0; i < num_sub_ops; i++) {
    CeedElemRestriction elem_restriction_u_coarse;
    CeedBasis           basis_u_coarse;
    CeedOperator       *sub_ops_mass_fine, sub_op_mass_coarse, sub_op_prolong, sub_op_restrict;

    // -- Fine grid operator
    CeedCompositeOperatorGetSubList(op_mass_fine, &sub_ops_mass_fine);

    // -- Restrictions
    CeedInt offset = num_elem_sub * i * (p_coarse - 1);
    for (CeedInt j = 0; j < num_elem_sub; j++) {
      for (CeedInt k = 0; k < p_coarse; k++) {
        ind_u_coarse[p_coarse * j + k] = offset + j * (p_coarse - 1) + k;
      }
    }
    CeedElemRestrictionCreate(ceed, num_elem_sub, p_coarse, num_comp, num_dofs_u_coarse, num_comp * num_dofs_u_coarse, CEED_MEM_HOST,
                              CEED_COPY_VALUES, ind_u_coarse, &elem_restriction_u_coarse);

    // -- Bases
    CeedBasisCreateTensorH1Lagrange(ceed, 1, num_comp, p_coarse, q, CEED_GAUSS, &basis_u_coarse);

    // -- Create multigrid level
    CeedOperatorMultigridLevelCreate(sub_ops_mass_fine[i], p_mult_fine, elem_restriction_u_coarse, basis_u_coarse, &sub_op_mass_coarse,
                                     &sub_op_prolong, &sub_op_restrict);

    // -- Composite operators
    CeedCompositeOperatorAddSub(op_mass_coarse, sub_op_mass_coarse);
    CeedCompositeOperatorAddSub(op_prolong, sub_op_prolong);
    CeedCompositeOperatorAddSub(op_restrict, sub_op_restrict);

    // -- Cleanup
    CeedElemRestrictionDestroy(&elem_restriction_u_coarse);
    CeedBasisDestroy(&basis_u_coarse);
    CeedOperatorDestroy(&sub_op_mass_coarse);
    CeedOperatorDestroy(&sub_op_prolong);
    CeedOperatorDestroy(&sub_op_restrict);
  }

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
  CeedOperatorDestroy(&op_mass_coarse);
  CeedOperatorDestroy(&op_mass_fine);
  CeedOperatorDestroy(&op_prolong);
  CeedOperatorDestroy(&op_restrict);
  CeedDestroy(&ceed);
  return 0;
}
