/// @file
/// Test composite operator with a passive output set to a different array before each apply
/// \test Test composite operator with a passive output set to a different array before each apply
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#include "t527-operator.h"

CeedScalar sum(CeedVector v) {
  const CeedScalar *v_array;
  CeedScalar        sum = 0.;
  CeedSize          length;

  CeedVectorGetLength(v, &length);
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
  for (CeedSize i = 0; i < length; i++) sum += v_array[i];
  CeedVectorRestoreArrayRead(v, &v_array);
  return sum;
}

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass, op_composite;
  CeedVector          x, u1, u2, v1, v2, q_data, passive_in, passive_in_arrays[4], passive_out, passive_out_arrays[4];
  CeedMemType         mem_type;
  CeedInt num_elem = 3, num_nodes_x = num_elem + 1, p = 2, q = 3, num_dofs = num_elem * (p - 1) + 1, num_qpts = num_elem * q, num_applies = 4;
  CeedInt ind_x[num_elem * 2], ind_u[num_elem * p];

  CeedInit(argv[1], &ceed);
  CeedGetPreferredMemType(ceed, &mem_type);

  // Vectors
  CeedVectorCreate(ceed, num_nodes_x, &x);
  {
    CeedScalar x_array[num_nodes_x];

    for (CeedInt i = 0; i < num_nodes_x; i++) x_array[i] = (CeedScalar)i / (num_nodes_x - 1);
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_dofs, &u1);
  CeedVectorCreate(ceed, num_dofs, &u2);
  CeedVectorCreate(ceed, num_dofs, &v1);
  CeedVectorCreate(ceed, num_dofs, &v2);
  CeedVectorCreate(ceed, num_qpts, &q_data);
  CeedVectorCreate(ceed, num_qpts, &passive_in);
  CeedVectorSetValue(passive_in, 0);
  CeedVectorCreate(ceed, num_qpts, &passive_out);
  for (CeedInt k = 0; k < num_applies; k++) {
    CeedScalar *arr;

    CeedVectorCreate(ceed, num_qpts, &passive_in_arrays[k]);
    CeedVectorCreate(ceed, num_qpts, &passive_out_arrays[k]);
    CeedVectorGetArrayWrite(passive_in_arrays[k], CEED_MEM_HOST, &arr);
    for (CeedInt i = 0; i < num_qpts; i++) {
      arr[i] = 1000 * k + i;
    }
    CeedVectorRestoreArray(passive_in_arrays[k], &arr);
  }

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_nodes_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p; j++) ind_u[p * i + j] = i * (p - 1) + j;
  }
  CeedElemRestrictionCreate(ceed, num_elem, p, 1, 1, num_dofs, CEED_MEM_HOST, CEED_COPY_VALUES, ind_u, &elem_restriction_u);
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, num_qpts, CEED_STRIDES_BACKEND, &elem_restriction_q);

  // Basis
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, q, CEED_GAUSS, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_setup, "dx", 1, CEED_EVAL_GRAD);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "passive_in", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "passive_out", 1, CEED_EVAL_NONE);

  // Operators, with an active input and a passive output
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", elem_restriction_q, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "rho", elem_restriction_q, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass, "passive_in", elem_restriction_q, CEED_BASIS_NONE, passive_in);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "passive_out", elem_restriction_q, CEED_BASIS_NONE, passive_out);
  CeedOperatorCreateComposite(ceed, &op_composite);
  CeedOperatorCompositeAddSub(op_composite, op_mass);

  // Apply with different active vectors
  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

  // Test changing active input vector
  CeedVectorSetValue(u1, 1.0);
  CeedOperatorApply(op_composite, u1, v1, CEED_REQUEST_IMMEDIATE);
  if (fabs(sum(v1) - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum(v1));
  CeedVectorSetValue(u1, 2.0);
  CeedOperatorApply(op_composite, u1, v1, CEED_REQUEST_IMMEDIATE);
  if (fabs(sum(v1) - 2.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 2.0\n", sum(v1));
  CeedVectorSetValue(u2, 1.0);
  CeedOperatorApply(op_composite, u2, v1, CEED_REQUEST_IMMEDIATE);
  if (fabs(sum(v1) - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum(v1));
  CeedOperatorApply(op_composite, u1, v1, CEED_REQUEST_IMMEDIATE);
  if (fabs(sum(v1) - 2.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 2.0\n", sum(v1));

  // Test changing active output vector
  CeedOperatorApply(op_composite, u1, v1, CEED_REQUEST_IMMEDIATE);
  if (fabs(sum(v1) - 2.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 2.0\n", sum(v1));
  CeedVectorSetValue(u1, 1.0);
  CeedOperatorApply(op_composite, u1, v2, CEED_REQUEST_IMMEDIATE);
  if (fabs(sum(v2) - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum(v2));
  {
    CeedScalar *array;

    CeedVectorGetArrayWrite(v1, mem_type, &array);
    CeedVectorSetArray(v2, mem_type, CEED_USE_POINTER, array);
    CeedOperatorApply(op_composite, u1, v2, CEED_REQUEST_IMMEDIATE);
    CeedVectorTakeArray(v2, mem_type, &array);
    CeedVectorRestoreArray(v1, &array);
  }
  if (fabs(sum(v1) - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum(v1));

  // Check changing passive input
  for (CeedInt k = 0; k < num_applies; k++) {
    const CeedScalar *array;

    CeedVectorGetArrayRead(passive_in_arrays[k], mem_type, &array);
    CeedVectorSetArray(passive_in, mem_type, CEED_USE_POINTER, (CeedScalar *)array);
    CeedOperatorApply(op_composite, u1, v1, CEED_REQUEST_IMMEDIATE);
    CeedVectorTakeArray(passive_in, mem_type, (CeedScalar **)&array);
    CeedVectorRestoreArrayRead(passive_in_arrays[k], &array);
    CeedVectorGetArrayRead(passive_in_arrays[k], CEED_MEM_HOST, &array);
    for (CeedInt i = 0; i < num_qpts; i++) {
      if (fabs(array[i] - (1000. * k + i)) > 1000. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Apply %" CeedInt_FMT ", passive output entry %" CeedInt_FMT ": %f != %f\n", k, i, array[i], (1000. * k + i));
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(passive_in_arrays[k], &array);
  }

  // Check changing passive output
  for (CeedInt k = 0; k < num_applies; k++) {
    CeedScalar       *array;
    const CeedScalar *array_read;

    CeedVectorSetValue(passive_in, k + 1.0);
    CeedVectorGetArrayWrite(passive_out_arrays[k], mem_type, &array);
    CeedVectorSetArray(passive_out, mem_type, CEED_USE_POINTER, (CeedScalar *)array);
    CeedOperatorApply(op_composite, u1, v1, CEED_REQUEST_IMMEDIATE);
    CeedVectorTakeArray(passive_out, mem_type, (CeedScalar **)&array);
    CeedVectorRestoreArray(passive_out_arrays[k], &array);
    CeedVectorGetArrayRead(passive_out_arrays[k], CEED_MEM_HOST, &array_read);
    for (CeedInt i = 0; i < num_qpts; i++) {
      if (fabs(array_read[i] - (k + 1.0)) > 1000. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Apply %" CeedInt_FMT ", passive output entry %" CeedInt_FMT ": %f != %f\n", k, i, array_read[i], (k + 1.0));
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(passive_out_arrays[k], &array_read);
  }

  // Cleanup
  CeedVectorDestroy(&u1);
  CeedVectorDestroy(&u2);
  CeedVectorDestroy(&v1);
  CeedVectorDestroy(&v2);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&passive_in);
  CeedVectorDestroy(&passive_out);
  for (CeedInt k = 0; k < num_applies; k++) CeedVectorDestroy(&passive_in_arrays[k]);
  for (CeedInt k = 0; k < num_applies; k++) CeedVectorDestroy(&passive_out_arrays[k]);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedOperatorDestroy(&op_composite);
  CeedDestroy(&ceed);
  return 0;
}
