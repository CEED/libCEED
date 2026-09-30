/// @file
/// Test CeedOperatorApply for a composite operator overwrites the output, including entries shared by suboperators and entries no element contributes to
/// \test Test CeedOperatorApply for a composite operator overwrites the output, including entries shared by suboperators and entries no element contributes to
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t500-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x[2], elem_restriction_u[2], elem_restriction_q_data[2];
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup[2], op_mass[2], op_composite;
  CeedVector          q_data[2], x, u, v;
  CeedInt             num_elem = 15, p = 5, q = 8;
  CeedInt             num_elem_part[2] = {7, num_elem - 7}, first_elem_part[2] = {0, 7};
  CeedInt             num_nodes_x = num_elem + 1, num_nodes_u = num_elem * (p - 1) + 1;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_nodes_x, &x);
  {
    CeedScalar x_array[num_nodes_x];
    for (CeedInt i = 0; i < num_nodes_x; i++) x_array[i] = (CeedScalar)i / (num_nodes_x - 1);
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_nodes_u + 1, &u);
  CeedVectorCreate(ceed, num_nodes_u + 1, &v);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, q, CEED_GAUSS, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  // Suboperators on the two parts of the mesh, which share the node between them
  CeedOperatorCreateComposite(ceed, &op_composite);
  for (CeedInt part = 0; part < 2; part++) {
    const CeedInt num_elem_p = num_elem_part[part], first_elem = first_elem_part[part];
    CeedInt       ind_x[num_elem_p * 2], ind_u[num_elem_p * p];

    // Restrictions
    for (CeedInt i = 0; i < num_elem_p; i++) {
      ind_x[2 * i + 0] = first_elem + i;
      ind_x[2 * i + 1] = first_elem + i + 1;
    }
    CeedElemRestrictionCreate(ceed, num_elem_p, 2, 1, 1, num_nodes_x, CEED_MEM_HOST, CEED_COPY_VALUES, ind_x, &elem_restriction_x[part]);

    for (CeedInt i = 0; i < num_elem_p; i++) {
      for (CeedInt j = 0; j < p; j++) {
        ind_u[p * i + j] = (first_elem + i) * (p - 1) + j;
      }
    }
    // Last L-vector entry has no element contributions
    CeedElemRestrictionCreate(ceed, num_elem_p, p, 1, 1, num_nodes_u + 1, CEED_MEM_HOST, CEED_COPY_VALUES, ind_u, &elem_restriction_u[part]);

    CeedInt strides_q_data[3] = {1, q, q};
    CeedElemRestrictionCreateStrided(ceed, num_elem_p, q, 1, q * num_elem_p, strides_q_data, &elem_restriction_q_data[part]);
    CeedVectorCreate(ceed, num_elem_p * q, &q_data[part]);

    // Operators
    CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup[part]);
    CeedOperatorSetField(op_setup[part], "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_setup[part], "dx", elem_restriction_x[part], basis_x, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setup[part], "rho", elem_restriction_q_data[part], CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
    CeedOperatorApply(op_setup[part], x, q_data[part], CEED_REQUEST_IMMEDIATE);

    CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass[part]);
    CeedOperatorSetField(op_mass[part], "rho", elem_restriction_q_data[part], CEED_BASIS_NONE, q_data[part]);
    CeedOperatorSetField(op_mass[part], "u", elem_restriction_u[part], basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_mass[part], "v", elem_restriction_u[part], basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorCompositeAddSub(op_composite, op_mass[part]);
  }

  // Apply with V = 1
  CeedVectorSetValue(u, 1.0);
  CeedVectorSetValue(v, 1.0);
  CeedOperatorApply(op_composite, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        sum = 0.;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_nodes_u; i++) sum += v_array[i];
    if (fabs(sum - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum);
    if (v_array[num_nodes_u] != 0.0) printf("Entry without contributions: %f != 0.0\n", v_array[num_nodes_u]);
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  for (CeedInt part = 0; part < 2; part++) {
    CeedVectorDestroy(&q_data[part]);
    CeedElemRestrictionDestroy(&elem_restriction_x[part]);
    CeedElemRestrictionDestroy(&elem_restriction_u[part]);
    CeedElemRestrictionDestroy(&elem_restriction_q_data[part]);
    CeedOperatorDestroy(&op_setup[part]);
    CeedOperatorDestroy(&op_mass[part]);
  }
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_composite);
  CeedDestroy(&ceed);
  return 0;
}
