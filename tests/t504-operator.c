/// @file
/// Test viewing of mass matrix operator
/// \test Test viewing of mass matrix operator
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t500-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_qd_i;
  CeedBasis           bx, bu;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data;
  CeedInt             num_elem = 15, P = 5, Q = 8;
  CeedInt             num_nodes_x = num_elem + 1, num_nodes_u = num_elem * (P - 1) + 1;
  CeedInt             ind_x[num_elem * 2], ind_u[num_elem * P];

  CeedInit(argv[1], &ceed);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  // Restrictions
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_nodes_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_x);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < P; j++) {
      ind_u[P * i + j] = 2 * (i * (P - 1) + j);
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, P, 2, 1, 2 * num_nodes_u, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restr_u);
  CeedInt strides_qd[3] = {1, Q, Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q, 1, Q * num_elem, strides_qd, &elem_restr_qd_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 2, P, Q, CEED_GAUSS, &bu);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1 * 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 2, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 2, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);

  CeedVectorCreate(ceed, num_elem * Q, &q_data);

  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, bx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restr_x, bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass, "rho", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restr_u, bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restr_u, bu, CEED_VECTOR_ACTIVE);

  CeedOperatorView(op_setup, stdout);
  CeedOperatorView(op_mass, stdout);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedElemRestrictionDestroy(&elem_restr_u);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_qd_i);

  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&q_data);
  CeedDestroy(&ceed);
  return 0;
}
