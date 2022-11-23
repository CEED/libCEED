/// @file
/// Test creation, action, and destruction for mass matrix operator with passive inputs and outputs
/// \test Test creation, action, and destruction for mass matrix operator with passive inputs and outputs
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t500-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_qd_i;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, X, U, V;
  const CeedScalar   *hv;
  CeedInt             num_elem = 15, P = 5, Q = 8;
  CeedInt             Nx = num_elem + 1, Nu = num_elem * (P - 1) + 1;
  CeedInt             ind_x[num_elem * 2], ind_u[num_elem * P];
  CeedScalar          x[Nx];
  CeedScalar          sum;

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, Nu, &U);
  CeedVectorSetValue(U, 1.0);

  CeedVectorCreate(ceed, Nu, &V);

  for (CeedInt i = 0; i < Nx; i++) x[i] = (CeedScalar)i / (Nx - 1);
  CeedVectorCreate(ceed, Nx, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  CeedVectorCreate(ceed, num_elem * Q, &q_data);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, Nx, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_x);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < P; j++) {
      ind_u[P * i + j] = i * (P - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, P, 1, 1, Nu, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restr_u);
  CeedInt strides_qd[3] = {1, Q, Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q, 1, Q * num_elem, strides_qd, &elem_restr_qd_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);

  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restr_x, basis_x, X);
  CeedOperatorSetField(op_setup, "rho", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);

  CeedOperatorSetField(op_mass, "rho", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restr_u, basis_u, U);
  CeedOperatorSetField(op_mass, "v", elem_restr_u, basis_u, V);

  // Note - It is atypical to use only passive fields; this test is intended
  //   as a test for all passive input modes rather than as an example.
  CeedOperatorApply(op_setup, CEED_VECTOR_NONE, CEED_VECTOR_NONE, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_mass, CEED_VECTOR_NONE, CEED_VECTOR_NONE, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i = 0; i < Nu; i++) sum += hv[i];
  if (fabs(sum - 1.) > 1000. * CEED_EPSILON) printf("Computed Area: %f != True Area: 1.0\n", sum);
  CeedVectorRestoreArrayRead(V, &hv);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedElemRestrictionDestroy(&elem_restr_u);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_qd_i);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&q_data);
  CeedDestroy(&ceed);
  return 0;
}
