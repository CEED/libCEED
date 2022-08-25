/// @file
/// Test assembly of mass matrix operator (multi-component) see t537
/// \test Test assembly of mass matrix operator (multi-component)
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t537-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_qd_i;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  CeedVector          q_data, X, U, V;
  CeedInt             P = 3, Q = 4, dim = 2, num_comp = 2;
  CeedInt             n_x = 1, n_y = 1;
  CeedInt             num_elem = n_x * n_y;
  CeedInt             num_dofs = (n_x * 2 + 1) * (n_y * 2 + 1), num_qpts = num_elem * Q * Q;
  CeedInt             ind_x[num_elem * P * P];
  CeedScalar          assembled[num_comp * num_comp * num_dofs * num_dofs];
  CeedScalar          x[dim * num_dofs], assembled_true[num_comp * num_comp * num_dofs * num_dofs];
  CeedScalar         *u;
  const CeedScalar   *v;

  CeedInit(argv[1], &ceed);

  // DoF Coordinates
  for (CeedInt i = 0; i < n_x * 2 + 1; i++) {
    for (CeedInt j = 0; j < n_y * 2 + 1; j++) {
      x[i + j * (n_x * 2 + 1) + 0 * num_dofs] = (CeedScalar)i / (2 * n_x);
      x[i + j * (n_x * 2 + 1) + 1 * num_dofs] = (CeedScalar)j / (2 * n_y);
    }
  }
  CeedVectorCreate(ceed, dim * num_dofs, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vector
  CeedVectorCreate(ceed, num_qpts, &q_data);

  // Element Setup
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;
    col    = i % n_x;
    row    = i / n_x;
    offset = col * (P - 1) + row * (n_x * 2 + 1) * (P - 1);
    for (CeedInt j = 0; j < P; j++) {
      for (CeedInt k = 0; k < P; k++) ind_x[P * (P * i + k) + j] = offset + k * (n_x * 2 + 1) + j;
    }
  }

  // Restrictions
  CeedElemRestrictionCreate(ceed, num_elem, P * P, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_x);
  CeedElemRestrictionCreate(ceed, num_elem, P * P, num_comp, num_dofs, num_comp * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_u);
  CeedInt strides_qd[3] = {1, Q * Q, Q * Q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q * Q, 1, num_qpts, strides_qd, &elem_restr_qd_i);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, P, Q, CEED_GAUSS, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", num_comp, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", num_comp, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", elem_restr_qd_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "rho", elem_restr_qd_i, CEED_BASIS_COLLOCATED, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, X, q_data, CEED_REQUEST_IMMEDIATE);

  // Fuly assemble operator
  for (CeedInt k = 0; k < num_comp * num_comp * num_dofs * num_dofs; k++) {
    assembled[k]      = 0.0;
    assembled_true[k] = 0.0;
  }
  CeedSize   nentries;
  CeedInt   *rows;
  CeedInt   *cols;
  CeedVector values;
  CeedOperatorLinearAssembleSymbolic(op_mass, &nentries, &rows, &cols);
  CeedVectorCreate(ceed, nentries, &values);
  CeedOperatorLinearAssemble(op_mass, values);
  const CeedScalar *vals;
  CeedVectorGetArrayRead(values, CEED_MEM_HOST, &vals);
  for (CeedInt k = 0; k < nentries; ++k) assembled[rows[k] * num_comp * num_dofs + cols[k]] += vals[k];
  CeedVectorRestoreArrayRead(values, &vals);

  // Manually assemble operator
  CeedVectorCreate(ceed, num_comp * num_dofs, &U);
  CeedVectorSetValue(U, 0.0);
  CeedVectorCreate(ceed, num_comp * num_dofs, &V);
  CeedInt indOld = -1;
  for (CeedInt j = 0; j < num_dofs * num_comp; j++) {
    // Set input
    CeedVectorGetArray(U, CEED_MEM_HOST, &u);
    CeedInt ind = j;
    u[ind]      = 1.0;
    if (ind > 0) u[indOld] = 0.0;
    indOld = ind;
    CeedVectorRestoreArray(U, &u);

    // Compute effect of DoF j
    CeedOperatorApply(op_mass, U, V, CEED_REQUEST_IMMEDIATE);

    CeedVectorGetArrayRead(V, CEED_MEM_HOST, &v);
    for (CeedInt k = 0; k < num_dofs * num_comp; k++) assembled_true[j * num_dofs * num_comp + k] = v[k];
    CeedVectorRestoreArrayRead(V, &v);
  }

  // Check output
  for (CeedInt i = 0; i < num_comp * num_dofs; i++) {
    for (CeedInt j = 0; j < num_comp * num_dofs; j++) {
      if (fabs(assembled[j * num_dofs * num_comp + i] - assembled_true[j * num_dofs * num_comp + i]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT ", %" CeedInt_FMT "] Error in assembly: %f != %f\n", i, j, assembled[j * num_dofs * num_comp + i],
               assembled_true[j * num_dofs * num_comp + i]);
        // LCOV_EXCL_STOP
      }
    }
  }

  // Cleanup
  free(rows);
  free(cols);
  CeedVectorDestroy(&values);
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
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
