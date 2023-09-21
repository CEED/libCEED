/// @file
/// Test assembly of H(div) mass matrix operator diagonal
/// \test Test assembly of H(div) mass matrix operator diagonal
#include "t570-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t330-basis.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_mass;
  CeedOperator        op_mass;
  CeedVector          x, assembled, u, v;
  CeedInt             dim = 2, p = 8, q = 3, px = 2;
  CeedInt             n_x = 1, n_y = 1;  // Currently only implemented for single element
  CeedInt             row, column, offset;
  CeedInt             num_elem = n_x * n_y, num_faces = (n_x + 1) * n_y + (n_y + 1) * n_x;
  CeedInt             num_dofs_x = (n_x + 1) * (n_y + 1), num_dofs_u = num_faces * 2, num_qpts = q * q;
  CeedInt             ind_x[num_elem * px * px], ind_u[num_elem * p];
  bool                orient_u[num_elem * p];
  CeedScalar          assembled_true[num_dofs_u];
  CeedScalar          q_ref[dim * num_qpts], q_weight[num_qpts];
  CeedScalar          interp[dim * p * num_qpts], div[p * num_qpts];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs_x, &x);
  {
    CeedScalar x_array[dim * num_dofs_x];

    for (CeedInt i = 0; i < n_x + 1; i++) {
      for (CeedInt j = 0; j < n_y + 1; j++) {
        x_array[i + j * (n_x + 1) + 0 * num_dofs_x] = i / (CeedScalar)n_x;
        x_array[i + j * (n_x + 1) + 1 * num_dofs_x] = j / (CeedScalar)n_y;
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_dofs_u, &u);
  CeedVectorCreate(ceed, num_dofs_u, &v);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    column = i % n_x;
    row    = i / n_x;
    offset = column * (px - 1) + row * (n_x + 1) * (px - 1);

    for (CeedInt j = 0; j < px; j++) {
      for (CeedInt k = 0; k < px; k++) {
        ind_x[px * (px * i + k) + j] = offset + k * (n_x + 1) + j;
      }
    }
  }
  bool    orient_u_local[8] = {false, false, false, false, true, true, true, true};
  CeedInt ind_u_local[8]    = {0, 1, 6, 7, 2, 3, 4, 5};
  for (CeedInt j = 0; j < n_y; j++) {
    for (CeedInt i = 0; i < n_x; i++) {
      for (CeedInt k = 0; k < p; k++) {
        ind_u[k]    = ind_u_local[k];
        orient_u[k] = orient_u_local[k];
      }
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, px * px, dim, num_dofs_x, dim * num_dofs_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
  CeedElemRestrictionCreateOriented(ceed, num_elem, p, 1, 1, num_dofs_u, CEED_MEM_HOST, CEED_COPY_VALUES, ind_u, orient_u, &elem_restriction_u);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, px, q, CEED_GAUSS, &basis_x);

  BuildHdivQuadrilateral(q, q_ref, q_weight, interp, div, CEED_GAUSS);
  CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, 1, p, num_qpts, interp, div, q_ref, q_weight, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_mass, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_mass, "u", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", dim, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_mass, "dx", elem_restriction_x, basis_x, x);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Assemble diagonal
  CeedVectorCreate(ceed, num_dofs_u, &assembled);
  CeedOperatorLinearAssembleDiagonal(op_mass, assembled, CEED_REQUEST_IMMEDIATE);

  // Manually assemble diagonal
  CeedVectorSetValue(u, 0.0);
  for (int i = 0; i < num_dofs_u; i++) {
    CeedScalar       *u_array;
    const CeedScalar *v_array;

    // Set input
    CeedVectorGetArray(u, CEED_MEM_HOST, &u_array);
    u_array[i] = 1.0;
    if (i) u_array[i - 1] = 0.0;
    CeedVectorRestoreArray(u, &u_array);

    // Compute diag entry for DoF i
    CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

    // Retrieve entry
    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    assembled_true[i] = v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Check output
  {
    const CeedScalar *assembled_array;

    CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
    for (int i = 0; i < num_dofs_u; i++) {
      if (fabs(assembled_array[i] - assembled_true[i]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT "] Error in assembly: %f != %f\n", i, assembled_array[i], assembled_true[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(assembled, &assembled_array);
  }

  // Cleanup
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&assembled);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_mass);
  CeedDestroy(&ceed);
  return 0;
}
