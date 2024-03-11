/// @file
/// Bug reproducer for memcheck backends at points
/// \test Test 1D mass matrix operator at points with heterogeneous points per element
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t500-operator.h"

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedInt    num_elem = 3, dim = 1, p = 3, q = 5;
  CeedInt    num_nodes_x = num_elem + 1, num_nodes_u = num_elem * (p - 1) + 1, num_points_per_elem = 4, num_points = num_elem * num_points_per_elem;
  CeedInt    ind_x[num_elem * 2], ind_u[num_elem * p], ind_x_points[num_elem + 1 + num_points];
  CeedScalar x_array_mesh[num_nodes_x], x_array_points[num_points];
  CeedVector x_points = NULL, x_elem = NULL, q_data = NULL, u = NULL, v = NULL;
  CeedElemRestriction elem_restriction_x_points, elem_restriction_q_data, elem_restriction_x, elem_restriction_u;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass;
  bool                is_at_points;

  CeedInit(argv[1], &ceed);

  // Mesh coordinates
  for (CeedInt i = 0; i < num_nodes_x; i++) x_array_mesh[i] = (CeedScalar)i / (num_nodes_x - 1);
  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_nodes_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
  CeedVectorCreate(ceed, num_nodes_x, &x_elem);
  CeedVectorSetArray(x_elem, CEED_MEM_HOST, CEED_USE_POINTER, x_array_mesh);

  // U mesh
  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p; j++) {
      ind_u[p * i + j] = i * (p - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p, 1, 1, num_nodes_u, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restriction_u);

  // Point reference coordinates
  {
    CeedScalar weight_tmp[num_points_per_elem + 1];
    CeedInt    current_index = 0;

    // Use num_points_per_elem + 1 to test non-uniform quadrature
    CeedGaussQuadrature(num_points_per_elem + 1, x_array_points, weight_tmp);
    ind_x_points[0] = num_elem + 1;
    for (CeedInt p = 0; p < num_points_per_elem + 1; p++, current_index++) {
      ind_x_points[num_elem + 1 + current_index] = current_index;
    }
    // Use num_points_per_elem for middle elements
    for (CeedInt e = 1; e < num_elem - 1; e++) {
      CeedGaussQuadrature(num_points_per_elem, &x_array_points[current_index], weight_tmp);
      ind_x_points[e] = num_elem + 1 + current_index;
      for (CeedInt p = 0; p < num_points_per_elem; p++, current_index++) {
        ind_x_points[num_elem + 1 + current_index] = current_index;
      }
    }
    // Use num_points_per_elem - 1 to test non-uniform quadrature
    CeedGaussQuadrature(num_points_per_elem - 1, &x_array_points[current_index], weight_tmp);
    ind_x_points[num_elem - 1] = num_elem + 1 + current_index;
    for (CeedInt p = 0; p < num_points_per_elem - 1; p++, current_index++) {
      ind_x_points[num_elem + 1 + current_index] = current_index;
    }
    ind_x_points[num_elem] = num_elem + 1 + current_index;

    CeedVectorCreate(ceed, num_elem * num_points_per_elem, &x_points);
    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_USE_POINTER, x_array_points);
    CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, 1, num_points, CEED_MEM_HOST, CEED_COPY_VALUES, ind_x_points,
                                      &elem_restriction_x_points);
    CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, 1, num_points, CEED_MEM_HOST, CEED_COPY_VALUES, ind_x_points,
                                      &elem_restriction_q_data);

    // Q data
    CeedVectorCreate(ceed, num_points, &q_data);
  }

  // Basis creation
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis_u);

  // Setup geometric scaling
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "x", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedOperatorCreateAtPoints(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "x", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "rho", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  CeedOperatorAtPointsSetPoints(op_setup, elem_restriction_x_points, x_points);

  CeedOperatorApply(op_setup, x_elem, q_data, CEED_REQUEST_IMMEDIATE);

  // Mass operator
  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  CeedOperatorCreateAtPoints(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorAtPointsSetPoints(op_mass, elem_restriction_x_points, x_points);

  CeedOperatorIsAtPoints(op_mass, &is_at_points);
  if (!is_at_points) printf("Error: Operator should be at points\n");

  CeedVectorCreate(ceed, num_nodes_u, &u);
  CeedVectorSetValue(u, 0.0);
  CeedVectorCreate(ceed, num_nodes_u, &v);

  // Assemble QFunction
  CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_nodes_u; i++) {
      if (fabs(v_array[i]) > 1e-14) printf("[%" CeedInt_FMT "] v %g != 0.0\n", i, v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Cleanup
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&x_elem);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_x_points);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedDestroy(&ceed);
  return 0;
}
