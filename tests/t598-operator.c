/// @file
/// Test creation, action, and destruction for mass matrix operator AtPoints
/// \test Test creation, action, and destruction for mass matrix operator AtPoints
#include "t591-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedInt             num_elem_1d = 3, num_elem = num_elem_1d * num_elem_1d, dim = 2, p_coarse = 2, p_fine = 3, q = 5;
  CeedInt             num_points_per_elem = 4, num_points = num_elem * num_points_per_elem;
  CeedInt             num_nodes_coarse = (num_elem_1d * (p_coarse - 1) + 1) * (num_elem_1d * (p_coarse - 1) + 1);
  CeedInt             num_nodes_fine   = (num_elem_1d * (p_fine - 1) + 1) * (num_elem_1d * (p_fine - 1) + 1);
  CeedVector          x_points, x_elem, q_data, u_coarse, u_fine, v_coarse, v_fine, p_mult_fine;
  CeedElemRestriction elem_restriction_x_points, elem_restriction_q_data, elem_restriction_x, elem_restriction_u_coarse, elem_restriction_u_fine;
  CeedBasis           basis_x, basis_u_coarse, basis_u_fine;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass_coarse, op_mass_fine, op_prolong, op_restrict;

  CeedInit(argv[1], &ceed);

  // Point reference coordinates
  CeedVectorCreate(ceed, dim * num_points, &x_points);
  {
    CeedScalar x_array[dim * num_points];

    for (CeedInt e = 0; e < num_elem; e++) {
      for (CeedInt d = 0; d < dim; d++) {
        x_array[num_points_per_elem * (e * dim + d) + 0] = 0.25;
        x_array[num_points_per_elem * (e * dim + d) + 1] = d == 0 ? -0.25 : 0.25;
        x_array[num_points_per_elem * (e * dim + d) + 2] = d == 0 ? 0.25 : -0.25;
        x_array[num_points_per_elem * (e * dim + d) + 3] = 0.25;
      }
    }
    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  {
    CeedInt ind_x[num_elem + 1 + num_points];

    for (CeedInt i = 0; i <= num_elem; i++) ind_x[i] = num_elem + 1 + i * num_points_per_elem;
    for (CeedInt i = 0; i < num_points; i++) ind_x[num_elem + 1 + i] = i;
    CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, dim, num_points * dim, CEED_MEM_HOST, CEED_COPY_VALUES, ind_x,
                                      &elem_restriction_x_points);
    CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, 1, num_points, CEED_MEM_HOST, CEED_COPY_VALUES, ind_x, &elem_restriction_q_data);
  }

  // Q data
  CeedVectorCreate(ceed, num_points, &q_data);

  // Cell coordinates
  {
    CeedInt p = 2, num_nodes = (num_elem_1d * (p - 1) + 1) * (num_elem_1d * (p - 1) + 1);
    CeedInt ind_x[num_elem * p * p];

    for (CeedInt e = 0; e < num_elem; e++) {
      CeedInt elem_xy[2] = {1, 1}, n_d[2] = {0, 0};

      for (CeedInt d = 0; d < dim; d++) n_d[d] = num_elem_1d * (p - 1) + 1;
      {
        CeedInt r_e = e;

        for (CeedInt d = 0; d < dim; d++) {
          elem_xy[d] = r_e % num_elem_1d;
          r_e /= num_elem_1d;
        }
      }
      CeedInt num_nodes_in_elem = p * p, *elem_nodes = ind_x + e * num_nodes_in_elem;

      for (CeedInt n = 0; n < num_nodes_in_elem; n++) {
        CeedInt g_node = 0, g_node_stride = 1, r_node = n;

        for (CeedInt d = 0; d < dim; d++) {
          g_node += (elem_xy[d] * (p - 1) + r_node % p) * g_node_stride;
          g_node_stride *= n_d[d];
          r_node /= p;
        }
        elem_nodes[n] = p * g_node;
      }
    }
    CeedElemRestrictionCreate(ceed, num_elem, p * p, dim, 1, dim * num_nodes, CEED_MEM_HOST, CEED_COPY_VALUES, ind_x, &elem_restriction_x);
    CeedVectorCreate(ceed, dim * num_nodes, &x_elem);
    {
      CeedScalar x_array[dim * num_nodes];

      for (CeedInt i = 0; i <= num_elem_1d; i++) {
        for (CeedInt j = 0; j <= num_elem_1d; j++) {
          x_array[(i * (num_elem_1d + 1) + j) * dim + 0] = j;
          x_array[(i * (num_elem_1d + 1) + j) * dim + 1] = i;
        }
      }
      CeedVectorSetArray(x_elem, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    }
  }

  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, q, CEED_GAUSS, &basis_x);

  // Cell solution
  {
    CeedInt ind_u[num_elem * p_coarse * p_coarse];

    for (CeedInt e = 0; e < num_elem; e++) {
      CeedInt elem_xy[2] = {1, 1}, n_d[2] = {0, 0};

      for (CeedInt d = 0; d < dim; d++) n_d[d] = num_elem_1d * (p_coarse - 1) + 1;
      {
        CeedInt r_e = e;

        for (CeedInt d = 0; d < dim; d++) {
          elem_xy[d] = r_e % num_elem_1d;
          r_e /= num_elem_1d;
        }
      }
      CeedInt num_nodes_in_elem = p_coarse * p_coarse, *elem_nodes = ind_u + e * num_nodes_in_elem;

      for (CeedInt n = 0; n < num_nodes_in_elem; n++) {
        CeedInt g_node = 0, g_node_stride = 1, r_node = n;

        for (CeedInt d = 0; d < dim; d++) {
          g_node += (elem_xy[d] * (p_coarse - 1) + r_node % p_coarse) * g_node_stride;
          g_node_stride *= n_d[d];
          r_node /= p_coarse;
        }
        elem_nodes[n] = g_node;
      }
    }
    CeedElemRestrictionCreate(ceed, num_elem, p_coarse * p_coarse, 1, 1, num_nodes_coarse, CEED_MEM_HOST, CEED_COPY_VALUES, ind_u,
                              &elem_restriction_u_coarse);
  }
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p_coarse, q, CEED_GAUSS, &basis_u_coarse);
  {
    CeedInt ind_u[num_elem * p_fine * p_fine];

    for (CeedInt e = 0; e < num_elem; e++) {
      CeedInt elem_xy[2] = {1, 1}, n_d[2] = {0, 0};

      for (CeedInt d = 0; d < dim; d++) n_d[d] = num_elem_1d * (p_fine - 1) + 1;
      {
        CeedInt r_e = e;

        for (CeedInt d = 0; d < dim; d++) {
          elem_xy[d] = r_e % num_elem_1d;
          r_e /= num_elem_1d;
        }
      }
      CeedInt num_nodes_in_elem = p_fine * p_fine, *elem_nodes = ind_u + e * num_nodes_in_elem;

      for (CeedInt n = 0; n < num_nodes_in_elem; n++) {
        CeedInt g_node = 0, g_node_stride = 1, r_node = n;

        for (CeedInt d = 0; d < dim; d++) {
          g_node += (elem_xy[d] * (p_fine - 1) + r_node % p_fine) * g_node_stride;
          g_node_stride *= n_d[d];
          r_node /= p_fine;
        }
        elem_nodes[n] = g_node;
      }
    }
    CeedElemRestrictionCreate(ceed, num_elem, p_fine * p_fine, 1, 1, num_nodes_fine, CEED_MEM_HOST, CEED_COPY_VALUES, ind_u,
                              &elem_restriction_u_fine);
  }
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p_fine, q, CEED_GAUSS, &basis_u_fine);

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

  CeedOperatorCreateAtPoints(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_fine);
  CeedOperatorSetField(op_mass_fine, "u", elem_restriction_u_fine, basis_u_fine, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_fine, "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass_fine, "v", elem_restriction_u_fine, basis_u_fine, CEED_VECTOR_ACTIVE);
  CeedOperatorAtPointsSetPoints(op_mass_fine, elem_restriction_x_points, x_points);

  CeedVectorCreate(ceed, num_nodes_fine, &u_fine);
  CeedVectorCreate(ceed, num_nodes_fine, &v_fine);
  CeedVectorCreate(ceed, num_nodes_fine, &p_mult_fine);
  CeedVectorCreate(ceed, num_nodes_coarse, &u_coarse);
  CeedVectorCreate(ceed, num_nodes_coarse, &v_coarse);

  // Create multigrid level
  CeedVectorSetValue(p_mult_fine, 1.0);
  CeedOperatorMultigridLevelCreate(op_mass_fine, p_mult_fine, elem_restriction_u_coarse, basis_u_coarse, &op_mass_coarse, &op_prolong, &op_restrict);

  // Coarse problem
  CeedVectorSetValue(u_coarse, 1.0);
  CeedOperatorApply(op_mass_coarse, u_coarse, v_coarse, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        sum = 0.;

    CeedVectorGetArrayRead(v_coarse, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_nodes_coarse; i++) {
      sum += v_array[i];
    }
    CeedVectorRestoreArrayRead(v_coarse, &v_array);
    if (fabs(sum - num_elem) > 1000. * CEED_EPSILON) printf("Computed Area Coarse Grid: %f != True Area: 2.0\n", sum);
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
    for (CeedInt i = 0; i < num_nodes_fine; i++) {
      sum += v_array[i];
    }
    CeedVectorRestoreArrayRead(v_fine, &v_array);

    if (fabs(sum - num_elem) > 1000. * CEED_EPSILON) printf("Computed Area Fine Grid: %f != True Area: 2.0\n", sum);
  }
  // Restrict state to coarse grid
  CeedOperatorApply(op_restrict, v_fine, v_coarse, CEED_REQUEST_IMMEDIATE);

  // Check output
  {
    const CeedScalar *v_array;
    CeedScalar        sum = 0.;

    CeedVectorGetArrayRead(v_coarse, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_nodes_coarse; i++) {
      sum += v_array[i];
    }
    CeedVectorRestoreArrayRead(v_coarse, &v_array);
    if (fabs(sum - num_elem) > 1000. * CEED_EPSILON) printf("Computed Area Coarse Grid: %f != True Area: 2.0\n", sum);
  }

  CeedVectorDestroy(&x_points);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&x_elem);
  CeedVectorDestroy(&u_coarse);
  CeedVectorDestroy(&u_fine);
  CeedVectorDestroy(&v_fine);
  CeedVectorDestroy(&v_coarse);
  CeedVectorDestroy(&p_mult_fine);
  CeedElemRestrictionDestroy(&elem_restriction_x_points);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_u_coarse);
  CeedElemRestrictionDestroy(&elem_restriction_u_fine);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u_coarse);
  CeedBasisDestroy(&basis_u_fine);
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
