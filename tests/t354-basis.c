/// @file
/// Test polynomial interpolation to arbitrary points in multiple dimensions
/// \test Test polynomial interpolation to arbitrary points in multiple dimensions
#include <ceed.h>
#include <math.h>
#include <stdio.h>

static CeedScalar Eval(CeedInt dim, const CeedScalar x[]) {
  CeedScalar result = 1, center = 0.1;
  for (CeedInt d = 0; d < dim; d++) {
    result *= tanh(x[d] - center);
    center += 0.1;
  }
  return result;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 1; dim <= 3; dim++) {
    CeedVector    x, x_nodes, x_points, x_point, u, v, u_point, v_point;
    CeedBasis     basis_x, basis_u;
    const CeedInt p = 9, q = 9, num_points = 4, x_dim = CeedIntPow(2, dim), p_dim = CeedIntPow(p, dim);

    CeedVectorCreate(ceed, x_dim * dim, &x);
    CeedVectorCreate(ceed, p_dim * dim, &x_nodes);
    CeedVectorCreate(ceed, num_points * dim, &x_points);
    CeedVectorCreate(ceed, dim, &x_point);
    CeedVectorCreate(ceed, p_dim, &u);
    CeedVectorCreate(ceed, num_points, &v);
    CeedVectorCreate(ceed, p_dim, &u_point);
    CeedVectorCreate(ceed, 1, &v_point);
    CeedVectorSetValue(v_point, 1.0);

    // Get nodal coordinates
    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, p, CEED_GAUSS_LOBATTO, &basis_x);
    {
      CeedScalar x_array[x_dim * dim];

      for (CeedInt d = 0; d < dim; d++) {
        for (CeedInt i = 0; i < x_dim; i++) x_array[d * x_dim + i] = (i % CeedIntPow(2, d + 1)) / CeedIntPow(2, d) ? 1 : -1;
      }
      CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    }
    CeedBasisApply(basis_x, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, x_nodes);

    // Set values of u at nodes
    {
      const CeedScalar *x_array;
      CeedScalar        u_array[p_dim];

      CeedVectorGetArrayRead(x_nodes, CEED_MEM_HOST, &x_array);
      for (CeedInt i = 0; i < p_dim; i++) {
        CeedScalar coord[dim];

        for (CeedInt d = 0; d < dim; d++) coord[d] = x_array[d * p_dim + i];
        u_array[i] = Eval(dim, coord);
      }
      CeedVectorRestoreArrayRead(x_nodes, &x_array);
      CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, (CeedScalar *)&u_array);
    }

    // Interpolate to arbitrary points
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis_u);
    {
      CeedScalar x_array[12] = {-0.33, -0.65, 0.16, 0.99, -0.65, 0.16, 0.99, -0.33, 0.16, 0.99, -0.33, -0.65};

      CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
    }
    CeedBasisApplyAtPoints(basis_u, num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);

    for (CeedInt i = 0; i < num_points; i++) {
      CeedScalar        fx = 0.0;
      CeedScalar        coord[dim];
      const CeedScalar *x_array, *u_array, *v_array, *u_point_array;

      CeedVectorGetArrayRead(x_points, CEED_MEM_HOST, &x_array);
      CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
      CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
      for (CeedInt d = 0; d < dim; d++) coord[d] = x_array[d + i * dim];
      CeedVectorSetArray(x_point, CEED_MEM_HOST, CEED_COPY_VALUES, coord);
      CeedBasisApplyAtPoints(basis_u, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, x_point, v_point, u_point);
      CeedVectorGetArrayRead(u_point, CEED_MEM_HOST, &u_point_array);
      for (CeedInt j = 0; j < p_dim; j++) fx += u_array[j] * u_point_array[j];
      if (fabs(v_array[i] - fx) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT "] %f != %f = f(%f", dim, v_array[i], fx, coord[0]);
        for (CeedInt d = 1; d < dim; d++) printf(", %f", coord[d]);
        puts(")");
        // LCOV_EXCL_STOP
      }
      CeedVectorRestoreArrayRead(u_point, &u_point_array);
      CeedVectorRestoreArrayRead(x_points, &x_array);
      CeedVectorRestoreArrayRead(u, &u_array);
      CeedVectorRestoreArrayRead(v, &v_array);
    }

    CeedVectorDestroy(&x);
    CeedVectorDestroy(&x_nodes);
    CeedVectorDestroy(&x_points);
    CeedVectorDestroy(&x_point);
    CeedVectorDestroy(&u);
    CeedVectorDestroy(&v);
    CeedVectorDestroy(&u_point);
    CeedVectorDestroy(&v_point);
    CeedBasisDestroy(&basis_x);
    CeedBasisDestroy(&basis_u);
  }

  CeedDestroy(&ceed);
  return 0;
}
