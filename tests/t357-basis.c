/// @file
/// Test gradient transpose in multiple dimensions at arbitrary points
/// \test Test gradient transpose in multiple dimensions at arbitrary points
#include <ceed.h>
#include <math.h>
#include <stdio.h>

static CeedScalar Eval(CeedInt dim, const CeedScalar x[]) {
  CeedScalar result = tanh(x[0] + 0.1);
  if (dim > 1) result += atan(x[1] + 0.2);
  if (dim > 2) result += exp(-(x[2] + 0.3) * (x[2] + 0.3));
  return result;
}

static CeedScalar GetTolerance(CeedScalarType scalar_type, int dim) {
  CeedScalar tol;
  if (scalar_type == CEED_SCALAR_FP32) {
    if (dim == 3) tol = 0.005;
    else tol = 1.e-4;
  } else {
    tol = 1.e-11;
  }
  return tol;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 1; dim <= 3; dim++) {
    CeedVector    x, x_nodes, x_points, u, u_points, v, ones;
    CeedBasis     basis_x, basis_u;
    const CeedInt p = 9, q = 9, num_points = 4, x_dim = CeedIntPow(2, dim), p_dim = CeedIntPow(p, dim);
    CeedScalar    sum_1 = 0, sum_2 = 0;

    CeedVectorCreate(ceed, x_dim * dim, &x);
    CeedVectorCreate(ceed, p_dim * dim, &x_nodes);
    CeedVectorCreate(ceed, num_points * dim, &x_points);
    CeedVectorCreate(ceed, p_dim, &u);
    CeedVectorCreate(ceed, num_points * dim, &u_points);
    CeedVectorCreate(ceed, p_dim, &v);
    CeedVectorCreate(ceed, num_points * dim, &ones);

    CeedVectorSetValue(ones, 1);
    CeedVectorSetValue(v, 0);

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

    // Calculate G u at arbitrary points, G' * 1 at dofs
    CeedBasisApplyAtPoints(basis_u, num_points, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, x_points, u, u_points);
    CeedBasisApplyAtPoints(basis_u, num_points, CEED_TRANSPOSE, CEED_EVAL_GRAD, x_points, ones, v);
    {
      const CeedScalar *u_array, *v_array, *u_points_array;

      CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
      CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
      CeedVectorGetArrayRead(u_points, CEED_MEM_HOST, &u_points_array);
      for (CeedInt i = 0; i < p_dim; i++) sum_1 += v_array[i] * u_array[i];
      for (CeedInt i = 0; i < num_points * dim; i++) sum_2 += u_points_array[i];
      CeedVectorRestoreArrayRead(u, &u_array);
      CeedVectorRestoreArrayRead(v, &v_array);
      CeedVectorRestoreArrayRead(u_points, &u_points_array);
    }
    CeedScalar tol = GetTolerance(CEED_SCALAR_TYPE, dim);
    if (fabs(sum_1 - sum_2) > tol) printf("[%" CeedInt_FMT "] %f != %f\n", dim, sum_1, sum_2);

    CeedVectorDestroy(&x);
    CeedVectorDestroy(&x_nodes);
    CeedVectorDestroy(&x_points);
    CeedVectorDestroy(&u);
    CeedVectorDestroy(&u_points);
    CeedVectorDestroy(&ones);
    CeedVectorDestroy(&v);
    CeedBasisDestroy(&basis_x);
    CeedBasisDestroy(&basis_u);
  }
  CeedDestroy(&ceed);
  return 0;
}
