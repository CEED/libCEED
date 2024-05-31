/// @file
/// Test polynomial interpolation transpose from arbitrary points in 1D
/// \test Test polynomial interpolation transpose from arbitrary points in 1D
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#define ALEN(a) (sizeof(a) / sizeof((a)[0]))

static CeedScalar Eval(CeedScalar x, CeedInt n, const CeedScalar *c) {
  CeedScalar y = c[n - 1];
  for (CeedInt i = n - 2; i >= 0; i--) y = y * x + c[i];
  return y;
}

int main(int argc, char **argv) {
  Ceed             ceed;
  CeedVector       x, x_nodes, x_points, x_point, u, v, u_point, v_point;
  CeedBasis        basis_x, basis_u;
  const CeedInt    p = 5, q = 5, num_points = 4;
  const CeedScalar c[4] = {1, 2, 3, 4};  // 1 + 2x + 3x^2 + ...

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 2, &x);
  CeedVectorCreate(ceed, p, &x_nodes);
  CeedVectorCreate(ceed, num_points, &x_points);
  CeedVectorCreate(ceed, 1, &x_point);
  CeedVectorCreate(ceed, p, &u);
  CeedVectorCreate(ceed, num_points, &v);
  CeedVectorCreate(ceed, p, &u_point);
  CeedVectorCreate(ceed, 1, &v_point);
  CeedVectorSetValue(v_point, 1.0);

  // Get nodal coordinates
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, p, CEED_GAUSS_LOBATTO, &basis_x);
  {
    CeedScalar x_array[2];

    for (CeedInt i = 0; i < 2; i++) x_array[i] = CeedIntPow(-1, i + 1);
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedBasisApply(basis_x, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, x_nodes);

  // Set values of u at nodes
  {
    const CeedScalar *x_array;
    CeedScalar        u_array[p];

    CeedVectorGetArrayRead(x_nodes, CEED_MEM_HOST, &x_array);
    for (CeedInt i = 0; i < p; i++) u_array[i] = Eval(x_array[i], ALEN(c), c);
    CeedVectorRestoreArrayRead(x_nodes, &x_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, (CeedScalar *)&u_array);
  }

  // Interpolate to arbitrary points
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, q, CEED_GAUSS, &basis_u);
  {
    CeedScalar x_array[4] = {-0.33, -0.65, 0.16, 0.99};

    CeedVectorSetArray(x_points, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedBasisApplyAtPoints(basis_u, 1, &num_points, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x_points, u, v);

  for (CeedInt i = 0; i < num_points; i++) {
    const CeedInt     num_point[1] = {1};
    CeedScalar        fx           = 0.0;
    const CeedScalar *x_array, *u_array, *v_array, *u_point_array;

    CeedVectorGetArrayRead(x_points, CEED_MEM_HOST, &x_array);
    CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    CeedVectorSetValue(x_point, x_array[i]);
    CeedBasisApplyAtPoints(basis_u, 1, num_point, CEED_TRANSPOSE, CEED_EVAL_INTERP, x_point, v_point, u_point);
    CeedVectorGetArrayRead(u_point, CEED_MEM_HOST, &u_point_array);
    for (CeedInt j = 0; j < p; j++) fx += u_array[j] * u_point_array[j];
    if (fabs(v_array[i] - fx) > 100. * CEED_EPSILON) printf("%f != %f = f(%f)\n", v_array[i], fx, x_array[i]);
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
  CeedDestroy(&ceed);
  return 0;
}
