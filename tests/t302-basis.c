/// @file
/// Test Collocated Grad calculated matches basis with Lobatto points
/// \test Test Collocated Grad calculated matches basis with Lobatto points
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedInt           p = 4;
  CeedScalar        collocated_gradient_1d[(p + 2) * (p + 2)], x_2[p + 2];
  const CeedScalar *gradient_1d, *q_ref;
  CeedScalar        sum = 0.0;
  CeedBasis         basis;

  CeedInit(argv[1], &ceed);

  // Already collocated, GetCollocatedGrad will return grad_1d
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, p, CEED_GAUSS_LOBATTO, &basis);
  CeedBasisGetCollocatedGrad(basis, collocated_gradient_1d);
  CeedBasisGetGrad(basis, &gradient_1d);

  for (CeedInt i = 0; i < p; i++) {
    for (CeedInt j = 0; j < p; j++) {
      if (fabs(collocated_gradient_1d[j + p * i] - gradient_1d[j + p * i]) > 100 * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Error in collocated gradient %f != %f\n", collocated_gradient_1d[j + p * i], gradient_1d[j + p * i]);
        // LCOV_EXCL_START
      }
    }
  }
  CeedBasisDestroy(&basis);

  // Q = P, not already collocated
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, p, CEED_GAUSS, &basis);
  CeedBasisGetCollocatedGrad(basis, collocated_gradient_1d);

  CeedBasisGetQRef(basis, &q_ref);
  for (CeedInt i = 0; i < p; i++) x_2[i] = q_ref[i] * q_ref[i];

  // Verify collo_grad * x^2 = 2x for quadrature points
  for (CeedInt i = 0; i < p; i++) {
    sum = 0.0;
    for (CeedInt j = 0; j < p; j++) sum += collocated_gradient_1d[j + p * i] * x_2[j];
    if (fabs(sum - 2 * q_ref[i]) > 100 * CEED_EPSILON) printf("Error in collocated gradient %f != %f\n", sum, 2 * q_ref[i]);
  }
  CeedBasisDestroy(&basis);

  // Q = P + 2, not already collocated
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, p + 2, CEED_GAUSS, &basis);
  CeedBasisGetCollocatedGrad(basis, collocated_gradient_1d);

  CeedBasisGetQRef(basis, &q_ref);
  for (CeedInt i = 0; i < p + 2; i++) x_2[i] = q_ref[i] * q_ref[i];

  // Verify collocated_gradient * x^2 = 2x for quadrature points
  for (CeedInt i = 0; i < p + 2; i++) {
    sum = 0.0;
    for (CeedInt j = 0; j < p + 2; j++) sum += collocated_gradient_1d[j + (p + 2) * i] * x_2[j];
    if (fabs(sum - 2 * q_ref[i]) > 100 * CEED_EPSILON) printf("Error in collocated gradient %f != %f\n", sum, 2 * q_ref[i]);
  }
  CeedBasisDestroy(&basis);

  CeedDestroy(&ceed);
  return 0;
}
