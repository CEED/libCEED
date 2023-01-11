/// @file
/// Test Collocated Grad calculated matches basis with Lobatto points
/// \test Test Collocated Grad calculated matches basis with Lobatto points
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedInt           P = 4;
  CeedScalar        collo_grad_1d[(P + 2) * (P + 2)], x_2[P + 2];
  const CeedScalar *grad_1d, *q_ref;
  CeedScalar        sum = 0.0;
  CeedBasis         b;

  CeedInit(argv[1], &ceed);

  // Already collocated, GetCollocatedGrad will return grad_1d
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, P, CEED_GAUSS_LOBATTO, &b);
  CeedBasisGetCollocatedGrad(b, collo_grad_1d);
  CeedBasisGetGrad(b, &grad_1d);

  for (CeedInt i = 0; i < P; i++) {
    for (CeedInt j = 0; j < P; j++) {
      if (fabs(collo_grad_1d[j + P * i] - grad_1d[j + P * i]) > 100 * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Error in collocated gradient %f != %f\n", collo_grad_1d[j + P * i], grad_1d[j + P * i]);
        // LCOV_EXCL_START
      }
    }
  }
  CeedBasisDestroy(&b);

  // Q = P, not already collocated
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, P, CEED_GAUSS, &b);
  CeedBasisGetCollocatedGrad(b, collo_grad_1d);

  CeedBasisGetQRef(b, &q_ref);
  for (CeedInt i = 0; i < P; i++) x_2[i] = q_ref[i] * q_ref[i];

  // Verify collo_grad * x^2 = 2x for quadrature points
  for (CeedInt i = 0; i < P; i++) {
    sum = 0.0;
    for (CeedInt j = 0; j < P; j++) sum += collo_grad_1d[j + P * i] * x_2[j];
    if (fabs(sum - 2 * q_ref[i]) > 100 * CEED_EPSILON) printf("Error in collocated gradient %f != %f\n", sum, 2 * q_ref[i]);
  }
  CeedBasisDestroy(&b);

  // Q = P + 2, not already collocated
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, P + 2, CEED_GAUSS, &b);
  CeedBasisGetCollocatedGrad(b, collo_grad_1d);

  CeedBasisGetQRef(b, &q_ref);
  for (CeedInt i = 0; i < P + 2; i++) x_2[i] = q_ref[i] * q_ref[i];

  // Verify collo_grad * x^2 = 2x for quadrature points
  for (CeedInt i = 0; i < P + 2; i++) {
    sum = 0.0;
    for (CeedInt j = 0; j < P + 2; j++) sum += collo_grad_1d[j + (P + 2) * i] * x_2[j];
    if (fabs(sum - 2 * q_ref[i]) > 100 * CEED_EPSILON) printf("Error in collocated gradient %f != %f\n", sum, 2 * q_ref[i]);
  }
  CeedBasisDestroy(&b);

  CeedDestroy(&ceed);
  return 0;
}
