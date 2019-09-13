/// @file
/// Test integration with a 2D Simplex non-tensor H1 basis
/// \test Test integration with a 2D Simplex non-tensor H1 basis
#include <ceed.h>
#include <math.h>
#include "t320-basis.h"

double feval(double x1, double x2) {
  return x1*x1 + x2*x2 + x1*x2 + 1;
}

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector In, Out, Weights;
  const CeedInt P = 6, Q = 4, dim = 2;
  CeedBasis b;
  CeedScalar qref[dim*Q], qweight[Q];
  CeedScalar interp[P*Q], grad[dim*P*Q];
  CeedScalar xr[] = {0., 0.5, 1., 0., 0.5, 0., 0., 0., 0., 0.5, 0.5, 1.};
  const CeedScalar *out, *weights;
  CeedScalar in[P], sum;

  buildmats(qref, qweight, interp, grad);

  CeedInit(argv[1], &ceed);

  CeedBasisCreateH1(ceed, CEED_TRIANGLE, 1, P, Q, interp, grad, qref,
                    qweight, &b);

  // Interpolate function to quadrature points
  for (int i=0; i<P; i++)
    in[i] = feval(xr[0*P+i], xr[1*P+i]);

  CeedVectorCreate(ceed, P, &In);
  CeedVectorSetArray(In, CEED_MEM_HOST, CEED_USE_POINTER, in);
  CeedVectorCreate(ceed, Q, &Out);
  CeedVectorSetValue(Out, 0);
  CeedVectorCreate(ceed, Q, &Weights);
  CeedVectorSetValue(Weights, 0);

  CeedBasisApply(b, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, In, Out);
  CeedBasisApply(b, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, NULL, Weights);

  // Check values at quadrature points
  CeedVectorGetArrayRead(Out, CEED_MEM_HOST, &out);
  CeedVectorGetArrayRead(Weights, CEED_MEM_HOST, &weights);
  sum = 0;
  for (int i=0; i<Q; i++)
    sum += out[i]*weights[i];
  if (fabs(sum - 17./24.) > 1e-10)
    printf("%f != %f\n", sum, 17./24.);
  CeedVectorRestoreArrayRead(Out, &out);
  CeedVectorRestoreArrayRead(Weights, &weights);

  CeedVectorDestroy(&In);
  CeedVectorDestroy(&Out);
  CeedVectorDestroy(&Weights);
  CeedBasisDestroy(&b);
  CeedDestroy(&ceed);
  return 0;
}
