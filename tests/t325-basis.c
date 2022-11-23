/// @file
/// Test grad transpose with a 2D Simplex non-tensor H1 basis
/// \test Test grad transposewith a 2D Simplex non-tensor H1 basis
#include <ceed.h>
#include <math.h>

#include "t320-basis.h"

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        In, Out;
  const CeedInt     P = 6, Q = 4, dim = 2, num_comp = 3;
  CeedBasis         b;
  CeedScalar        q_ref[dim * Q], q_weight[Q];
  CeedScalar        interp[P * Q], grad[dim * P * Q];
  const CeedScalar *out;
  CeedScalar        colsum[P], *in;

  buildmats(q_ref, q_weight, interp, grad);

  CeedInit(argv[1], &ceed);

  for (int i = 0; i < P; i++) {
    colsum[i] = 0;
    for (int j = 0; j < Q * dim; j++) {
      colsum[i] += grad[i + j * P];
    }
  }

  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, num_comp, P, Q, interp, grad, q_ref, q_weight, &b);

  CeedVectorCreate(ceed, Q * dim * num_comp, &In);
  CeedVectorGetArrayWrite(In, CEED_MEM_HOST, &in);
  for (int d = 0; d < dim; d++) {
    for (int n = 0; n < num_comp; n++) {
      for (int q = 0; q < Q; q++) in[q + (n + d * num_comp) * Q] = n * 1.0;
    }
  }
  CeedVectorRestoreArray(In, &in);
  CeedVectorCreate(ceed, P * num_comp, &Out);
  CeedVectorSetValue(Out, 0);

  CeedBasisApply(b, 1, CEED_TRANSPOSE, CEED_EVAL_GRAD, In, Out);

  // Check values at quadrature points
  CeedVectorGetArrayRead(Out, CEED_MEM_HOST, &out);
  for (int p = 0; p < P; p++) {
    for (int n = 0; n < num_comp; n++) {
      if (fabs(n * colsum[p] - out[p + n * P]) > 100. * CEED_EPSILON) printf("[%" CeedInt_FMT "] %f != %f\n", p, out[p + n * P], n * colsum[p]);
    }
  }
  CeedVectorRestoreArrayRead(Out, &out);

  CeedVectorDestroy(&In);
  CeedVectorDestroy(&Out);
  CeedBasisDestroy(&b);
  CeedDestroy(&ceed);
  return 0;
}
