/// @file
/// Test Collocated Grad calculated matches basis with Lobatto points
/// \test Test Collocated Grad calculated matches basis with Lobatto points
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedInt P=4, Q=4;
  CeedScalar collo_grad_1d[Q*Q], collo_grad_1d_2[(P+2)*(P+2)];
  CeedBasis b;

  CeedInit(argv[1], &ceed);

  // Already collocated, GetCollocatedGrad will return grad_1d
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS_LOBATTO, &b);
  CeedBasisGetCollocatedGrad(b, collo_grad_1d);

  for (int i=0; i<Q; i++) {
    fprintf(stdout, "%12s[%d]:", "collograd1d", i);
    for (int j=0; j<Q; j++) {
      if (fabs(collo_grad_1d[j+Q*i]) <= 100.*CEED_EPSILON) collo_grad_1d[j+Q*i] = 0;
      fprintf(stdout, "\t% 12.8f", collo_grad_1d[j+Q*i]);
    }
    fprintf(stdout, "\n");
  }
  CeedBasisDestroy(&b);

  // Q = P, not already collocated
  CeedBasisCreateTensorH1Lagrange(ceed, 1,  1, P, Q, CEED_GAUSS, &b);
  CeedBasisGetCollocatedGrad(b, collo_grad_1d);

  for (int i=0; i<Q; i++) {
    fprintf(stdout, "%12s[%d]:", "collograd1d", i);
    for (int j=0; j<Q; j++) {
      if (fabs(collo_grad_1d[j+Q*i]) <= 100.*CEED_EPSILON) collo_grad_1d[j+Q*i] = 0;
      fprintf(stdout, "\t% 12.8f", collo_grad_1d[j+Q*i]);
    }
    fprintf(stdout, "\n");
  }
  CeedBasisDestroy(&b);

  // Q = P + 2, not already collocated
  CeedBasisCreateTensorH1Lagrange(ceed, 1,  1, P, P+2, CEED_GAUSS, &b);
  CeedBasisGetCollocatedGrad(b, collo_grad_1d_2);

  for (int i=0; i<P+2; i++) {
    fprintf(stdout, "%12s[%d]:", "collograd1d", i);
    for (int j=0; j<P+2; j++) {
      if (fabs(collo_grad_1d_2[j+(P+2)*i]) <= 100.*CEED_EPSILON) collo_grad_1d_2[j+
            (P+2)*i] = 0;
      fprintf(stdout, "\t% 12.8f", collo_grad_1d_2[j+(P+2)*i]);
    }
    fprintf(stdout, "\n");
  }
  CeedBasisDestroy(&b);

  CeedDestroy(&ceed);
  return 0;
}
