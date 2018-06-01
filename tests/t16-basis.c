// Test Colocated Grad calculated matches basis with Lobatto points
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedInt P=4, Q=4;
  CeedScalar colograd1d[Q*Q];
  CeedBasis b;

  CeedInit(argv[1], &ceed);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS_LOBATTO, &b);
  CeedBasisGetColocatedGrad(b, colograd1d);

  CeedBasisView(b, stdout);
  for (int i=0; i<Q; i++) {
    fprintf(stdout, "%12s[%d]:", "colograd1d", i);
    for (int j=0; j<Q; j++) {
      if (fabs(colograd1d[j+Q*i]) <= 1E-14) colograd1d[j+Q*i] = 0;
      fprintf(stdout, "\t% 12.8f", colograd1d[j+Q*i]);
    }
    fprintf(stdout, "\n");
  }
  CeedBasisDestroy(&b);

  CeedBasisCreateTensorH1Lagrange(ceed, 1,  1, P, Q, CEED_GAUSS, &b);
  CeedBasisGetColocatedGrad(b, colograd1d);

  CeedBasisView(b, stdout);
  for (int i=0; i<Q; i++) {
    fprintf(stdout, "%12s[%d]:", "colograd1d", i);
    for (int j=0; j<Q; j++) {
      if (fabs(colograd1d[j+Q*i]) <= 1E-14) colograd1d[j+Q*i] = 0;
      fprintf(stdout, "\t% 12.8f", colograd1d[j+Q*i]);
    }
    fprintf(stdout, "\n");
  }
  CeedBasisDestroy(&b);

  CeedDestroy(&ceed);
  return 0;
}
