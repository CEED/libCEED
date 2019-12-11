/// @file
/// Test viewing a vector
/// \test Test viewing a vector
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n;
  CeedScalar a[10];

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i=0; i<n; i++)
    a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  CeedVectorView(x, "%12.8f", stdout);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
