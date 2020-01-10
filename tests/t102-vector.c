/// @file
/// Test CeedVectorGetArrayRead state counter
/// \test Test CeedVectorGetArrayRead state counter
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n;
  const CeedScalar *a, *b;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);

  // Two read accesses should not generate an error
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &a);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);

  CeedVectorRestoreArrayRead(x, &a);
  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
