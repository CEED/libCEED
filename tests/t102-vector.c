/// @file
/// Test CeedVectorGetArrayRead state counter
/// \test Test CeedVectorGetArrayRead state counter
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x;
  CeedInt    len = 10;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  CeedVectorSetValue(x, 0.0);

  // Two read accesses should not generate an error
  {
    const CeedScalar *a, *b;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &a);
    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);

    CeedVectorRestoreArrayRead(x, &a);
    CeedVectorRestoreArrayRead(x, &b);
  }

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
