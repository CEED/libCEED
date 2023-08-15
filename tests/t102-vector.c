/// @file
/// Test CeedVectorGetArrayRead state counter
/// \test Test CeedVectorGetArrayRead state counter

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x;
  CeedInt    len = 10;

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

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
