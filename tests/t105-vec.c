/// @file
/// Test CeedVector readers counter
/// \test Test CeedVector readers counter
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n;
  const CeedScalar *a;
  CeedScalar *b;

  CeedInit(argv[1], &ceed);
  n = 10;
  CeedVectorCreate(ceed, n, &x);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &a);

  // Write access with read access generate an error
  CeedVectorGetArray(x, CEED_MEM_HOST, &b);

  CeedVectorRestoreArrayRead(x, &a);
  CeedVectorRestoreArray(x, &b);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
