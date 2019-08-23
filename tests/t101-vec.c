/// @file
/// Test CeedVectorSetValue
/// \test Test CeedVectorSetValue
#include <ceed.h>

static int CheckValues(Ceed ceed, CeedVector x, CeedScalar value) {
  const CeedScalar *b;
  CeedInt n;
  CeedVectorGetLength(x, &n);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i=0; i<n; i++) {
    if (b[i] != value)
      // LCOV_EXCL_START
      printf("Error reading array b[%d] = %f",i,
             (double)b[i]);
      // LCOV_EXCL_STOP
  }
  CeedVectorRestoreArrayRead(x, &b);
  return 0;
}

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n;
  CeedScalar a[10];
  const CeedScalar *b;

  CeedInit(argv[1], &ceed);
  n = 10;
  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i=0; i<n; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i=0; i<n; i++)
    if (b[i] != 10+i)
      // LCOV_EXCL_START
      printf("Error reading array b[%d] = %f",i,
             (double)b[i]);
      // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorSetValue(x, 3.0);
  CheckValues(ceed, x, 3.0);
  CeedVectorDestroy(&x);

  CeedVectorCreate(ceed, n, &x);
  CeedVectorSetValue(x, 5.0); // Set value before setting or getting the array
  CheckValues(ceed, x, 5.0);
  CeedVectorDestroy(&x);

  CeedDestroy(&ceed);
  return 0;
}
